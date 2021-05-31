from typing import Optional, Dict, Callable, Tuple, Iterator, Iterable
import itertools

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from .reinforce_model import ReinforceModel
from ..wait_k_transformer import WaitKTransformer
from .transformer_encoder_rnn_decoder_reinforce import TransformerEncoderRNNDecoderReinforce
from ..utils import get_key_padding_mask, get_square_subseq_mask
from ..utils import label_smoothed_cross_entropy


class SimpleBaselineEstimator(nn.Module):
    """
    Baseline estimator
    """
    def __init__(self, d_model: int, use_bias: bool) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features=d_model, out_features=1, bias=use_bias)

        self.init_weights()

    def init_weights(self, mean: float = 0.0):
        std = (2.0 / self.proj.in_features) ** 0.5
        nn.init.normal_(self.proj.weight, mean=mean, std=std)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x):
        output = self.proj(x)
        return output

class TransformerReinforce(WaitKTransformer, ReinforceModel):
    """
    Transformer with REINFORCE
    """
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        assert cfg.model.baseline in ['linear', 'self'], f'Unknown baseline opiton: {cfg.model.baseline}'
        if cfg.model.baseline == 'linear':
            self.bse = SimpleBaselineEstimator(d_model=cfg.model.d_model,
                                               use_bias=cfg.model.bias)
        elif cfg.model.baseline == 'self':
            self.bse = None

    def _reinforce(self,
                   src_ids: torch.Tensor,
                   src_lens: torch.Tensor,
                   tgt_ids: torch.Tensor,
                   tgt_lens: torch.Tensor,
                   max_output_len: int,
                   tok_ids: Dict[str, int],
                   smoothing_eps: float,
                   lam: float,
                   reward_fn: Callable,
                   reward_type: str) -> Dict:

        use_ce = lam > 0.0
        use_rl = lam < 1.0
        use_bse = self.bse is not None
        batch_size, max_tgt_len = tgt_ids.size()
        max_tgt_len -= 1
        max_src_len = src_ids.size(1)
        device = src_ids.device

        # Prepare masks
        src_pad_mask = get_key_padding_mask(src_lens)  # [batch_size, src_len]
        src_mask = src_pad_mask[:, None, None, :]  # [batch_size, 1, 1, src_len]
        cross_mask = src_mask  # [batch_size, 1, 1, src_len]

        tgt_pad_mask = get_key_padding_mask(tgt_lens - 1)  # [batch_size, tgt_len]
        tgt_subseq_mask = get_square_subseq_mask(max_len=max_tgt_len,
                                                 device=device)  # [tgt_len, tgt_len]
        tgt_mask = tgt_pad_mask[:, None, :] | tgt_subseq_mask[None, :, :]  # [batch_size, tgt_len, tgt_len]
        tgt_mask = tgt_mask[:, None, :, :]  # [batch_size, 1, src_len, src_len]

        # Encoder block
        src_input = self._token_dropout(src_ids, lens=src_lens, tok_ids=tok_ids)
        src_input = self._embed_input(src_input, embedding=self.src_embed)
        enc_output = self.encoder(src_input=src_input,
                                  src_mask=src_mask,
                                  return_attn=False)

        if use_ce:
            # Cross-entropy
            tgt_input = tgt_ids[:, :-1]  # [batch_size, tgt_len - 1] (Exclude eos)
            tgt_input = self._token_dropout(tgt_input, lens=tgt_lens - 1, tok_ids=tok_ids)
            tgt_input = self._embed_input(tgt_input, embedding=self.tgt_embed)
            dec_output = self.decoder(tgt_input=tgt_input,
                                      enc_output=enc_output['output'],
                                      tgt_mask=tgt_mask,
                                      enc_mask=cross_mask,
                                      return_attn=False)
            # Project to target vocab, ouput = [tgt_len, batch_size, vocab_size]
            output = self.tgt_proj(dec_output['output'])

            # Prepare tgt labels for calculating loss (target is shifted left to exclude bos)
            tgt_label_mask = get_key_padding_mask(lens=tgt_lens - 1, max_len=max_tgt_len) == 0
            tgt_label = tgt_ids[:, 1:]

            loss = label_smoothed_cross_entropy(model_outputs=output,
                                                tgt_labels=tgt_label,
                                                mask=tgt_label_mask,
                                                smoothing_eps=smoothing_eps,
                                                reduce=True)
            ce_loss = loss
        else:
            ce_loss = 0.0

        # REINFORCE
        if use_rl:
            # Sample outputs
            sample_output = self.decode(method='sample',
                                        enc_output=enc_output['output'],
                                        cross_mask=cross_mask,
                                        tok_ids=tok_ids,
                                        max_output_len=max_output_len,
                                        return_attn=False,
                                        return_state=use_bse,
                                        return_proj=True)
            out_ids = sample_output['output_ids']
            if use_bse:
                out_states = sample_output['output_states']
            else:
                out_states = None
            out_projs = sample_output['output_projs']
            out_lens = sample_output['output_lens']

            out_ids = out_ids[:, 1:]  # out_ids contains BOS
            if reward_type == 'alignment':
                rewards = reward_fn(src_ids, out_ids)  # [batch_size]
            elif reward_type in ['xribes', 'kendall']:
                rewards = reward_fn(src_ids, out_ids)
            else:
                tgt_ids = tgt_ids[:, 1:].cpu().numpy()  # Exclude BOS
                tgt_lens = (tgt_lens - 1).cpu().numpy()
                rewards = reward_fn(out_ids.cpu().numpy(), out_lens.cpu().numpy(), tgt_ids, tgt_lens)
            rewards = rewards.to(device)

            if out_states is not None:  # Use baseline score estimator
                # Normalize baseline to [0, 1], make sure to detach decoder states
                baseline = torch.sigmoid(self.bse(out_states.detach()))  # [batch_size, max_len, 1]
            else:  # Self-critic
                with torch.no_grad():
                    greedy_output = self.decode(method='greedy',
                                                enc_output=enc_output['output'],
                                                cross_mask=cross_mask,
                                                tok_ids=tok_ids,
                                                max_output_len=max_output_len,
                                                return_attn=False,
                                                return_state=False,
                                                return_proj=False)
                    greedy_ids = greedy_output['output_ids']
                    if reward_type == 'alignment':
                        baseline = reward_fn(src_ids, greedy_ids)  # [batch_size]
                    elif reward_type in ['xribes', 'kendall']:
                        baseline = reward_fn(src_ids, greedy_ids)  # [batch_size]
                    else:
                        greedy_ids = greedy_ids.cpu().numpy()
                        greedy_lens = greedy_output['output_lens'].cpu().numpy()
                        # tgt_ids & tgt_lens should have already become numpy array
                        baseline = reward_fn(greedy_ids, greedy_lens, tgt_ids, tgt_lens)


                    baseline = baseline.to(device)
                    baseline = baseline[:, None, None]

            # Get reward diff and its detached version
            reward_diff = rewards[:, None, None] - baseline  # [batch_size, max_len, 1]
            reward_diff_detached = reward_diff.detach()  # [batch_size, max_len, 1]
            sample_mask = get_key_padding_mask(out_lens, pad_with_false=True)[:, :, None]  # [batch_size, max_len, 1]

            # Calculate RL loss
            log_probs = F.log_softmax(out_projs, dim=2)  # [batch_size, max_len, vocab_size]
            sampled_log_probs = log_probs.gather(dim=2, index=out_ids[:, :, None])  # [batch_size, max_len, 1]
            rl_loss = torch.sum(torch.masked_select(-sampled_log_probs.mul(reward_diff_detached), mask=sample_mask))

            if use_bse:
                # Calculate loss for baseline estimator
                bse_loss = torch.sum(torch.masked_select(reward_diff.mul(reward_diff), mask=sample_mask))
            else:
                bse_loss = 0.0
        else:
            rl_loss = 0.0
            bse_loss = 0.0
            out_lens = None
            out_ids = None

        result = {}
        result['ce_loss'] = ce_loss
        result['rl_loss'] = rl_loss
        result['bse_loss'] = bse_loss
        if use_rl:
            result['output_len'] = out_lens
            result['output_ids'] = out_ids
        return result


    def forward(self,
                mode: str,
                src_ids: torch.Tensor,
                src_lens: torch.Tensor,
                tok_ids: Dict[str, int],
                tgt_ids: Optional[torch.Tensor] = None,
                tgt_lens: Optional[torch.Tensor] = None,
                smoothing_eps: float = 0.1,
                max_output_len: int = 256,
                lam: float = 0.1,
                reward_fn: Optional[Callable] = None,
                reward_type: Optional[str] = None,
                decode: str = 'greedy',
                return_attn: bool = False) -> Dict:

        if mode == 'ce_only':
            result = super()._train(src_ids=src_ids,
                                    src_lens=src_lens,
                                    tgt_ids=tgt_ids,
                                    tgt_lens=tgt_lens,
                                    wait_k=0,
                                    tok_ids=tok_ids,
                                    smoothing_eps=smoothing_eps)
        elif mode == 'rl':
            result = self._reinforce(src_ids=src_ids,
                                     src_lens=src_lens,
                                     tgt_ids=tgt_ids,
                                     tgt_lens=tgt_lens,
                                     max_output_len=max_output_len,
                                     tok_ids=tok_ids,
                                     smoothing_eps=smoothing_eps,
                                     lam=lam,
                                     reward_fn=reward_fn,
                                     reward_type=reward_type)
        elif mode == 'infer':
            assert decode == 'greedy', f'Only greedy decoding is available for now'
            result = super()._infer(src_ids=src_ids,
                                    src_lens=src_lens,
                                    wait_k=0,
                                    max_output_len=max_output_len,
                                    tok_ids=tok_ids,
                                    decode=decode,
                                    return_attn=return_attn)
        else:
            assert False, f'Unknown mode: {mode}'

        return result


def build_model(cfg: DictConfig) -> ReinforceModel:
    model_type = cfg.model.type
    assert model_type in ['transformer_reinforce', 'transformer_enc_rnn_dec_reinforce'], f'Unimplemented model type {model_type}'

    if model_type == 'transformer_reinforce':
        model = TransformerReinforce(cfg)
    else:
        model = TransformerEncoderRNNDecoderReinforce(cfg)
    return model


