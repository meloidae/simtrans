from typing import Dict, Optional, Callable
from omegaconf import DictConfig
import torch
import torch.nn.functional as F

from .reinforce_model import ReinforceModel
from ..transformer_encoder_rnn_decoder import TransformerEncoderRNNDecoder
from ..utils import get_key_padding_mask
from ..utils import label_smoothed_cross_entropy

class TransformerEncoderRNNDecoderReinforce(TransformerEncoderRNNDecoder, ReinforceModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        assert cfg.model.baseline == 'self', f'Unimplemented baseline option {cfg.model.baseline}'

        # TODO implement baseline estimator?
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
        batch_size, max_tgt_len = tgt_ids.size()
        max_tgt_len -= 1
        max_src_len = src_ids.size(1)
        device = src_ids.device
        use_ce = lam > 0.0
        use_rl = lam < 1.0

        use_bse = self.bse is not None
        # Prepare masks
        src_pad_mask = get_key_padding_mask(src_lens)  # [batch_size, src_len]
        src_mask = src_pad_mask[:, None, None, :]  # [batch_size, 1, 1, src_len]

        # Encoder block
        src_input = self._token_dropout(src_ids, lens=src_lens, tok_ids=tok_ids)
        src_input = self._embed_input(src_input, embedding=self.src_embed)
        enc_output = self.encoder(src_input=src_input,
                                  src_mask=src_mask,
                                  return_attn=False)

        # Cross-entropy
        if use_ce:
            ce_output = self.cross_entropy(enc_output=enc_output['output'],
                                           tgt_ids=tgt_ids,
                                           tgt_lens=tgt_lens,
                                           cross_mask=src_mask.expand(-1, -1, max_tgt_len, -1),
                                           tok_ids=tok_ids,
                                           smoothing_eps=smoothing_eps)
            ce_loss = ce_output['loss']
        else:
            ce_loss = 0.0


        # REINFORCE
        if use_rl:
            cross_mask = src_mask.expand(-1, -1, max_output_len, -1)  # [batch_size, 1, tgt_len, src_len]
            sample_output = self.decode(method='sample',
                                        enc_output=enc_output['output'],
                                        cross_mask=cross_mask,
                                        tok_ids=tok_ids,
                                        max_output_len=max_output_len,
                                        return_attn=False,
                                        return_state=use_bse,
                                        return_proj=True)
            out_ids = sample_output['output_ids']
            out_ids = out_ids[:, 1:]  # remove BOS
            if use_bse:
                out_states = sample_output['output_states']
            else:
                out_states = None
            out_projs = sample_output['output_projs']
            out_lens = sample_output['output_lens']

            # Get reward
            if reward_type == 'alignment':
                rewards = reward_fn(src_ids, out_ids)  # [batch_size]
            elif reward_type in ['xribes', 'kendall']:
                rewards = reward_fn(src_ids, out_ids)
            else:
                tgt_ids = tgt_ids[:, 1:].cpu().numpy()  # Exclude BOS
                tgt_lens = (tgt_lens - 1).cpu().numpy()
                rewards = reward_fn(out_ids.cpu().numpy(), out_lens.cpu().numpy(), tgt_ids, tgt_lens)
            rewards = rewards.to(device)

            if use_bse:
                baseline = torch.sigmoid(self.bse(out_states.detach()))  # [batch_size, max_len, 1]
            else:  # Self-crtic
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
                    baseline = baseline[:, None, None]  # [batch_size, 1, 1]

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
            out_lens = None
            out_ids = None
            rl_loss = 0.0
            bse_loss = 0.0


        result = {}
        result['ce_loss'] = ce_loss
        result['rl_loss'] = rl_loss
        result['bse_loss'] = bse_loss
        if use_rl:
            result['output_len'] = out_lens
            result['output_ids'] = out_ids
        return result


    def _teacher_forcing_rl(self,
                            src_ids: torch.Tensor,
                            src_lens: torch.Tensor,
                            tgt_ids: torch.Tensor,
                            tgt_lens: torch.Tensor,
                            teacher_ids: torch.Tensor,
                            teacher_lens: torch.Tensor,
                            tok_ids: Dict[str, int],
                            max_output_len: int,
                            smoothing_eps: float,
                            lam: float,
                            reward_fn: Callable,
                            reward_type: str,
                            reward_boost: float) -> Dict:
        batch_size, max_tgt_len = tgt_ids.size()
        max_tgt_len -= 1
        max_src_len = src_ids.size(1)
        device = src_ids.device
        use_ce = lam > 0.0
        use_rl = lam < 1.0

        use_bse = self.bse is not None
        # Prepare masks
        src_pad_mask = get_key_padding_mask(src_lens)  # [batch_size, src_len]
        src_mask = src_pad_mask[:, None, None, :]  # [batch_size, 1, 1, src_len]

        # Encoder block
        src_input = self._token_dropout(src_ids, lens=src_lens, tok_ids=tok_ids)
        src_input = self._embed_input(src_input, embedding=self.src_embed)
        enc_output = self.encoder(src_input=src_input,
                                  src_mask=src_mask,
                                  return_attn=False)

        # Cross-entropy
        if use_ce:
            ce_output = self.cross_entropy(enc_output=enc_output['output'],
                                           tgt_ids=tgt_ids,
                                           tgt_lens=tgt_lens,
                                           cross_mask=src_mask.expand(-1, -1, max_tgt_len, -1),
                                           tok_ids=tok_ids,
                                           smoothing_eps=smoothing_eps)
            ce_loss = ce_output['loss']
        else:
            ce_loss = 0.0

        # REINFORCE
        if use_rl:
            cross_mask = src_mask.expand(-1, -1, max_output_len, -1)  # [batch_size, 1, tgt_len, src_len]
            sample_output, sample_states = self.teacher_forcing(enc_output=enc_output['output'],
                                                                tgt_ids=teacher_ids,
                                                                tgt_lens=teacher_lens,
                                                                cross_mask=cross_mask,
                                                                tok_ids=tok_ids)
            out_ids = teacher_ids
            out_ids = out_ids[:, 1:]  # remove BOS
            if use_bse:
                out_states = sample_states
            else:
                out_states = None
            out_projs = sample_output
            out_lens = teacher_lens - 1

            # Get reward
            if reward_type == 'alignment':
                rewards = reward_fn(src_ids, out_ids)  # [batch_size]
            elif reward_type in ['xribes', 'kendall']:
                rewards = reward_fn(src_ids, out_ids)
            else:
                tgt_ids = tgt_ids[:, 1:].cpu().numpy()  # Exclude BOS
                tgt_lens = (tgt_lens - 1).cpu().numpy()
                rewards = reward_fn(out_ids.cpu().numpy(), out_lens.cpu().numpy(), tgt_ids, tgt_lens)
            rewards = rewards.to(device)
            rewards += reward_boost

            if use_bse:
                baseline = torch.sigmoid(self.bse(out_states.detach()))  # [batch_size, max_len, 1]
            else:  # Self-crtic
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
                    baseline = baseline[:, None, None]  # [batch_size, 1, 1]

            # Get reward diff and its detached version
            rewards = rewards[:, None, None]
            reward_diff = rewards - baseline  # [batch_size, max_len, 1]
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
            out_lens = None
            out_ids = None
            rl_loss = 0.0
            bse_loss = 0.0

        result = {}
        result['ce_loss'] = ce_loss
        result['rl_loss'] = rl_loss
        result['bse_loss'] = bse_loss
        if use_rl:
            result['projs'] = out_projs
            result['output_len'] = out_lens
            result['reward'] = rewards
            result['baseline'] = baseline
            result['greedy_ids'] = greedy_ids
        return result


    def forward(self,
                mode: str,
                src_ids: torch.Tensor,
                src_lens: torch.Tensor,
                tok_ids: Dict[str, int],
                tgt_ids: Optional[torch.Tensor] = None,
                tgt_lens: Optional[torch.Tensor] = None,
                teacher_ids: Optional[torch.Tensor] = None,
                teacher_lens: Optional[torch.Tensor] = None,
                smoothing_eps: float = 0.1,
                max_output_len: int = 256,
                lam: float = 0.1,
                reward_fn: Optional[Callable] = None,
                reward_type: Optional[str] = None,
                reward_boost: float = 0.0,
                decode: str = 'greedy',
                return_attn: bool = False) -> Dict:

        if mode == 'ce_only':
            result = self._train(src_ids=src_ids,
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
            result = self._infer(src_ids=src_ids,
                                 src_lens=src_lens,
                                 wait_k=0,
                                 max_output_len=max_output_len,
                                 tok_ids=tok_ids,
                                 decode=decode,
                                 return_attn=return_attn)
        elif mode == 'teacher_forcing_rl':
            result = self._teacher_forcing_rl(src_ids=src_ids,
                                              src_lens=src_lens,
                                              tgt_ids=tgt_ids,
                                              tgt_lens=tgt_lens,
                                              teacher_ids=teacher_ids,
                                              teacher_lens=teacher_lens,
                                              max_output_len=max_output_len,
                                              tok_ids=tok_ids,
                                              smoothing_eps=smoothing_eps,
                                              lam=lam,
                                              reward_fn=reward_fn,
                                              reward_type=reward_type,
                                              reward_boost=reward_boost)
        else:
            assert False, f'Unknown mode: {mode}'

        return result
