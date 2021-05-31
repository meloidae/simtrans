from typing import Optional, Dict, Any

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerEncoder, TransformerDecoder
from .utils import get_key_padding_mask, get_square_subseq_mask, get_wait_k_mask
from .utils import label_smoothed_cross_entropy


class DistributedDataParallelPassthrough(nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_sinusoidal_positional_embedding(d_model: int, max_length: int = 5000):
    div_term = -(torch.arange(end=float(d_model)) // 2) * 2.0 / d_model
    div_term = torch.pow(10000.0, div_term).reshape(1, -1)
    pos = torch.arange(end=float(max_length)).reshape(-1, 1)
    pe = torch.matmul(pos, div_term)
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.sin(pe[:, 1::2])
    return pe


class WaitKTransformer(nn.Module):
    """
    Wait-k Transformer
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.scale = cfg.model.d_model ** 0.5
        self.tok_dropout = cfg.model.token_dropout

        self.src_embed = nn.Parameter(torch.Tensor(cfg.tokenizer.vocab_size, cfg.model.d_model))

        pos_enc = get_sinusoidal_positional_embedding(d_model=cfg.model.d_model)
        self.register_buffer('pe', pos_enc, persistent=False)

        if cfg.model.share_src_tgt_emb_weight:
            self.tgt_embed = self.src_embed
        else:
            self.tgt_embed = nn.Parameter(torch.Tensor(cfg.tokenizer.vocab_size, cfg.model.d_model))

        self.encoder = TransformerEncoder(d_model=cfg.model.d_model,
                                          d_ff=cfg.model.d_ff,
                                          n_head=cfg.model.n_head,
                                          n_layer=cfg.model.n_enc,
                                          prenorm=cfg.model.prenorm,
                                          activation=cfg.model.activation,
                                          dropout=cfg.model.dropout,
                                          use_bias=cfg.model.bias)

        self.decoder = TransformerDecoder(d_model=cfg.model.d_model,
                                          d_ff=cfg.model.d_ff,
                                          n_head=cfg.model.n_head,
                                          n_layer=cfg.model.n_dec,
                                          prenorm=cfg.model.prenorm,
                                          activation=cfg.model.activation,
                                          dropout=cfg.model.dropout,
                                          use_bias=cfg.model.bias)

        self.tgt_proj = nn.Linear(in_features=cfg.model.d_model,
                                  out_features=cfg.tokenizer.vocab_size,
                                  bias=not cfg.model.share_tgt_emb_proj_weight)

        if cfg.model.share_tgt_emb_proj_weight:
            self.tgt_proj.weight = self.tgt_embed

        self.init_weights(d_model=cfg.model.d_model, use_bias=not cfg.model.share_tgt_emb_proj_weight)


    def init_weights(self, d_model: int, use_bias: bool) -> None:
        src_tgt_embed_same = self.src_embed.data_ptr() == self.tgt_embed.data_ptr()
        tgt_embed_proj_same = self.tgt_embed.data_ptr() == self.tgt_proj.weight.data_ptr()
        all_embed_proj_same = src_tgt_embed_same and tgt_embed_proj_same

        if all_embed_proj_same:  # Only have to initalize src_embed
            nn.init.normal_(self.src_embed, mean=0.0, std=d_model ** -0.5)
        elif src_tgt_embed_same:  # Initialize src_embed and tgt_proj
            nn.init.normal_(self.src_embed, mean=0.0, std=d_model ** -0.5)
            nn.init.normal_(self.tgt_proj.weight, mean=0.0, std=d_model ** -0.5)
            if use_bias:
                nn.init.constant_(self.tgt_proj.bias, 0.0)
        elif tgt_embed_proj_same:  # Initialize src_embed and tgt_emb
            nn.init.normal_(self.src_embed, mean=0.0, std=d_model ** -0.5)
            nn.init.normal_(self.tgt_embed, mean=0.0, std=d_model ** -0.5)
        else:  # Initialize all
            nn.init.normal_(self.src_embed, mean=0.0, std=d_model ** -0.5)
            nn.init.normal_(self.tgt_embed, mean=0.0, std=d_model ** -0.5)
            nn.init.normal_(self.tgt_proj.weight, mean=0.0, std=d_model ** -0.5)
            if use_bias:
                nn.init.constant_(self.tgt_proj.bias, 0.0)


    def _token_dropout(self, x: torch.Tensor, lens: torch.Tensor, tok_ids: Dict[str, int]):
        if self.training and 0.0 < self.tok_dropout < 1.0:
            non_pad_mask = get_key_padding_mask(lens) == 0
            non_bos_mask = x != tok_ids['bos']
            dropout_mask = torch.rand(x.size(), device=x.device) <= self.tok_dropout
            dropout_mask = non_pad_mask & non_bos_mask & dropout_mask
            output = torch.masked_fill(x, mask=dropout_mask, value=tok_ids['unk'])
            return output
        else:
            return x


    def _embed_input(self, x: torch.Tensor, embedding: torch.Tensor, use_pe: bool = True):
        tok_embed = F.embedding(x, embedding) * self.scale  # [batch_size, len, d_model]
        if use_pe:
            pos_embed = self.pe[:x.size(1), :][None, :, :]  # [1, len, d_model]
            output = tok_embed + pos_embed
        else:
            output = tok_embed
        return output


    def _train(self,
               src_ids: torch.Tensor,
               src_lens: torch.Tensor,
               tgt_ids: torch.Tensor,
               tgt_lens: torch.Tensor,
               wait_k: int,
               tok_ids: Dict[str, int],
               smoothing_eps: float) -> Dict:
        batch_size, max_tgt_len = tgt_ids.size()
        max_tgt_len -= 1
        max_src_len = src_ids.size(1)
        device = src_ids.device

        # Prepare masks
        src_pad_mask = get_key_padding_mask(src_lens)  # [batch_size, src_len]
        if wait_k > 0:
            src_subseq_mask = get_square_subseq_mask(max_len=max_src_len,
                                                     device=device)  # [src_len, src_len]
            src_mask = src_pad_mask[:, None, :] | src_subseq_mask[None, :, :]  # [batch_size, src_len, src_len]
            src_mask = src_mask[:, None, :, :]  # [batch_size, 1, src_len, src_len]

            cross_wait_k_mask = get_wait_k_mask(q_len=max_tgt_len,
                                                k_len=max_src_len,
                                                wait_k=wait_k,
                                                device=device)  # [tgt_len, src_len]
            cross_mask = src_pad_mask[:, None, :] | cross_wait_k_mask[None, :, :]
            cross_mask = cross_mask[:, None, :, :]  # [batch_size, 1, tgt_len, src_len]
        else:
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

        # Decoder block
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

        result = {'loss': loss}
        return result


    def _infer(self,
               src_ids: torch.Tensor,
               src_lens: torch.Tensor,
               wait_k: int,
               max_output_len: int,
               tok_ids: Dict[str, int],
               decode: str,
               return_attn: bool) -> Dict:

        batch_size, max_src_len = src_ids.size()
        max_tgt_len = max_output_len
        device = src_ids.device

        if return_attn:
            enc_self_attn_list = []

        # Prepare masks
        src_pad_mask = get_key_padding_mask(src_lens)  # [batch_size, src_len]
        if wait_k > 0:
            src_subseq_mask = get_square_subseq_mask(max_len=max_src_len,
                                                     device=device)  # [src_len, src_len]
            src_mask = src_pad_mask[:, None, :] | src_subseq_mask[None, :, :]  # [batch_size, src_len, src_len]
            src_mask = src_mask[:, None, :, :]  # [batch_size, 1, src_len, src_len]
            cross_wait_k_mask = get_wait_k_mask(q_len=max_tgt_len,
                                         k_len=max_src_len,
                                         wait_k=wait_k,
                                         device=device)  # [max_tgt_len, src_len]
            cross_mask = src_pad_mask[:, None, :] | cross_wait_k_mask[None, :, :]
            cross_mask = cross_mask[:, None, :, :]  # [batch_size, 1, max_tgt_len, src_len]
        else:
            src_mask = src_pad_mask[:, None, None, :]  # [batch_size, 1, 1, src_len]
            cross_mask = src_mask  # [batch_size, 1, 1, src_len]

        # Encoder block
        src_input = self._embed_input(src_ids, embedding=self.src_embed)
        enc_output = self.encoder(src_input=src_input,
                                  src_mask=src_mask,
                                  return_attn=return_attn)  # [batch_size, src_len, d_model]
        if return_attn:
            enc_self_attn_list.append(enc_output['self_attn'])

        
        dec_output = self.decode(method='greedy',
                                 enc_output=enc_output['output'],
                                 cross_mask=cross_mask,
                                 tok_ids=tok_ids,
                                 max_output_len=max_output_len,
                                 return_attn=return_attn,
                                 return_state=False,
                                 return_proj=False)

        result = {}
        result['output'] = dec_output['output_ids']
        if return_attn:
            result['enc_self_attn'] = enc_self_attn_list
            result['dec_self_attn'] = dec_output['dec_self_attn']
            result['dec_cross_attn'] = dec_output['dec_cross_attn']
        return result


    def decode(self,
               method: str,
               enc_output: torch.Tensor,
               cross_mask: torch.Tensor,
               tok_ids: Dict[str, int],
               max_output_len: int,
               return_attn: bool,
               return_state: bool,
               return_proj: bool) -> Dict:
        if return_attn:
            dec_self_attn_list = []
            dec_cross_attn_list = []
        if return_state:
            tgt_states = []
        if return_proj:
            tgt_projs = []

        batch_size, _, _ = enc_output.size()
        max_tgt_len = max_output_len
        device = enc_output.device

        # Prepare first target and masks
        tgt_ids = torch.full(size=(batch_size, 1),
                             fill_value=tok_ids['bos'],
                             dtype=torch.int64, device=device)
        tgt_input = self._embed_input(tgt_ids, embedding=self.tgt_embed)
        tgt_subseq_mask = get_square_subseq_mask(max_len=max_tgt_len,
                                                 device=device)  # [max_tgt_len, max_tgt_len]
        tgt_mask = tgt_subseq_mask[None, None, :1, :1]

        eos_flags = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device)
        tgt_lens = torch.ones_like(eos_flags, dtype=torch.int64)

        for i in range(max_output_len):
            dec_output = self.decoder(tgt_input=tgt_input,
                                      enc_output=enc_output,
                                      tgt_mask=tgt_mask,
                                      enc_mask=cross_mask[:, :, :i + 1, :],
                                      return_attn=return_attn)

            # Project to target vocab, only deal with the current timestep
            dec_state = dec_output['output'][:, i, :]  # [batch_size, d_model]
            output = self.tgt_proj(dec_state)  # [batch_size, vocab_size]

            # Save attention, states & projections
            if return_attn:
                dec_self_attn_list.append(dec_output['self_attn'])
                dec_cross_attn_list.append(dec_output['cross_attn'])
            if return_state:
                tgt_states.append(dec_state)  # List of [batch_size, d_model]
            if return_proj:
                tgt_projs.append(output)  # List of [batch_size, vocab_size]

            with torch.no_grad():
                if method == 'greedy':
                    _, output_ids = torch.max(output, dim=1)  # [batch_size]
                elif method == 'sample':
                    output_dist = F.softmax(output, dim=1)
                    output_ids = torch.multinomial(output_dist, num_samples=1, replacement=True)[:, 0]  # [batch_size]
                else:
                    assert False, f'Unrecognized decoding method: {method}'

                # Examine output of current time step
                new_ids = output_ids  # [batch_size]
                new_eos = new_ids == tok_ids['eos']  # [batch_size]
                eos_flags = eos_flags | new_eos
                append_ids = new_ids.masked_fill(eos_flags, tok_ids['eos'])
                tgt_ids = torch.cat((tgt_ids, append_ids[:, None]), dim=1)  # [batch_size, i + 2]
                tgt_lens = tgt_lens.masked_fill(eos_flags == 0, i + 2)

            if int(eos_flags.sum()) == batch_size or i == max_output_len - 1:
                # Check termination conditions
                break
            else:
                append_input = self._embed_input(append_ids[:, None],
                                                 embedding=self.tgt_embed)  # [batch_size, 1, d_model]
                tgt_input = torch.cat((tgt_input, append_input), dim=1)  # [batch_size, i + 2, d_model]
                tgt_pad_mask = get_key_padding_mask(lens=tgt_lens, max_len=i + 2)  # [batch_size, i + 2]
                tgt_mask = tgt_pad_mask[:, None, :] | tgt_subseq_mask[None, :i + 2, :i + 2]
                tgt_mask = tgt_mask[:, None, :, :]  # [batch_size, 1, i + 2, i + 2]

        
        # Make sure that tgt lengths are shorter or equal to a maximum output length
        tgt_lens = torch.minimum(tgt_lens, 
                                 torch.full(size=(1,), 
                                            fill_value=max_tgt_len,
                                            dtype=torch.int64,
                                            device=device))  # [batch_size]

        result = {}
        if return_state:
            tgt_states = torch.stack(tgt_states, dim=1)  # [batch_size, max_len, d_model]
            result['output_states'] = tgt_states
        if return_proj:
            tgt_projs = torch.stack(tgt_projs, dim=1)  # [batch_size, max_len, vocab_size]
            result['output_projs'] = tgt_projs
        if return_attn:
            result['dec_self_attn'] = dec_self_attn_list
            result['dec_cross_attn'] = dec_cross_attn_list

        result['output_ids'] = tgt_ids
        result['output_lens'] = tgt_lens

        return result


    def forward(self,
                mode: str,
                src_ids: torch.Tensor,
                src_lens: torch.Tensor,
                wait_k: int,
                tok_ids: Dict[str, int],
                tgt_ids: Optional[torch.Tensor] = None,
                tgt_lens: Optional[torch.Tensor] = None,
                smoothing_eps: float = 0.1,
                max_output_len: int = 256,
                decode: str = 'greedy',
                return_attn: bool = False) -> Dict:
        if mode == 'train':
            result = self._train(src_ids=src_ids,
                                 src_lens=src_lens,
                                 tgt_ids=tgt_ids,
                                 tgt_lens=tgt_lens,
                                 wait_k=wait_k,
                                 tok_ids=tok_ids,
                                 smoothing_eps=smoothing_eps)
        elif mode == 'infer':
            result = self._infer(src_ids=src_ids,
                                 src_lens=src_lens,
                                 wait_k=wait_k,
                                 max_output_len=max_output_len,
                                 tok_ids=tok_ids,
                                 decode=decode,
                                 return_attn=return_attn)
        else:
            result = {}
            assert False, f'Unknown mode: {mode}'

        return result


