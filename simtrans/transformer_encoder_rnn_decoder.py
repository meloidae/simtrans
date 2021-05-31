from typing import Dict, Tuple

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from .wait_k_transformer import WaitKTransformer
from .utils import get_key_padding_mask, get_wait_k_mask, get_square_subseq_mask
from .utils import label_smoothed_cross_entropy


class TransformerEncoderRNNDecoder(WaitKTransformer):
    """
    Encoder-Decoder model with transformer encoder and rnn decoder
    """
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        del self.decoder
        self.decoder = nn.LSTM(input_size=cfg.model.d_model * 2,  # Do input feeding
                               hidden_size=cfg.model.d_model,
                               num_layers=cfg.model.n_dec,
                               bias=cfg.model.bias,
                               bidirectional=False,
                               batch_first=True,
                               dropout=cfg.model.dropout)
        self.pre_proj = nn.Linear(in_features=cfg.model.d_model * 2,
                                  out_features=cfg.model.d_model,
                                  bias=cfg.model.bias)
        self.dropout = cfg.model.dropout

        self.rnn_dec_init_weights()

    def rnn_dec_init_weights(self, mean: float = 0.0):
        # Initialize pre_proj
        std = (2.0 / self.pre_proj.in_features) ** 0.5
        nn.init.normal_(self.pre_proj.weight, mean=mean, std=std)
        if self.pre_proj.bias is not None:
            nn.init.constant_(self.pre_proj.bias, 0.0)

        # Initalize lstm
        hidden_size = self.decoder.hidden_size
        std = (2.0 / hidden_size) ** 0.5
        for name, p in self.decoder.named_parameters():
            if 'bias' in name:
                nn.init.constant_(p, 0.0)
                # Set a non-zero bias for forget gate
                if 'hh' in name:
                    p.data[hidden_size: hidden_size * 2].fill_(1.0)
            elif 'weight' in name:
                nn.init.normal_(p, mean=mean, std=std)


    def _attention(self,
                   enc_output: torch.Tensor,
                   dec_output: torch.Tensor,
                   cross_mask: torch.Tensor):
        # Expect enc_output to be transposed, so [batch_size, d_model, src_len]
        # Calculate attention weights
        attn_weights = torch.bmm(dec_output, enc_output)  # [batch_size, 1, src_len]
        attn_weights.masked_fill_(cross_mask, -float('inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)

        return attn_weights


    def teacher_forcing(self,
                        enc_output: torch.Tensor,
                        tgt_ids: torch.Tensor,
                        tgt_lens: torch.Tensor,
                        cross_mask: torch.Tensor,
                        tok_ids: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_tgt_len = tgt_ids.size()
        max_tgt_len -= 1
        device = tgt_ids.device

        tgt_input = tgt_ids[:, :-1]  # [batch_size, tgt_len] (Exclude EOS)
        tgt_input = self._token_dropout(tgt_input, lens=tgt_lens - 1, tok_ids=tok_ids)
        tgt_input = self._embed_input(tgt_input, embedding=self.tgt_embed, use_pe=False)  # [batch_size, tgt_len, d_model]
        prev_final_state = torch.zeros(size=(batch_size, 1, tgt_input.size(2)), dtype=torch.float32, device=device)
        internal_state = None
        stacked_lstm = self.decoder.num_layers != 1
        states = []
        enc_output_trans = enc_output.transpose(1, 2)  # [batch_size, d_model, src_len]

        for i in range(max_tgt_len):
            dec_input = torch.cat((tgt_input[:, i: i + 1, :], prev_final_state), dim=2)  # [batch_size, 1, d_model * 2]
            dec_output, internal_state = self.decoder(dec_input, internal_state)
            if stacked_lstm:  # Sum up hidden states if stacked lstm
                all_states = internal_state[0]  # [n_layers, batch_size, d_model]
                dec_output = torch.sum(all_states, dim=0)  # [batch_size, d_model]
                dec_output = dec_output[:, None, :]  # [batch_size, 1, d_model]

            attn_weights = self._attention(enc_output=enc_output_trans,
                                           dec_output=dec_output,
                                           cross_mask=cross_mask[:, :, i, :])  # [batch_size, 1, src_len]

            context = torch.bmm(attn_weights, enc_output)  # [batch_size, 1, d_model]
            final_state = torch.cat((dec_output, context), dim=-1)  # [batch_size, 1, d_model * 2]
            final_state = F.dropout(final_state, p=self.dropout, training=self.training)
            final_state = self.pre_proj(final_state)  # [batch_size, 1, d_model]
            final_state = torch.tanh(final_state)
            final_state = F.dropout(final_state, p=self.dropout, training=self.training)

            states.append(final_state)
            prev_final_state = final_state  # [batch_size, 1, d_model]

        all_states = torch.cat(states, dim=1)  # [batch_size, tgt_len, d_model]
        output = self.tgt_proj(all_states)  # [batch_size, tgt_len, vocab_size]

        return output, all_states


    def cross_entropy(self,
                      enc_output: torch.Tensor,
                      tgt_ids: torch.Tensor,
                      tgt_lens: torch.Tensor,
                      cross_mask: torch.Tensor,
                      tok_ids: Dict[str, int],
                      smoothing_eps: float) -> Dict:
        max_tgt_len = tgt_ids.size(1)
        max_tgt_len -= 1

        output, _ = self.teacher_forcing(enc_output=enc_output,
                                         tgt_ids=tgt_ids,
                                         tgt_lens=tgt_lens,
                                         cross_mask=cross_mask,
                                         tok_ids=tok_ids)

        # Prepare tgt labels for calculating loss (target is shifted left to exclude bos)
        tgt_label_mask = get_key_padding_mask(lens=tgt_lens - 1, max_len=max_tgt_len) == 0
        tgt_label = tgt_ids[:, 1:]

        loss = label_smoothed_cross_entropy(model_outputs=output,
                                            tgt_labels=tgt_label,
                                            mask=tgt_label_mask,
                                            smoothing_eps=smoothing_eps,
                                            reduce=True)

        result = {}
        result['loss'] = loss
        return result


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
            cross_mask = src_mask.expand(-1, -1, max_tgt_len, -1)  # [batch_size, 1, tgt_len, src_len]

        # Encoder block
        src_input = self._token_dropout(src_ids, lens=src_lens, tok_ids=tok_ids)
        src_input = self._embed_input(src_input, embedding=self.src_embed)
        enc_output = self.encoder(src_input=src_input,
                                  src_mask=src_mask,
                                  return_attn=False)
        enc_state = enc_output['output']  # [batch_size, src_len, d_model]
        enc_state_trans = enc_state.transpose(1, 2)  # [batch_size, d_model, src_len]

        # Decoder
        dec_output = self.cross_entropy(enc_output=enc_output['output'],
                                        tgt_ids=tgt_ids,
                                        tgt_lens=tgt_lens,
                                        cross_mask=cross_mask,
                                        tok_ids=tok_ids,
                                        smoothing_eps=smoothing_eps)

        result = {}
        result['loss'] = dec_output['loss']
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

        dec_cross_attn_list = [] if return_attn else None
        tgt_states = [] if return_state else None
        tgt_projs = [] if return_proj else None

        batch_size, _, _ = enc_output.size()
        max_tgt_len = max_output_len
        device = enc_output.device

        # Prepare first target
        tgt_ids = torch.full(size=(batch_size, 1),
                             fill_value=tok_ids['bos'],
                             dtype=torch.int64, device=device)
        tgt_input = self._embed_input(tgt_ids, embedding=self.tgt_embed, use_pe=False)  # [batch_size, 1, d_model]
        prev_final_state = torch.zeros(size=(batch_size, 1, tgt_input.size(2)), dtype=torch.float32, device=device)
        internal_state = None
        stacked_lstm = self.decoder.num_layers != 1

        eos_flags = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device)
        tgt_lens = torch.ones_like(eos_flags, dtype=torch.int64)

        enc_output_trans = enc_output.transpose(1, 2)  # [batch_size, d_model, src_len]

        # Decode loop
        for i in range(max_output_len):
            dec_input = torch.cat((tgt_input, prev_final_state), dim=2)  # [batch_size, 1, d_model * 2]
            dec_output, internal_state = self.decoder(dec_input, internal_state)
            if stacked_lstm:  # Sum up hidden states if stacked lstm
                all_states = internal_state[0]  # [n_layers, batch_size, d_model]
                dec_output = torch.sum(all_states, dim=0)  # [batch_size, d_model]
                dec_output = dec_output[:, None, :]  # [batch_size, 1, d_model]

            attn_weights = self._attention(enc_output=enc_output_trans,
                                           dec_output=dec_output,
                                           cross_mask=cross_mask[:, :, i, :])  # [batch_size, 1, src_len]
            context = torch.bmm(attn_weights, enc_output)  # [batch_size, 1, d_model]
            final_state = torch.cat((dec_output, context), dim=-1)  # [batch_size, 1, d_model * 2]
            final_state = F.dropout(final_state, p=self.dropout, training=self.training)
            final_state = self.pre_proj(final_state)  # [batch_size, 1, d_model]
            final_state = F.dropout(final_state, p=self.dropout, training=self.training)
            final_state = torch.tanh(final_state)

            output = self.tgt_proj(final_state)  # [batch_size, 1, vocab_size]
            prev_final_state = final_state  # [batch_size, 1, d_model]

            # Save attention, states & projections
            if dec_cross_attn_list is not None:
                dec_cross_attn_list.append(attn_weights)
            if tgt_states is not None:
                tgt_states.append(final_state)
            if tgt_projs is not None:
                tgt_projs.append(output)

            with torch.no_grad():
                if method == 'greedy':
                    _, output_ids = torch.max(output[:, 0, :], dim=1)  # [batch_size]
                elif method == 'sample':
                    output_dist = F.softmax(output[:, 0, :], dim=1)  # [batch_size, vocab_size]
                    output_ids = torch.multinomial(output_dist, num_samples=1, replacement=True)[:, 0]  # [batch_size]
                else:
                    assert False, f'Unimplemented decoding method: {method}'

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
                tgt_input = self._embed_input(append_ids[:, None],
                                              embedding=self.tgt_embed,
                                              use_pe=False)  # [batch_size, d_model]

        # Make sure that tgt lengths are shorter or equal to a maximum output length
        tgt_lens = torch.minimum(tgt_lens, 
                                 torch.full(size=(1,), 
                                            fill_value=max_tgt_len,
                                            dtype=torch.int64,
                                            device=device))  # [batch_size]

        result = {}
        if dec_cross_attn_list is not None:
            result['dec_cross_attn'] = dec_cross_attn_list
        if tgt_states is not None:
            tgt_states = torch.cat(tgt_states, dim=1)  # [batch_size, max_len, d_model]
            result['output_states'] = tgt_states
        if tgt_projs is not None:
            tgt_projs = torch.cat(tgt_projs, dim=1)  # [batch_size, max_len, vocab_size]
            result['output_projs'] = tgt_projs

        result['output_ids'] = tgt_ids
        result['output_lens'] = tgt_lens
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

        enc_self_attn_list = [] if return_attn else None

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
            cross_mask = src_mask.expand(-1, -1, max_tgt_len, -1)  # [batch_size, 1, tgt_len, src_len]

        # Encoder block
        src_input = self._token_dropout(src_ids, lens=src_lens, tok_ids=tok_ids)
        src_input = self._embed_input(src_input, embedding=self.src_embed)
        enc_output = self.encoder(src_input=src_input,
                                  src_mask=src_mask,
                                  return_attn=return_attn)  # [batch_size, src_len, d_model]
        if enc_self_attn_list is not None:
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
        result['output_len'] = dec_output['output_lens']
        if return_attn:
            result['enc_self_attn'] = enc_self_attn_list
            result['dec_cross_attn'] = dec_output['dec_cross_attn']
        return result


