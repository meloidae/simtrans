from typing import Optional, Tuple, Dict
import torch
from torch import nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    """
    MultiheadAttention
    """
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 dropout: float,
                 use_bias: bool):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.use_bias = use_bias

        if self.d_model % self.n_head != 0:
            raise ValueError(f'd_model = {self.d_model} must be divisible by n_head = {self.n_head}')
        
        self.d_head = self.d_model // self.n_head
        self.scale = self.d_head ** - 0.5

        # Parameters for projecting q, k, v and output + optional bias
        self.proj_weights = nn.Parameter(torch.Tensor(4 * self.d_model, self.d_model))
        if self.use_bias:
            self.proj_biases = nn.Parameter(torch.Tensor(4 * self.d_model))

        self.init_weights()


    def init_weights(self, mean: float = 0.0, std_divisor_multiplier: float = 5.0):
        std = (2.0 / (std_divisor_multiplier * self.d_model)) ** 0.5
        nn.init.normal_(self.proj_weights, mean=mean, std=std)
        if self.use_bias:
            nn.init.constant_(self.proj_biases, 0.0)


    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.BoolTensor] = None,
                do_qkv_projection: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        def split_to_heads(x: torch.Tensor) -> torch.Tensor:
            batch_size, length, _ = x.size()
            x = x.reshape(batch_size, length, self.n_head, self.d_head)
            x = x.transpose(1, 2)  # [batch_size, n_head, length, d_head]
            x = x.reshape(batch_size * self.n_head, -1, self.d_head)  # [batch_size * n_head, length, d_head]
            return x

        if do_qkv_projection:  # This maybe False when trying out other attention scheme
            q, k, v = self._project_qkv(q, k, v)

        q = split_to_heads(q)
        k = split_to_heads(k)
        v = split_to_heads(v)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [batch_size * n_head, q_len, k_len]
        batch_size_times_n_head, q_len, k_len = attn_weights.size()
        batch_size = batch_size_times_n_head // self.n_head
        attn_weights = attn_weights.reshape(batch_size, self.n_head, q_len, k_len)

        # Apply mask that combines both padding and causal masks
        if mask is not None:
            # mask: [batch_size, n_head, q_len, k_len]
            attn_weights.masked_fill_(mask, -float('inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        _attn_weights = attn_weights.reshape(batch_size_times_n_head, q_len, k_len)
        output = torch.bmm(_attn_weights, v)
        output = output.reshape(batch_size, self.n_head, q_len, self.d_head)
        output = output.transpose(1, 2)
        output = output.reshape(batch_size, q_len, -1)  # [batch_size, q_len, d_model = n_head * d_head]
        output = self._project(output, start=3 * self.d_model)

        return output, attn_weights


    def _project_qkv(self,
                     q: torch.Tensor,
                     k: torch.Tensor,
                     v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Check q, k, v equality to eliminate some calculations
        qkv_same = q.data_ptr() == k.data_ptr() == v.data_ptr()
        kv_same = k.data_ptr() == v.data_ptr()

        if qkv_same:
            q, k, v = self._project(q, end=3 * self.d_model).chunk(3, dim=-1)
        elif kv_same:
            q = self._project(q, end=self.d_model)
            k, v = self._project(k, start=self.d_model, end=3 * self.d_model).chunk(2, dim=-1)
        else:
            q = self._project(q, end=self.d_model)
            k = self._project(k, start=self.d_model, end=2 * self.d_model)
            v = self._project(v, start=3 * self.d_model)

        return q, k, v

    
    def _project(self,
                 x: torch.Tensor,
                 start: int = 0,
                 end: Optional[int] = None) -> torch.Tensor:
        w = self.proj_weights[start: end, :]
        b = None if not self.use_bias else self.proj_biases[start: end]
        return F.linear(x, weight=w, bias=b)


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, use_bias: bool, activation: str):
        super().__init__()
        self.dropout = dropout
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_bias = use_bias

        if activation == 'relu':
            self.act_fn = F.relu
        elif activation == 'gelu':
            self.act_fn = F.gelu
        else:
            raise ValueError("Activation function must be either 'relu' or 'gelu'")

        self.in_proj = nn.Linear(self.d_model, self.d_ff, bias=self.use_bias)
        self.out_proj = nn.Linear(self.d_ff, self.d_model, bias=self.use_bias)

        self.init_weights()


    def init_weights(self, mean: float = 0.0, std_divisor_multiplier: float = 1.0):
        std = (2.0 / (std_divisor_multiplier * (self.d_model + self.d_ff))) ** 0.5
        nn.init.normal_(self.in_proj.weight, mean=mean, std=std)
        nn.init.normal_(self.out_proj.weight, mean=mean, std=std)
        if self.use_bias:
            nn.init.constant_(self.in_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)


    def forward(self, x):
        output = self.in_proj(x)
        output = self.act_fn(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        return self.out_proj(output)


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    """
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 n_head: int,
                 n_layer: int,
                 prenorm: bool,
                 activation: str,
                 dropout: float,
                 use_bias: bool):
        super().__init__()
        self.dropout = dropout
        self.n_layer = n_layer
        self.prenorm = prenorm
        self.attns = nn.ModuleList([MultiheadAttention(d_model=d_model,
                                                       n_head=n_head,
                                                       dropout=self.dropout,
                                                       use_bias=use_bias) for _ in range(self.n_layer)])
        self.ffs = nn.ModuleList([PositionwiseFFN(d_model=d_model,
                                                  d_ff=d_ff,
                                                  dropout=self.dropout,
                                                  use_bias=use_bias,
                                                  activation=activation) for _ in range(self.n_layer)])

        n_norm = self.n_layer * 2 + 1 if self.prenorm else self.n_layer * 2
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_norm)])


    def forward(self,
                src_input: torch.Tensor,
                src_mask: Optional[torch.BoolTensor] = None,
                return_attn: bool = False) -> Dict:
        prenorm = self.prenorm
        if return_attn:
            attn_list = []

        x = F.dropout(src_input, p=self.dropout, training=self.training)
        for i in range(self.n_layer):
            attn = self.attns[i]
            ff = self.ffs[i]
            attn_norm = self.norms[2 * i]
            ff_norm = self.norms[2 * i + 1]

            # Attention block
            residual = x
            x = attn_norm(x) if prenorm else x
            x, a = attn(q=x, k=x, v=x, mask=src_mask)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = attn_norm(x) if not prenorm else x

            # Feedforward block
            residual = x
            x = ff_norm(x) if prenorm else x
            x = ff(x)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = ff_norm(x) if not prenorm else x

            if return_attn:
                attn_list.append(a)
        x = self.norms[-1](x) if prenorm else x

        result = {'output': x}
        if return_attn:
            result['self_attn'] = attn_list
        return result


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 n_head: int,
                 n_layer: int,
                 prenorm: bool,
                 activation: str,
                 dropout: float,
                 use_bias: bool):
        super().__init__()
        self.dropout = dropout
        self.n_layer = n_layer
        self.prenorm = prenorm

        self.self_attns = nn.ModuleList([MultiheadAttention(d_model=d_model,
                                                           n_head=n_head,
                                                           dropout=self.dropout,
                                                           use_bias=use_bias) for _ in range(self.n_layer)])
        self.cross_attns = nn.ModuleList([MultiheadAttention(d_model=d_model,
                                                            n_head=n_head,
                                                            dropout=self.dropout,
                                                            use_bias=use_bias) for _ in range(self.n_layer)])
        self.ffs = nn.ModuleList([PositionwiseFFN(d_model=d_model,
                                                  d_ff=d_ff,
                                                  dropout=self.dropout,
                                                  use_bias=use_bias,
                                                  activation=activation) for _ in range(self.n_layer)])

        n_norm = self.n_layer * 3 + 1 if self.prenorm else self.n_layer * 3
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_norm)])


    def forward(self,
                tgt_input: torch.Tensor,
                enc_output: torch.Tensor,
                tgt_mask: Optional[torch.BoolTensor] = None,
                enc_mask: Optional[torch.BoolTensor] = None,
                return_attn: bool = False):
        prenorm = self.prenorm
        if return_attn:
            self_attn_list = []
            cross_attn_list = []

        x = F.dropout(tgt_input, p=self.dropout, training=self.training)
        for i in range(self.n_layer):
            self_attn = self.self_attns[i]
            cross_attn = self.cross_attns[i]
            ff = self.ffs[i]
            self_attn_norm = self.norms[3 * i]
            cross_attn_norm = self.norms[3 * i + 1]
            ff_norm = self.norms[3 * i + 2]

            # Self-attention block
            residual = x
            x = self_attn_norm(x) if prenorm else x
            x, sa = self_attn(q=x, k=x, v=x, mask=tgt_mask)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = self_attn_norm(x) if not prenorm else x

            # Cross-attention block
            residual = x
            x = cross_attn_norm(x) if prenorm else x
            x, ca = cross_attn(q=x, k=enc_output, v=enc_output, mask=enc_mask)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = cross_attn_norm(x) if not prenorm else x

            # Feedforward block
            residual = x
            x = ff_norm(x) if prenorm else x
            x = ff(x)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = ff_norm(x) if not prenorm else x

            if return_attn:
                self_attn_list.append(sa)
                cross_attn_list.append(ca)
        x = self.norms[-1](x) if prenorm else x

        result = {'output': x}
        if return_attn:
            result['self_attn'] = self_attn_list
            result['cross_attn'] = cross_attn_list
        return result
# EOF
