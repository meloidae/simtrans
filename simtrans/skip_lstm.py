import torch.nn as nn
import torch.nn.functional as F

class SkipLSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 bias: bool,
                 dropout: float):
        super().__init__()
        self.dropout = dropout
        self.n_layer = num_layers

        if self.n_layer == 1:
            self.rnns = nn.ModuleList([nn.LSTMCell(input_size=input_size,
                                                   hidden_size=hidden_size,
                                                   bias=bias)])
        else:
            self.rnns = nn.ModuleList(
                [nn.LSTMCell(input_size=input_size,
                             hidden_size=hidden_size,
                             bias=bias)] +
                [nn.LSTMCell(input_size=hidden_size,
                             hidden_size=hidden_size,
                             bias=bias) for _ in range(self.n_layer - 1)])

    def forward(self, x, prev_states):
        # Expcet x: [batch_size, 1, input_size]
        states = []

        residual = x
        h, state = self.rnns[0](x, prev_states[0])
        states.append(state)

        h = F.dropout(h, p=self.dropout, training=self.training)
        if residual.size(1) != h.size(1):
            hidden_size = h.size(1)
            h = h + residual[:, :, :hidden_size] + residual[:, :, hidden_size:]
        else:
            h = h + residual

        for i in range(1, self.n_layer):
            h, state = self.rnns[i](h, prev_states[i])



        return h, state
        

