from typing import Iterator
import itertools

import torch.nn as nn

class ReinforceModel(nn.Module):

    def non_bse_parameters(self) -> Iterator[nn.Parameter]:
        names, mods = zip(*filter(lambda x: x[0] != 'bse', self.named_children()))
        return itertools.chain.from_iterable([m.parameters() for m in mods])

