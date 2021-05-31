from typing import Optional, Dict
import math

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import init_bert_params, get_key_padding_mask, get_square_subseq_mask, get_wait_k_mask
from .utils import label_smoothed_cross_entropy
from .wait_k_transformer import WaitKTransformer
from .transformer_encoder_rnn_decoder import TransformerEncoderRNNDecoder

def build_model(cfg: DictConfig) -> nn.Module:
    model_type = cfg.model.type
    assert model_type in ['wait_k_transformer', 'transformer_encoder_rnn_decoder'], f'Unknown model type: {model_type}'
    
    if model_type == 'wait_k_transformer':
        model = WaitKTransformer(cfg)
    else:
        model = TransformerEncoderRNNDecoder(cfg)

    return model

class DistributedDataParallelPassthrough(nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

