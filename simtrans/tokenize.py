from typing import Optional, List
import os

from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import sentencepiece as spm


def get_sp_tokenizer(cfg: DictConfig):
    tokenizer = SPTokenizer(model=os.path.join(hydra.utils.get_original_cwd(), cfg.tokenizer.spm_path),
                            name='sp_tokenizer')
    return tokenizer


class BaseTokenizer(object):
    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name


class SPTokenizer(BaseTokenizer):
    def __init__(self, model: str, name: str) -> None:
        super().__init__(name)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model)

    def __len__(self) -> int:
        return len(self.sp)

    def __getitem__(self, tok: str) -> int:
        return self.sp[tok]

    def idx_to_token(self, idx: int) -> str:
        return self.sp.IdToPiece(idx)

    def tokenize(self, text: str) -> List[str]:
        toks = self.sp.EncodeAsPieces(text)
        return toks

    def convert_text_to_ids(self,
                            text: str,
                            attach_head: Optional[str] = None,
                            attach_tail: Optional[str] = None) -> List[int]:
        ids = self.sp.EncodeAsIds(text)
        if attach_head is not None:
            ids = [self.sp[attach_head]] + ids
        if attach_tail is not None:
            ids += [self.sp[attach_tail]]
        return ids

    def convert_ids_to_text(self, ids: List[int]) -> str:
        text = self.sp.DecodeIds(ids)
        return text
