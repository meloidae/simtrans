from typing import Optional, Iterable, List
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
import numpy as np
from numba import njit, prange


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_lines_to_file(path: str, lines: List[str]) -> None:
    with open(path, mode='w', encoding='UTF-8') as f:
        for l in lines:
            f.write(f'{l}\n')


def save_checkpoint(params: dict) -> None:
    save_state = { key: value for key, value in params.items() }
    torch.save(save_state, os.path.join(os.getcwd(), "checkpoint.pt"))


def sort_by_index(items_to_sort: Iterable, indices: Iterable) -> Iterable:
    sorted_items = sorted(zip(indices, items_to_sort), key=lambda x: x[0])  # ascending order
    _, sorted_items = zip(*sorted_items)  # unzip
    return sorted_items


def iterable_to_device(iterable: Iterable, device: torch.device) -> tuple:
    moved = []
    for i in iterable:
        if isinstance(i, torch.Tensor):
            moved.append(i.to(device=device))
        else:
            moved.append(i)
    return tuple(moved)


def init_bert_params(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


def label_smoothed_cross_entropy(model_outputs: torch.Tensor,
                                 tgt_labels: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None,
                                 smoothing_eps: float = 0.1,
                                 reduce: bool = True):

    # model_outputs: [batch_size, length, vocab]
    # tgt_labels: [batch_size, length]
    # mask: [batch_size, length]

    n_labels = model_outputs.size(-1)
    model_outputs = model_outputs.contiguous().view(-1, n_labels)  # [batch_size * length, vocab]
    tgt_labels = tgt_labels.contiguous().view(-1, 1)  # [batch_size * length, 1]

    # Index by mask
    if mask is not None:
        mask = mask.reshape(-1)  # [batch_size * length]
        model_outputs = model_outputs[mask]
        tgt_labels = tgt_labels[mask]

    # Calculate loss
    log_probs = F.log_softmax(model_outputs, dim=-1)
    nll_loss = -log_probs.gather(dim=-1, index=tgt_labels)  # one_hot * -log_prob at target index
    smooth_loss = -log_probs.sum(dim=-1, keepdim=True)  # sum of -lob_prob

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

    # Mixture of one_hot and uniform dist
    # (sum of -lob_prob) / n_labels = sum of -log_prob * uniform_dist
    loss = (1 - smoothing_eps) * nll_loss + (smoothing_eps / n_labels) * smooth_loss
    return loss

def get_key_padding_mask(lens: torch.Tensor,
                         max_len: Optional[int] = None,
                         pad_with_false: bool = False) -> torch.Tensor:
    # True for pad positions
    if max_len is None:
        max_len = int(torch.max(lens).item())
    batch_size = lens.size(0)
    device = lens.device
    idx = torch.arange(end=max_len, device=device)[None, :].expand(batch_size, -1)
    if pad_with_false:
        return idx < lens[:, None] # [B, max_len]
    return idx >= lens[:, None]  # [B, max_len]

def get_subseq_mask(q_len: int, k_len: int, device: torch.device, offset: int = 0) -> torch.Tensor:
    # True for positions not supposed to be attended
    subseq_mask = torch.tril(torch.ones(q_len, k_len, device=device), diagonal=offset) == 0
    return subseq_mask

def get_square_subseq_mask(max_len: int, device: torch.device) -> torch.Tensor:
    return get_subseq_mask(q_len=max_len, k_len=max_len, device=device)

def get_wait_k_mask(q_len: int, k_len: int, wait_k: int, device: torch.device) -> torch.Tensor:
    return get_subseq_mask(q_len=q_len, k_len=k_len, device=device, offset=wait_k - 1)

@njit(parallel=True)
def gleu_part(sent1, len1, sent2, len2, max_n):
    total_ngrams = np.zeros(max_n)
    match_ngrams = np.zeros(max_n)

    for n in prange(max_n):
        m = n + 1
        for i in range(len1 - n):
            total_ngrams[n] += 1.0
            for j in range(len2 - n):
                match = True
                for k in range(m):
                    if sent1[i + k] != sent2[j + k]:
                        match = False
                        break
                if match:
                    match_ngrams[n] += 1.0
                    break

    match_ngram = np.sum(match_ngrams)
    total_ngram = np.sum(total_ngrams)
    return match_ngram / total_ngram


@njit(parallel=True)
def gleu(sents1, lens1, sents2, lens2, max_n=4):
    # Expect sents1 = [batch_size, seq_len]
    batch_size = sents1.shape[0]
    gleus1 = np.zeros(batch_size)
    gleus2 = np.zeros(batch_size)
    for b in prange(batch_size):
        gleus1[b] = gleu_part(sents1[b], lens1[b], sents2[b], lens2[b], max_n)
        gleus2[b] = gleu_part(sents2[b], lens2[b], sents1[b], lens1[b], max_n)
        if gleus2[b] < gleus1[b]:
            gleus1[b] = gleus2[b]

    return gleus1
