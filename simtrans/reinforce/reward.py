from typing import Callable, List, Tuple, Set, Dict, Any, Optional
import sys
import os
import functools
import itertools
import logging

from omegaconf import DictConfig
import torch
from torch.nn.utils.rnn import pad_sequence
import Mykytea
import numpy as np
from nltk.translate.ribes_score import sentence_ribes
from numba import njit, prange, typeof
import numba as nb

from ..tokenize import SPTokenizer
from ..utils import gleu

logger = logging.getLogger(__name__)

dirpath = os.path.dirname(__file__)
sys.path.append(os.path.join(dirpath, 'awesome_align'))

import modeling
from configuration_bert import BertConfig
from modeling import BertForMaskedLM
from tokenization_bert import BertTokenizer


def _prepare_for_model(tokenizer: BertTokenizer, split_fn: Callable, sent: str) -> Tuple[Dict, List[Any], int]:
    if split_fn is not None:
        sent_split = list(split_fn(sent.strip()))
    else:
        sent_split = sent.strip().split()
    tok = [tokenizer.tokenize(w) for w in sent_split]
    wid = [tokenizer.convert_tokens_to_ids(x) for x in tok]
    ids = tokenizer.prepare_for_model(list(itertools.chain(*wid)),
                                      return_tensors='pt',
                                      max_length=tokenizer.max_len)
    bpe2word_map = []
    for i, word_list in enumerate(tok):
        bpe2word_map += [i for x in word_list]
    return ids['input_ids'][0], bpe2word_map, len(bpe2word_map)

def _get_score(src_tgt_pairs: Set[Tuple[int, int]], src_len: int, tgt_len: int, normalize: bool = True):
    diffs = []
    for src, tgt in src_tgt_pairs:
        if normalize:
            src = src / src_len
            tgt = tgt / tgt_len
        diffs.append(max(tgt - src, 0.0))
    score = 1.0 - (sum(diffs) / max(len(src_tgt_pairs), 1))
    return score


def _get_ribes(src_tgt_pairs: Set[Tuple[int, int]], src_len: int, tgt_len: int):
    unique = set()
    sorted_pairs = sorted(list(src_tgt_pairs), key=lambda x: x[1])
    tgt = []
    for s, _ in sorted_pairs:
        if s not in unique:
            tgt.append(s)
            unique.add(s)
    # TODO: consider precision?
    # tgt.extend([-100] * (tgt_len - len(tgt)))
    if len(tgt) <= 1:
        return 0
    src = list(range(src_len))
    src = [src]
    return sentence_ribes(src, tgt)


@njit(parallel=True)
def _get_kendalls_tau_numba(inputs: nb.typed.List[Tuple[np.ndarray, np.ndarray, int]]):
    batch_size = len(inputs)
    scores = np.empty((batch_size,))
    for i in prange(batch_size):
        src, tgt, length = inputs[i]
        scores[i] = kendalls_tau_small_n(src, tgt, length)

    # Regularize so that the range is [0, 1] (originally [-1, 1])
    scores = (scores + 1) / 2
    return scores

def _get_kendalls_tau(src_tgt_pairs: List[Set[Tuple[int, int]]]):
    pairs = [tuple(zip(*list(src_tgt))) for src_tgt in src_tgt_pairs]
    inputs = [(np.array(s, dtype=np.int64), np.array(t, dtype=np.int64), len(s)) for s, t in pairs]
    nb_list = nb.typed.List.empty_list(typeof(inputs[0]))
    for i in inputs:
        nb_list.append(i)
    return _get_kendalls_tau_numba(nb_list)

@njit
def copyto(dst: np.ndarray, src: np.ndarray, size: int):
    for i in prange(size):
        dst[i] = src[i]

@njit
def insertion_sort(x: np.ndarray, length: int) -> int:
    if length < 2:
        return 0

    swaps = 0
    max_j = length - 1
    for i in range(length - 2, -1, -1):
        val = x[i]
        j = i
        while j < max_j and x[j + 1] < val:
            x[j] = x[j + 1]
            j += 1
        x[j] = val
        swaps += (j - i)

    return swaps


@njit
def merge(x: np.ndarray, buf: np.ndarray, middle: int, length: int) -> int:
    buf_index = 0
    swaps = 0
    left = x
    right = x[middle:]
    left_len = middle
    right_len = length - middle

    while left_len > 0 and right_len > 0:
        if right[0] < left[0]:
            buf[buf_index] = right[0]
            swaps += left_len
            right_len -= 1
            right = right[1:]
        else:
            buf[buf_index] = left[0]
            left_len -= 1
            left = left[1:]
        buf_index += 1

    if left_len > 0:
        copyto(buf[buf_index:], left, left_len)
    elif right_len > 0:
        copyto(buf[buf_index:], right, right_len)

    return swaps

@njit
def merge_sort(x: np.ndarray, buf: np.ndarray, length: int) -> int:
    if length < 10:
        return insertion_sort(x, length)
    swaps = 0
    if length < 2:
        return 0

    half = length // 2
    swaps += merge_sort(x, buf, half)
    swaps += merge_sort(x[half:], buf[half:], length - half)
    swaps += merge(x, buf, half, length)

    copyto(x, buf, length)

    return swaps

@njit
def get_ms(x: np.ndarray, length: int) -> int:
    ms = 0
    tie = 0
    for i in range(1, length):
        if x[i] == x[i - 1]:
            tie += 1
        elif tie > 0:
            ms += (tie * (tie + 1)) // 2
            tie = 0
    if tie > 0:
        ms += (tie * (tie + 1)) // 2

    return ms


@njit
def kendalls_tau(a: np.ndarray, b: np.ndarray, length: int) -> float:
    if length < 2:
        return 0

    m1 = 0
    m2 = 0
    n_pair = length * (length - 1) // 2
    s = n_pair
    tie = 0

    for i in range(1, length):
        if a[i] == a[i - 1]:
            tie += 1
        elif tie > 0:
            insertion_sort(b[i - tie - 1:], tie + 1)
            m1 += tie * (tie + 1) // 2
            s += get_ms(b[i - tie - 1:], tie + 1)
            tie = 0

    if tie > 0:
        insertion_sort(b[length - tie - 1:], tie + 1)
        m1 += tie * (tie + 1) // 2
        s += get_ms(b[length - tie - 1:], tie + 1)

    swaps = merge_sort(b, a, length)
    m2 = get_ms(b, length)
    s -= (m1 + m2) + 2 * swaps

    if m1 < n_pair and m2 < n_pair:
        return s / (np.sqrt(n_pair - m1) * np.sqrt(n_pair - m2))
    else:
        return 0


@njit(parallel=True)
def kendalls_tau_small_n(a: np.ndarray, b: np.ndarray, length: int) -> float:

    s = 0
    m1 = 0
    m2 = 0
    for i in prange(length):
        for j in prange(i + 1, length):
            if b[i] > b[j]:
                if a[i] > a[j]:
                    s += 1
                elif a[i] < a[j]:
                    s += -1
                else:
                    m1 += 1
            elif b[i] < b[j]:
                if a[i] > a[j]:
                    s += -1
                elif a[i] < a[j]:
                    s += 1
                else:
                    m1 += 1
            else:
                m2 += 1
                if a[i] == a[j]:
                    m1 += 1
    n_pair = length * (length - 1) // 2
    if m1 < n_pair and m2 < n_pair:
        return s / (np.sqrt(n_pair - m1) * np.sqrt(n_pair - m2))
    else:
        return 0


def get_kendalls_tau(src_tgt_pairs: Set[Tuple[int, int]]):
    sorted_pairs = sorted(list(src_tgt_pairs), key=lambda x: x[0])
    src, tgt = zip(*sorted_pairs)
    src_arr = np.array(src, dtype=np.int64)
    tgt_arr = np.array(tgt, dtype=np.int64)
    length = src_arr.shape[0]
    tau = kendalls_tau(src_arr, tgt_arr, length)
    return tau

def get_kendalls_tau_small_n(src_tgt_pairs: Set[Tuple[int, int]]):
    src, tgt = zip(*list(src_tgt_pairs))
    src_arr = np.array(src, dtype=np.int64)
    tgt_arr = np.array(tgt, dtype=np.int64)
    length = src_arr.shape[0]
    tau = kendalls_tau_small_n(src_arr, tgt_arr, length)
    return tau


def reward_align(model: BertForMaskedLM,
                 sp_tokenizer: SPTokenizer,
                 tokenizer: BertTokenizer,
                 split_fn: Tuple[Callable, Callable],
                 device: torch.device, src: torch.Tensor, tgt: torch.Tensor):
    if sp_tokenizer is not None:
        src_list = [sp_tokenizer.convert_ids_to_text(line) for line in src.tolist()]
        tgt_list = [sp_tokenizer.convert_ids_to_text(line) for line in tgt.tolist()]
    else:
        src_list = src
        tgt_list = tgt
    prep_fn_src = functools.partial(_prepare_for_model, tokenizer, split_fn[0])
    prep_fn_tgt = functools.partial(_prepare_for_model, tokenizer, split_fn[1])
    batch_size = len(src_list)

    with torch.no_grad():
        src_ids, src_bpe2word, src_lens = zip(*map(prep_fn_src, src_list))
        tgt_ids, tgt_bpe2word, tgt_lens = zip(*map(prep_fn_tgt, tgt_list))
        non_zero_length_tgts = [i for i, val in enumerate(tgt_lens) if val != 0]
        n_non_zero_length_tgts = len(non_zero_length_tgts)
        if n_non_zero_length_tgts == 0:
            # All targets have zero lengths -> set all scores to zero
            scores = [0.0] * batch_size
        else:
            if n_non_zero_length_tgts < batch_size:
                # Select only ones with tgts that are have non-zero lengths
                src_ids = [src_ids[i] for i in non_zero_length_tgts]
                tgt_ids = [tgt_ids[i] for i in non_zero_length_tgts]
                src_lens = [src_lens[i] for i in non_zero_length_tgts]
                tgt_lens = [tgt_lens[i] for i in non_zero_length_tgts]
                src_bpe2word = [src_bpe2word[i] for i in non_zero_length_tgts]
                tgt_bpe2word = [tgt_bpe2word[i] for i in non_zero_length_tgts]
            src_ids_padded = pad_sequence(src_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            tgt_ids_padded = pad_sequence(tgt_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

            try: 
                word_aligns_list = model.get_aligned_word(src_ids_padded,
                                                          tgt_ids_padded,
                                                          src_bpe2word,
                                                          tgt_bpe2word, 
                                                          device=device,
                                                          src_len=0,
                                                          tgt_len=0,
                                                          test=True)
            except IndexError as e:
                logger.info(f'src = {src_ids}')
                logger.info(f'tgt = {tgt_ids}')
                logger.exception(F'{e}')

            scores = [_get_score(wa, sl, tl) for wa, sl, tl in zip(word_aligns_list, src_lens, tgt_lens)]
            if n_non_zero_length_tgts < batch_size:
                base_scores = [0.0] * batch_size
                for i, v in zip(non_zero_length_tgts, scores):
                    base_scores[i] = v
                scores = base_scores
        scores = torch.tensor(scores, dtype=torch.float32)  # [batch_size]
    return scores

def reward_ribes(model: BertForMaskedLM,
                 sp_tokenizer: SPTokenizer,
                 tokenizer: BertTokenizer,
                 split_fn: Tuple[Callable, Callable],
                 device: torch.device,
                 src: torch.Tensor,
                 tgt: torch.Tensor):
    if sp_tokenizer is not None:
        src_list = [sp_tokenizer.convert_ids_to_text(line) for line in src.tolist()]
        tgt_list = [sp_tokenizer.convert_ids_to_text(line) for line in tgt.tolist()]
    else:
        src_list = src
        tgt_list = tgt
    prep_fn_src = functools.partial(_prepare_for_model, tokenizer, split_fn[0])
    prep_fn_tgt = functools.partial(_prepare_for_model, tokenizer, split_fn[1])
    batch_size = len(src_list)

    with torch.no_grad():
        src_ids, src_bpe2word, src_lens = zip(*map(prep_fn_src, src_list))
        tgt_ids, tgt_bpe2word, tgt_lens = zip(*map(prep_fn_tgt, tgt_list))
        non_zero_length_tgts = [i for i, val in enumerate(tgt_lens) if val != 0]
        n_non_zero_length_tgts = len(non_zero_length_tgts)
        if n_non_zero_length_tgts == 0:
            # All targets have zero lengths -> set all scores to zero
            scores = [0.0] * batch_size
        else:
            if n_non_zero_length_tgts < batch_size:
                # Select only ones with tgts that are have non-zero lengths
                src_ids = [src_ids[i] for i in non_zero_length_tgts]
                tgt_ids = [tgt_ids[i] for i in non_zero_length_tgts]
                src_lens = [src_lens[i] for i in non_zero_length_tgts]
                tgt_lens = [tgt_lens[i] for i in non_zero_length_tgts]
                src_bpe2word = [src_bpe2word[i] for i in non_zero_length_tgts]
                tgt_bpe2word = [tgt_bpe2word[i] for i in non_zero_length_tgts]
            src_ids_padded = pad_sequence(src_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            tgt_ids_padded = pad_sequence(tgt_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

            try: 
                word_aligns_list = model.get_aligned_word(src_ids_padded,
                                                          tgt_ids_padded,
                                                          src_bpe2word,
                                                          tgt_bpe2word, 
                                                          device=device,
                                                          src_len=0,
                                                          tgt_len=0,
                                                          test=True)
            except IndexError as e:
                logger.info(f'src = {src_ids}')
                logger.info(f'tgt = {tgt_ids}')
                logger.exception(F'{e}')

            scores = [_get_ribes(wa, sl, tl) for wa, sl, tl in zip(word_aligns_list, src_lens, tgt_lens)]
            if n_non_zero_length_tgts < batch_size:
                base_scores = [0.0] * batch_size
                for i, v in zip(non_zero_length_tgts, scores):
                    base_scores[i] = v
                scores = base_scores
        scores = torch.tensor(scores, dtype=torch.float32)  # [batch_size]
    # TODO: temporary hack to get word_aligns_list
    if sp_tokenizer is None:
        return scores, word_aligns_list
    return scores


def reward_kendall(model: BertForMaskedLM,
                   sp_tokenizer: SPTokenizer,
                   tokenizer: BertTokenizer,
                   split_fn: Tuple[Callable, Callable],
                   device: torch.device,
                   src: torch.Tensor,
                   tgt: torch.Tensor):
    if sp_tokenizer is not None:
        src_list = [sp_tokenizer.convert_ids_to_text(line) for line in src.tolist()]
        tgt_list = [sp_tokenizer.convert_ids_to_text(line) for line in tgt.tolist()]
    else:
        src_list = src
        tgt_list = tgt
    prep_fn_src = functools.partial(_prepare_for_model, tokenizer, split_fn[0])
    prep_fn_tgt = functools.partial(_prepare_for_model, tokenizer, split_fn[1])
    batch_size = len(src_list)

    with torch.no_grad():
        src_ids, src_bpe2word, src_lens = zip(*map(prep_fn_src, src_list))
        tgt_ids, tgt_bpe2word, tgt_lens = zip(*map(prep_fn_tgt, tgt_list))
        non_zero_length_tgts = [i for i, val in enumerate(tgt_lens) if val != 0]
        n_non_zero_length_tgts = len(non_zero_length_tgts)
        if n_non_zero_length_tgts == 0:
            # All targets have zero lengths -> set all scores to zero
            scores = [0.0] * batch_size
        else:
            if n_non_zero_length_tgts < batch_size:
                # Select only ones with tgts that are have non-zero lengths
                src_ids = [src_ids[i] for i in non_zero_length_tgts]
                tgt_ids = [tgt_ids[i] for i in non_zero_length_tgts]
                src_lens = [src_lens[i] for i in non_zero_length_tgts]
                tgt_lens = [tgt_lens[i] for i in non_zero_length_tgts]
                src_bpe2word = [src_bpe2word[i] for i in non_zero_length_tgts]
                tgt_bpe2word = [tgt_bpe2word[i] for i in non_zero_length_tgts]
            src_ids_padded = pad_sequence(src_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            tgt_ids_padded = pad_sequence(tgt_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

            word_aligns_list = model.get_aligned_word(src_ids_padded,
                                                      tgt_ids_padded,
                                                      src_bpe2word,
                                                      tgt_bpe2word, 
                                                      device=device,
                                                      src_len=0,
                                                      tgt_len=0,
                                                      test=True)
            scores = _get_kendalls_tau(word_aligns_list)
            if n_non_zero_length_tgts < batch_size:
                base_scores = [0.0] * batch_size
                for i, v in zip(non_zero_length_tgts, scores):
                    base_scores[i] = v
                scores = base_scores
        scores = torch.tensor(scores, dtype=torch.float32)  # [batch_size]
    # TODO: temporary hack to get word_aligns_list
    if sp_tokenizer is None:
        return scores, word_aligns_list
    return scores


def reward_gleu(out: np.ndarray, out_lens: np.ndarray, tgt: np.ndarray, tgt_lens: np.ndarray):
    scores = gleu(out, out_lens, tgt, tgt_lens)
    scores = torch.from_numpy(scores).to(torch.float32)  # [batch_size]
    return scores


def get_reward_fn(cfg: DictConfig,
                  sp_tokenizer: Optional[SPTokenizer] = None,
                  base_dir: Optional[str] = None) -> Callable:
    reward_type = cfg.model.reward
    if reward_type in ['alignment', 'xribes', 'kendall']:
        assert sp_tokenizer is not None, f'Must supply sp_tokenizer for reward_type = {reward_type}'
        assert base_dir is not None, f'Must supply base_dir for reward_type = {reward_type}'
        bert_path = os.path.join(base_dir, cfg.model.bert_path)
        config = BertConfig.from_pretrained(bert_path)
        tokenizer = BertTokenizer.from_pretrained(bert_path)
        modeling.PAD_ID = tokenizer.pad_token_id
        modeling.CLS_ID = tokenizer.cls_token_id
        modeling.SEP_ID = tokenizer.sep_token_id
        model = BertForMaskedLM.from_pretrained(bert_path,
                                                from_tf=bool(".ckpt" in bert_path),
                                                config=config,
                                                cache_dir=cfg.model.bert_cache_dir)

        device = torch.device('cuda', cfg.model.bert_gpu)
        model.to(device)
        model.eval()
        mk = Mykytea.Mykytea(f'-model {cfg.misc.kytea_path}')
        split_fn_src = mk.getWS if cfg.data.train.src.endswith(".ja") else None
        split_fn_tgt = mk.getWS if cfg.data.train.tgt.endswith(".ja") else None
        if reward_type == 'alignment':
            reward_fn = functools.partial(reward_align, model, sp_tokenizer, tokenizer, (split_fn_src, split_fn_tgt), device)
        elif reward_type == 'xribes':
            reward_fn = functools.partial(reward_ribes, model, sp_tokenizer, tokenizer, (split_fn_src, split_fn_tgt), device)
        else:
            reward_fn = functools.partial(reward_kendall, model, sp_tokenizer, tokenizer, (split_fn_src, split_fn_tgt), device)
    elif reward_type == 'gleu':
        reward_fn = reward_gleu
    else:
        assert False, f'Unknown reward_type: {reward_type}'

    return reward_fn


def get_reward_fn_no_cfg(reward_type: str,
                         sp_tokenizer: SPTokenizer,
                         base_dir: str,
                         bert_path: Optional[str] = None,
                         bert_gpu: Optional[int] = None,
                         bert_cache_dir: Optional[str] = None,
                         kytea_path: Optional[str] = None,
                         data_src_path: Optional[str] = None,
                         data_tgt_path: Optional[str] = None):
    if reward_type in ['alignment', 'xribes', 'kendall']:
        assert bert_path is not None
        assert bert_gpu is not None
        assert bert_cache_dir is not None
        assert kytea_path is not None
        assert data_src_path is not None
        assert data_tgt_path is not None
        bert_path = os.path.join(base_dir, bert_path)
        config = BertConfig.from_pretrained(bert_path)
        tokenizer = BertTokenizer.from_pretrained(bert_path)
        modeling.PAD_ID = tokenizer.pad_token_id
        modeling.CLS_ID = tokenizer.cls_token_id
        modeling.SEP_ID = tokenizer.sep_token_id
        model = BertForMaskedLM.from_pretrained(bert_path,
                                                from_tf=bool(".ckpt" in bert_path),
                                                config=config,
                                                cache_dir=bert_cache_dir)

        device = torch.device('cuda', bert_gpu)
        model.to(device)
        model.eval()
        mk = Mykytea.Mykytea(f'-model {kytea_path}')
        split_fn_src = mk.getWS if data_src_path.endswith(".ja") else None
        split_fn_tgt = mk.getWS if data_tgt_path.endswith(".ja") else None
        if reward_type == 'alignment':
            reward_fn = functools.partial(reward_align, model, sp_tokenizer, tokenizer, (split_fn_src, split_fn_tgt), device)
        elif reward_type == 'xribes':
            reward_fn = functools.partial(reward_ribes, model, sp_tokenizer, tokenizer, (split_fn_src, split_fn_tgt), device)
        else:
            reward_fn = functools.partial(reward_kendall, model, sp_tokenizer, tokenizer, (split_fn_src, split_fn_tgt), device)
    elif reward_type == 'gleu':
        reward_fn = reward_gleu
    else:
        assert False, f'Unknown reward_type: {reward_type}'

    return reward_fn

