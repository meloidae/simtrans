from typing import Optional, List, Tuple, Callable
import os
import random

from omegaconf import DictConfig
import hydra
import torch
import torch.distributed as dist

from .tokenize import SPTokenizer

def build_dataset(cfg: DictConfig, tokenizer: SPTokenizer):
    def load_ids(path: str, tgt: Optional[bool] = False):
        with open(os.path.join(hydra.utils.get_original_cwd(), path), encoding='UTF-8') as f:
            l_strip = [s.strip() for s in f.readlines()]
        return [tokenizer.convert_text_to_ids(text=l.lower(),  # TODO make everything lowercase for now
                                              attach_head='<s>' if tgt else None,
                                              attach_tail='</s>' if tgt else None) for l in l_strip]

    train_ids = load_ids(cfg.data.train.src), load_ids(cfg.data.train.tgt, True)
    dev_ids = load_ids(cfg.data.dev.src), load_ids(cfg.data.dev.tgt, True)
    test_ids = load_ids(cfg.data.test.src), load_ids(cfg.data.test.tgt, True)

    train_dataset = Seq2SeqDataset(src_itr=train_ids[0], tgt_itr=train_ids[1])
    dev_dataset = Seq2SeqDataset(src_itr=dev_ids[0], tgt_itr=dev_ids[1])
    test_dataset = Seq2SeqDataset(src_itr=test_ids[0], tgt_itr=test_ids[1])

    dataset = {
        'train': train_dataset,
        'dev': dev_dataset,
        'test': test_dataset
    }
    return dataset
    

def build_dataloader(cfg: DictConfig, dataset: dict, rank: Optional[int] = None):
    use_cuda = torch.cuda.is_available()
    if cfg.batch.distributed:
        train_sampler = DistributedDynamicBucketBatchSampler(data_source=dataset['train'],
                                                             max_tokens=cfg.batch.train.max_tokens,
                                                             seed=cfg.misc.seed,
                                                             num_replicas=cfg.batch.n_replica,
                                                             rank=rank)
    else:
        train_sampler = DynamicBucketBatchSampler(data_source=dataset['train'],
                                                  max_tokens=cfg.batch.train.max_tokens,
                                                  seed=cfg.misc.seed)
    train_loader = torch.utils.data.DataLoader(dataset=dataset['train'],
                                               batch_sampler=train_sampler,
                                               num_workers=cfg.batch.n_worker,
                                               collate_fn=seq2seq_collate_fn,
                                               pin_memory=use_cuda)
    
    dev_sampler = torch.utils.data.SequentialSampler(data_source=dataset['dev'])
    dev_loader = torch.utils.data.DataLoader(dataset=dataset['dev'],
                                             batch_size=cfg.batch.dev.batch_size,
                                             sampler=dev_sampler,
                                             collate_fn=seq2seq_collate_fn,
                                             pin_memory=use_cuda)

    test_sampler = torch.utils.data.SequentialSampler(data_source=dataset['test'])
    test_loader = torch.utils.data.DataLoader(dataset=dataset['test'],
                                              batch_size=cfg.batch.test.batch_size,
                                              sampler=test_sampler,
                                              collate_fn=seq2seq_collate_fn,
                                              pin_memory=use_cuda)

    loader = {
        'train': train_loader,
        'dev': dev_loader,
        'test': test_loader
    }
    return loader


def interleave_keys(a: int, b: int) -> int:
    ''' Taken from torchtext '''
    def interleave(args):
        return ''.join([x for t in zip(*args) for x in t])
    return int(''.join(interleave(format(int(x), '016b') for x in (a, b))), base=2)


def seq2seq_collate_fn(data: List) -> Tuple:
    batch_size = len(data)

    def merge(seqs: torch.Tensor, lens: List[int]) -> torch.Tensor:
        max_seq_len = max(lens)
        padded_seqs = torch.zeros(size=(batch_size, max_seq_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            end = lens[i]
            padded_seqs[i, :end] = s[:end]
        return padded_seqs

    for i, d in enumerate(data):
        d['local_idx'] = i

    data.sort(key=lambda x: -x['src'].size(0))

    src_seqs, src_lens, tgt_seqs, tgt_lens, local_idxs, global_idxs = [], [], [], [], [], []
    for d in data:
        src = d['src']
        tgt = d['tgt']
        src_seqs.append(src)
        src_lens.append(src.size(0))
        tgt_seqs.append(tgt)
        tgt_lens.append(tgt.size(0))
        local_idxs.append(d['local_idx'])
        global_idxs.append(d['global_idx'])

    merged_src = merge(src_seqs, src_lens)
    merged_tgt = merge(tgt_seqs, tgt_lens)
    src_lens = torch.LongTensor(src_lens)
    tgt_lens = torch.LongTensor(tgt_lens)

    return merged_src, src_lens, merged_tgt, tgt_lens, local_idxs, global_idxs


class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, src_itr, tgt_itr):
        self.data = [
            dict({
                'src': torch.LongTensor(src),
                'tgt': torch.LongTensor(tgt),
                'global_idx': i
            }) for i, (src, tgt) in enumerate(zip(src_itr, tgt_itr))
        ]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class DynamicBucketBatchSampler(torch.utils.data.Sampler):
    '''
    BucketBatchSampler with batching by a total token count (including padding)
    '''
    def __init__(self, data_source: List, max_tokens: int,
                 sort_key: Callable = lambda x: interleave_keys(x['src_len'], x['tgt_len']),
                 sort_by_batch_size: bool = False, drop_threshold: int = -1, padding_noise: float = 0.1,
                 seed: int = 42):
        super().__init__(data_source)
        self.data_source = data_source  # Dataset
        self.max_tokens = max_tokens
        self.sort_key = sort_key
        self.sort_by_batch_size = sort_by_batch_size
        self.drop_threshold = drop_threshold
        self.padding_noise = padding_noise

        # Impossible to calculate batch size until create_batches() is called
        self.num_batches = None

        longest_src = max(data_source, key=lambda x: x['src'].size(0))
        longest_tgt = max(data_source, key=lambda x: x['tgt'].size(0))

        self.max_src_len = longest_src['src'].size(0)
        self.max_tgt_len = longest_tgt['tgt'].size(0)
        self.seed = seed

    def __iter__(self):
        self.create_batches()
        for batch in self.batches:
            yield batch

    def __len__(self):
        return self.num_batches

    def create_batches(self) -> None:
        # Make batch generation deterministic
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        noise = torch.empty(size=(2, len(self.data_source)))
        noise.uniform_(-self.padding_noise, self.padding_noise)
        noise = noise.tolist()

        self.batches = []
        # Calculate padding lengths
        padding_lengths = []
        for i, d in enumerate(self.data_source):
            dict_obj = dict({
                'src_len': self.max_src_len - d['src'].size(0),
                'tgt_len': self.max_tgt_len - d['tgt'].size(0)
            })
            dict_obj['src_len'] += dict_obj['src_len'] * noise[0][i]
            dict_obj['tgt_len'] += dict_obj['tgt_len'] * noise[1][i]
            padding_lengths.append(dict_obj)

        # Sort based on padding lengths
        idx_list = range(len(self.data_source))
        sorted_idxs = sorted(idx_list, key=lambda x: self.sort_key(padding_lengths[x]))

        current_batch = []
        current_max_src_len = 0
        current_max_tgt_len = 0
        for i in sorted_idxs:
            item = self.data_source[i]
            src_len = item['src'].size(0)
            tgt_len = item['tgt'].size(0)
            if src_len > current_max_src_len:
                current_max_src_len = src_len
            if tgt_len > current_max_tgt_len:
                current_max_tgt_len = tgt_len
            total_seq_len = current_max_src_len + current_max_tgt_len
            # Determine if current item should be added to this batch
            if (len(current_batch) + 1) * total_seq_len <= self.max_tokens:
                current_batch.append(i)
            else:
                self.batches.append(current_batch)
                current_batch = [i]
                current_max_src_len = src_len
                current_max_tgt_len = tgt_len

        if len(current_batch) > 0:
            total_seq_len = current_max_src_len + current_max_tgt_len
            if len(current_batch) * total_seq_len >= self.drop_threshold:
                self.batches.append(current_batch)

        self.num_batches = len(self.batches)

        if self.sort_by_batch_size:
            self.batches = sorted(self.batches, key=lambda x: -len(x))
        else:
            self.shuffle()

        self.seed += 1  # Change seed every epoch

    def shuffle(self) -> None:
        random.Random(self.seed).shuffle(self.batches)

    def set_seed(self, seed: int) -> None:
        self.seed = seed


class DistributedDynamicBucketBatchSampler(DynamicBucketBatchSampler):
    '''
    BucketBatchSampler with batching by a total token count and support for distributed learning
    '''
    def __init__(self, data_source: List, max_tokens: int,
                 sort_key: Callable = lambda x: interleave_keys(x['src_len'], x['tgt_len']),
                 sort_by_batch_size: bool = False, drop_threshold: int = -1, padding_noise: float = 0.1,
                 seed: int = 42,
                 num_replicas: int = None, rank: int = None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        super().__init__(data_source=data_source,
                         max_tokens=max_tokens,
                         sort_key=sort_key,
                         sort_by_batch_size=sort_by_batch_size,
                         drop_threshold=drop_threshold,
                         padding_noise=padding_noise,
                         seed=seed)

        self.num_replicas = num_replicas
        self.rank = rank
        self.dist_batches = []

    def __iter__(self):
        self.create_batches()
        for batch in self.dist_batches:
            yield batch

    def __len__(self):
        return len(self.dist_batches)

    def create_batches(self) -> None:
        super().create_batches()
        remainder = len(self.batches) % self.num_replicas
        self.batches += self.batches[:(self.num_replicas - remainder)]
        assert len(self.batches) % self.num_replicas == 0

        self.dist_batches = self.batches[self.rank: len(self.batches): self.num_replicas]
