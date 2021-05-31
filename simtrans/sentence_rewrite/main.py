import sys
import os
import argparse
import itertools
import functools
from logging import getLogger, DEBUG
from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence
import Mykytea

logger = getLogger(__name__)
logger.setLevel(level=DEBUG)


def align_src_to_tgt(src_tgt_pairs, src_len, remove_duplicate):
    # Reorder src according to position of tgt
    pairs = sorted(list(src_tgt_pairs), key=lambda x: (x[1], x[0]))
    src_aligned, _ = zip(*pairs)
    if remove_duplicate:
        unique = set()
        unique_list = []
        for i in src_aligned:
            if i not in unique:
                unique.add(i)
                unique_list.append(i)
        src_aligned = unique_list
    # Find src that has no corresponding tgt
    src_all = set(range(src_len))
    src_rest = src_all.difference(set(src_aligned))
    src_rest = sorted(list(src_rest))
    return src_aligned, src_rest


def align_tgt_to_src(src_tgt_pairs, tgt_len, remove_duplicate):
    # Reorder tgt according to position of src
    pairs = sorted(list(src_tgt_pairs), key=lambda x: (x[0], x[1]))
    _, tgt_aligned = zip(*pairs)
    if remove_duplicate:
        unique = set()
        unique_list = []
        for i in tgt_aligned:
            if i not in unique:
                unique.add(i)
                unique_list.append(i)
        tgt_aligned = unique_list
    # Find src that has no corresponding tgt
    tgt_all = set(range(tgt_len))
    tgt_rest = tgt_all.difference(set(tgt_aligned))
    tgt_rest = sorted(list(tgt_rest))
    return tgt_aligned, tgt_rest


def reorder(args):
    def to_bpe2word_map(tok):
        bpe2word_map = []
        for i, word_list in enumerate(tok):
            bpe2word_map += [i for x in word_list]
        return bpe2word_map

    def to_ids(path, split_fn, tokenizer):
        all_ids, bpe2word_maps, lens, lines, line_splits = [], [], [], [], []
        with open(path) as f:
            for l in tqdm(f.readlines()):
                line_split = split_fn(l)
                lines.append(l.strip())
                line_splits.append(line_split)
                tok = [tokenizer.tokenize(w) for w in line_split]
                wid = [tokenizer.convert_tokens_to_ids(x) for x in tok]
                ids = tokenizer.prepare_for_model(list(itertools.chain(*wid)),
                                                   return_tensors='pt',
                                                   max_length=tokenizer.max_len)['input_ids'][0]
                all_ids.append(ids)
                bpe2word_maps.append(to_bpe2word_map(tok))
                lens.append(len(line_split))

        return all_ids, bpe2word_maps, lens, lines, line_splits

    def filter_fn(elem):
        _sid, smap, _slen, _sline, _stok, _tid, tmap, _tlen, _tline, _ttok = elem
        return len(smap) > 0 and len(tmap) > 0
    
    def write_aligned(ignore_unaligned, path, output):
        with open(path, 'w') as f:
            for aligned, rest in output:
                if ignore_unaligned:
                    f.write(f'{aligned}\n')
                else:
                    f.write(f'{aligned} ||| {rest}\n')

    def write_lines(path, output):
        with open(path, 'w') as f:
            for line in output:
                f.write(f'{line}\n')


    # Import awesome-align libs
    sys.path.append(os.path.join('', 'simtrans/reinforce/awesome_align'))
    import modeling
    from configuration_bert import BertConfig
    from modeling import BertForMaskedLM
    from tokenization_bert import BertTokenizer

    # Initialize the BERT model and tokenizer
    logger.info('Initializing BERT model and tokenizer')
    bert_path = os.path.join(args.base_dir, args.bert_path)
    config = BertConfig.from_pretrained(bert_path)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    modeling.PAD_ID = tokenizer.pad_token_id
    modeling.CLS_ID = tokenizer.cls_token_id
    modeling.SEP_ID = tokenizer.sep_token_id
    model = BertForMaskedLM.from_pretrained(bert_path,
                                            from_tf=bool(".ckpt" in bert_path),
                                            config=config,
                                            cache_dir=args.bert_cache_dir)

    device = torch.device('cuda', args.bert_gpu)
    model.to(device)
    model.eval()

    # Set up split functions
    mk = Mykytea.Mykytea(f'-model {args.kytea_path}') if args.kytea_path is not None else None

    if mk is not None and args.src_path.endswith('.ja'):
        split_src = lambda sent: list(mk.getWS(sent.strip()))
        join_src = lambda split: ''.join(split)
    else:
        split_src = lambda sent: sent.strip().split()
        join_src = lambda split: ' '.join(split)

    if mk is not None and args.tgt_path.endswith('.ja'):
        split_tgt = lambda sent: list(mk.getWS(sent.strip()))
        join_tgt = lambda split: ''.join(split)
    else:
        split_tgt = lambda sent: sent.strip().split()
        join_tgt = lambda split: ' '.join(split)

    # Preparing data
    logger.info('Preprocessing data for model')
    src_ids, src_bpe2word_map, src_lens, src_lines, src_toks = to_ids(args.src_path, split_src, tokenizer)
    tgt_ids, tgt_bpe2word_map, tgt_lens, tgt_lines, tgt_toks = to_ids(args.tgt_path, split_tgt, tokenizer)

    # Filter out anything src/tgt that has a length of zero after tokenization
    filtered = filter(filter_fn,
                      zip(src_ids, src_bpe2word_map, src_lens, src_lines, src_toks,
                          tgt_ids, tgt_bpe2word_map, tgt_lens, tgt_lines, tgt_toks))
    src_ids, src_bpe2word_map, src_lens, src_lines, src_toks, tgt_ids, tgt_bpe2word_map, tgt_lens, tgt_lines, tgt_toks = zip(*filtered)

    data_len = len(src_ids)
    indices = list(range(0, data_len, args.batch_size)) + [data_len]
    outputs = []
    logger.info('Reordering data using alignment information')
    with torch.no_grad():
        for start, end in tqdm(zip(indices[:-1], indices[1:])):
            # Make a minibatch
            src = src_ids[start:end]
            tgt = tgt_ids[start:end]
            src_tensor = pad_sequence(src, batch_first=True, padding_value=tokenizer.pad_token_id)
            tgt_tensor = pad_sequence(tgt, batch_first=True, padding_value=tokenizer.pad_token_id)
            src_len = src_lens[start:end]
            tgt_len = tgt_lens[start:end]
            src_tok = src_toks[start:end]
            tgt_tok = tgt_toks[start:end]

            # Calculate alignment
            word_aligns_list = model.get_aligned_word(src_tensor,
                                                      tgt_tensor,
                                                      src_bpe2word_map[start:end],
                                                      tgt_bpe2word_map[start:end],
                                                      device=device,
                                                      src_len=0,
                                                      tgt_len=0,
                                                      test=True)

            if args.align_opt == 'src_to_tgt':
                realigned = [align_src_to_tgt(w, l, args.remove_duplicate) for w, l in zip(word_aligns_list, src_len)]
                pos_to_toks = lambda ls, i: list(map(lambda x: src_tok[i][x], ls))
                join_toks = join_src
            elif args.align_opt == 'tgt_to_src':
                realigned = [align_tgt_to_src(w, l, args.remove_duplicate) for w, l in zip(word_aligns_list, tgt_len)]
                pos_to_toks = lambda ls, i: list(map(lambda x: tgt_tok[i][x], ls))
                join_toks = join_tgt
            else:
                assert False, f'Unimplemented align_opt: {args.align_opt}'

            # Convert positions to ids, then decode to str
            output = [(pos_to_toks(aligned_rest[0], i),
                       pos_to_toks(aligned_rest[1], i)) for i, aligned_rest in enumerate(realigned)]
            output = [(join_toks(aligned), join_toks(rest)) for aligned, rest in output]

            outputs.extend(output)

    if args.align_opt == 'src_to_tgt':
        src_outputs = outputs
        src_write = functools.partial(write_aligned, args.ignore_unaligned)
        tgt_outputs = tgt_lines
        tgt_write = write_lines
    elif args.align_opt == 'tgt_to_src':
        src_outputs = src_lines
        src_write = write_lines
        tgt_outputs = outputs
        tgt_write = functools.partial(write_aligned, args.ignore_unaligned)
    else:
        assert False, f'Unimplemented align_opt: {args.align_opt}'

    logger.info('Writing results to files')
    src_write(args.src_out_path, src_outputs)
    tgt_write(args.tgt_out_path, tgt_outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='')
    parser.add_argument('--bert_path', type=str, default='data/model_without_co')
    parser.add_argument('--bert_cache_dir', type=str, default='/data/local/gyasui/.cache/torch/transformers')
    parser.add_argument('--bert_gpu', type=int)
    parser.add_argument('--kytea_path', type=str, default=None)
    parser.add_argument('--src_path', type=str, default='data/jesc/split/train.en')
    parser.add_argument('--tgt_path', type=str, default='data/jesc/split/train.ja')
    parser.add_argument('--src_out_path', type=str, default='data/jesc/split/train.src_to_tgt.en')
    parser.add_argument('--tgt_out_path', type=str, default='data/jesc/split/train.src_to_tgt.ja')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--align_opt', type=str, default='src_to_tgt')
    parser.add_argument('--ignore_unaligned', action='store_true')
    parser.add_argument('--remove_duplicate', action='store_true')
    args = parser.parse_args()

    reorder(args)
  

if __name__ == '__main__':
    main()

