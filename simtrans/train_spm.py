from argparse import ArgumentParser

import sentencepiece as spm

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--model_prefix', type=str)
    parser.add_argument('--vocab_size', type=int, default=16000)
    parser.add_argument('--character_coverage', type=float, default=0.9995)
    parser.add_argument('--model_type', type=str, default='unigram')
    parser.add_argument('--extra_options', type=str, default=None)
    args = parser.parse_args()

    train_args = (
        f'--input={args.input} '
        f'--model_prefix={args.model_prefix} '
        f'--vocab_size={args.vocab_size} '
        f'--character_coverage={args.character_coverage} '
        f'--model_type={args.model_type}'
    )

    if args.extra_options is not None:
        train_args += '--extra_options={args.extra_options}'

    spm.SentencePieceTrainer.Train(train_args)

