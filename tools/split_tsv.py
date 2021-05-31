from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output_src', type=str)
    parser.add_argument('--output_tgt', type=str)
    parser.add_argument('--split_symbol', type=str, default='\t')

    args = parser.parse_args()

    with open(args.input) as f:
        lines = [l.strip().split(args.split_symbol) for l in f.readlines()]

    with open(args.output_src, 'w') as f:
        for l in lines:
            f.write(f'{l[0]}\n')

    with open(args.output_tgt, 'w') as f:
        for l in lines:
            f.write(f'{l[1]}\n')

if __name__ == '__main__':
    main()
