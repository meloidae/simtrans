from argparse import ArgumentParser


def get_line(path: str, line_number: int):
    with open(path, encoding='UTF-8') as f:
        line = f.readlines()[line_number].strip()
    return line


def write_line_with_repeat(path: str, line: str, repeat: int):
    with open(path, mode='w', encoding='UTF-8') as f:
        for i in range(repeat):
            f.write(f'{line}\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--src_path', type=str)
    parser.add_argument('-t', '--tgt_path', type=str)
    parser.add_argument('-o', '--out_prefix', type=str)
    parser.add_argument('-l', '--line_number', type=int, default=0)
    parser.add_argument('-r', '--repeat', type=int, default=1000)
    args = parser.parse_args()

    src_line = get_line(args.src_path, args.line_number)
    tgt_line = get_line(args.tgt_path, args.line_number)

    src_out_path = f'{args.out_prefix}.src'
    tgt_out_path = f'{args.out_prefix}.tgt'

    write_line_with_repeat(src_out_path, src_line, args.repeat)
    write_line_with_repeat(tgt_out_path, tgt_line, args.repeat)
