from teccl.cli import make_handle_solve

from argparse import SUPPRESS, ArgumentDefaultsHelpFormatter, ArgumentParser
import argcomplete
import sys


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='TE-CCL Collective Communication Schedule Generator')

    cmd_parsers = parser.add_subparsers(title='command', dest='command')
    cmd_parsers.required = True
    handlers = []
    handlers.append(make_handle_solve(cmd_parsers))

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    for handler in handlers:
        if handler(args, args.command):
            break

if __name__ == '__main__':
    main()