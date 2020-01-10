import parseSTM as ps
import sys, os
import argparse

def _parseargs():
    parser = argparse.ArgumentParser(
        description="Extract image buffers from Lyding STM file\n\n",
        epilog="Developed by Huy Nguyen, Gruebele-Lyding Groups\n"
        "University of Illinois at Urbana-Champaign\n",
    )
    parser.add_argument(
        "input",
        nargs="*",
        type=str,
        help="Contact lyding@illinois.edu for information about Lyding STM file format",
    )
    args = parser.parse_args()
    return args

def main():
    args = _parseargs()
    for f in args.input:
        stmfile = ps.STMfile(f)
        stmfile.toGwy()

if __name__ == "__main__":
    main()
