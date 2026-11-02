"""Compile a GPT-2 export with the shared watchdog and full ANE verification."""

import argparse
from pathlib import Path

from tinkerbox_ane.bundle import compile_bundle
from . import KIND


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--architecture')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    compile_bundle(args.source, args.out, KIND, args.architecture, args.force)


if __name__ == '__main__':
    main()
