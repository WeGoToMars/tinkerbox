"""Compile, stage and publish d3 bundles to Hugging Face Hub: `python -m tinkerbox_ane.d3.publish MODEL ...`."""

import sys

from tinkerbox_ane.publish import release_main
from . import KIND, VARIANTS


def main(argv=None):
    model, *argv = sys.argv[1:] if argv is None else argv
    repo, revision, _ = VARIANTS[model]
    return release_main(model, KIND, None, dict(repo=repo, revision=revision), 'LICENSES/Apache-2.0.txt', argv)


if __name__ == '__main__':
    raise SystemExit(main())
