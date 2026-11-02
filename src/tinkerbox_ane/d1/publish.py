"""Compile, stage and publish d1 bundles to Hugging Face Hub."""

from tinkerbox_ane.publish import release_main
from . import KIND, MODEL, REPO, REVISION, SOURCE


def main(argv=None):
    return release_main(MODEL, KIND, SOURCE, dict(repo=REPO, revision=REVISION),
                        'LICENSES/LFM-Open-License-v1.0.txt', argv)


if __name__ == '__main__':
    raise SystemExit(main())
