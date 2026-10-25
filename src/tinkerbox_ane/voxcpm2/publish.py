"""Compile, stage and publish VoxCPM2 bundles to Hugging Face Hub."""

from tinkerbox_ane.publish import release_main
from . import KIND, REPO, REVISION, SOURCE, UPSTREAM_REVISION


def main(argv=None):
    source = dict(repo=REPO, revision=REVISION, code_revision=UPSTREAM_REVISION)
    return release_main('voxcpm2', KIND, SOURCE, source, 'LICENSES/VoxCPM-Apache-2.0.txt', argv)


if __name__ == '__main__':
    raise SystemExit(main())
