"""Compile, stage and publish OmniVoice bundles to Hugging Face Hub."""

from tinkerbox_ane.publish import release_main
from . import KIND, REPO, REVISION, SOURCE


def main(argv=None):
    return release_main('omnivoice', KIND, SOURCE, dict(repo=REPO, revision=REVISION), 'LICENSE', argv)


if __name__ == '__main__':
    raise SystemExit(main())
