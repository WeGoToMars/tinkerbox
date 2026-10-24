"""Compile, stage and publish OmniVoice bundles to Hugging Face Hub."""

import argparse
from pathlib import Path

from tinkerbox_ane import publish
from tinkerbox_ane.bundle import read_bundle
from tinkerbox_ane.host import SUPPORTED_ARCHITECTURES
from tinkerbox_ane.models import entry
from . import KIND, REPO, REVISION, SOURCE
from .export import compile


def main(argv=None):
    item = entry('omnivoice')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--staging', type=Path,
                        default=Path('artifacts/huggingface') / item['repo'].split('/')[-1])
    parser.add_argument('--repo', default=item['repo'])
    parser.add_argument('--tag', default=item['revision'])
    parser.add_argument('--arch', nargs='+', choices=list(SUPPORTED_ARCHITECTURES),
                        default=item['architectures'])
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--metadata', type=Path, help='Directory with release README.md, LICENSE and NOTICE')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--no-upload', action='store_true', help='Compile and stage without uploading')
    modes.add_argument('--upload-only', action='store_true', help='Publish staged bundles')
    args = parser.parse_args(argv)
    source = {'repo': REPO, 'revision': REVISION}
    if not args.upload_only:
        if read_bundle(args.source, KIND)['source'] != source:
            raise ValueError('export source does not match OmniVoice')
        publish.build_architectures(compile, args.source, args.staging / 'macos27', args.arch, force=args.force)
    metadata = dict(license_file='LICENSE', expected_source=source, architectures=args.arch,
                    metadata_dir=args.metadata or args.staging)
    if args.no_upload:
        publish.prepare_release(args.staging, KIND, **metadata)
        print(f'Release staged in {args.staging}')
    else:
        commit = publish.upload_release(args.repo, args.staging, KIND, args.tag, **metadata)
        print(f'Published {args.repo} at {commit}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
