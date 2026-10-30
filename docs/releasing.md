# Releasing

Run `uv run python scripts/release.py` to check model bundles, run tests and create
the version tag. Push the branch and tag as instructed.

In GitHub Actions, run **Publish to PyPI** and enter the tag, or use:

```sh
gh workflow run publish.yml --ref main -f tag=v2.2.0
```

The workflow must be merged into `main` first. It builds the source archive and
Apple Silicon wheel with Xcode 27, then publishes using Trusted Publishing.
The wheel's macOS 26 tag is for PyPI compatibility; the runtime requires macOS 27.
Build and upload the signed app separately using the [app guide](../app/README.md#building-from-source).

One-time setup: create a GitHub environment named `pypi`, then add a
[PyPI Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
for `tinkerbox-ane` with owner `WeGoToMars`, repository `tinkerbox`, workflow
`publish.yml` and environment `pypi`. No API token is needed.
