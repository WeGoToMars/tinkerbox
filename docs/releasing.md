# Releasing

`uv run python scripts/release.py` checks that every catalog model this version ships has tagged bundles on Hugging Face, runs all tests on the Neural Engine under the watchdog (any skipped test fails the release), and creates the version tag. Push the branch and tag as it instructs.

Then run **Publish to PyPI** in GitHub Actions, from `main`:

```sh
gh workflow run publish.yml --ref main -f tag=vX.Y.Z
```

It builds the source archive and the Apple Silicon wheel with Xcode 27 and publishes them with Trusted Publishing. The wheel is tagged macOS 27, which the runtime needs. Build, sign and upload the app separately ([app guide](../app/README.md#development)).

One-time setup: create a GitHub environment named `pypi`, and add a [PyPI Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) for `tinkerbox-ane` with owner `WeGoToMars`, repository `tinkerbox`, workflow `publish.yml` and environment `pypi`.
