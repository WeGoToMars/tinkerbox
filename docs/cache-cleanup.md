# Core AI disk cleanup

Core AI keeps a specialization of every asset it has loaded, even after the bundle or export is deleted. `scripts/clean_coreai.py` removes those and other leftovers. Close the app and stop exports, benchmarks and other Core AI clients first, and keep them stopped until it finishes.

```sh
.venv/bin/python scripts/clean_coreai.py          # preview
.venv/bin/python scripts/clean_coreai.py --apply
```

By default it deletes, for every client and macOS version:

- specializations in `~/Library/Application Support/coreai-cache`
- locks and temporary entries in `~/Library/Caches/coreai-cache`
- abandoned `.host-*` and `.compile-*` staging in `artifacts/` and the model cache
- MPSGraph scratch directories in the temp directory whose creating process has exited

`--include-build` also deletes `artifacts/build`. `--include-models` also deletes `~/Library/Caches/tinkerbox-ane`, including locally installed bundles. The next load of each model specializes it again, which takes longer.

The script never kills processes, uses sudo, follows symlinks or touches the system `com.apple.aned` caches. It works across clients because Core AI's own [`AIModelCache`](https://developer.apple.com/documentation/coreai/aimodelcache) API only sees the calling app's cache. Reported sizes are estimates, since APFS clones share blocks and snapshots can keep deleted ones.

To remove one model's files, use `python -m tinkerbox_ane.storage --remove MODEL` or the app's Models page instead.
