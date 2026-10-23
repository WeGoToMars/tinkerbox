# Core AI disk cleanup

Preview the user-level caches and abandoned staging directories, then close the
app and stop benchmarks, model exports, and compiler clients before applying:

```sh
.venv/bin/python scripts/clean_coreai.py
.venv/bin/python scripts/clean_coreai.py --apply
```

The default plan covers **all client scopes and OS versions** in
`~/Library/Application Support/coreai-cache` (persistent specializations),
`~/Library/Caches/coreai-cache` (locks and temporary entries), abandoned
`.host-*`/`.compile-*` staging under this repository's artifacts and Tinkerbox
model cache, and MPSGraph scratch directories in the current user's macOS temp
directory whose creating PID has exited. Persistent specializations can remain
after their source export or build directory is deleted.

This is an offline disk maintenance script: it removes those specific user-owned
cache directories while clients are stopped. It does not call
`AIModelCache.default.deleteAll()` from a new helper executable, because that API
is [scoped to the calling app](https://developer.apple.com/documentation/coreai/aimodelcache)
and would miss other hosts' caches and old OS versions. For an individual model
in a running host, use Apple's
[`deleteEntry(for:options:)` or `deleteEntries(for:)`](https://developer.apple.com/documentation/coreai/managing-model-specialization-and-caching)
instead. The next model load recreates its specialization and may take longer.

Add optional targets explicitly (preview first):

```sh
.venv/bin/python scripts/clean_coreai.py --include-build --include-models
.venv/bin/python scripts/clean_coreai.py --include-build --include-models --apply
```

`--include-build` deletes this checkout's `artifacts/build`.
`--include-models` deletes `~/Library/Caches/tinkerbox-ane`, including legacy
downloads and locally installed bundles; reinstall local bundles or download
published ones again afterward. Other artifacts, research evidence, source
checkpoints in Hugging Face, environments, and uv/pip caches are preserved.
The script never kills processes, invokes sudo, changes permissions, or touches
protected `/Library/Caches/com.apple.aned` and `com.apple.aneuserd` directories,
and it never deletes through a symlinked target.

Allocated cache sizes, in decimal GB, are an estimate: APFS clones can share
blocks, open files can defer recovery, and snapshots can retain deleted blocks.
A dry run is read-only. Apply checks common Core AI clients before deleting, but
cannot prevent an external process from starting concurrently; keep clients
stopped for the whole cleanup.
