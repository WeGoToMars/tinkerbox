# ANE Copy-on-Write Staging

The native host uses APFS copy-on-write (CoW) clones to eliminate redundant copies of shared weight files during Core AI specialization on macOS 27.

## Background

Model constants frequently reference different byte offsets within a single large `resources.bin`. During on-device specialization, AppleNeuralEngine's staging service copies external weight files in an un-deduplicated per-constant loop without requesting APFS clones. When many constants share a backing file, this behavior causes repeated full-file copies and substantial I/O overhead.

Ahead-of-time (AOT) compilation does not bypass this step, as device-specific specialization still occurs on the host. This workaround intercepts compilation requests in the host process without modifying system services or changing `coreai-build`.

## Implementation

1. **Staging Directory Setup:** [host.py](../src/tinkerbox_ane/host.py) creates a temporary `.host-*` directory adjacent to the model bundle to guarantee CoW clones remain on the same APFS volume. It exports paths via `COW_CLIENT_DEST` and `COW_ASSET_PREFIX`.
2. **Method Interception:** [host.swift](../src/tinkerbox_ane/host.swift) initializes the hook before specialization begins. [cow-staging.m](../src/tinkerbox_ane/cow-staging.m) swizzles `-[_ANEClient compileModel:options:qos:error:]` in the runtime process.
3. **Deduplicated Cloning:** For each unique backing file in the request, the hook creates a single clone using `COPYFILE_CLONE_FORCE` and verifies `COPYFILE_STATE_WAS_CLONED`.
4. **Descriptor Remapping:** It updates `externConstants.FileConstants` entries to point to the cloned files while preserving byte offsets and metadata, stripping obsolete sandbox extension tokens (`_SBExtension`).
5. **Preflight Validation:** The hook executes `_ANECloneHelper` to verify descriptor integrity before forwarding the modified request to the original compilation method.

Cloned files share existing storage blocks with the source bundle. The host cleans up the staging directory on shutdown.

## Constraints and Safety Checks

- **Volume Compatibility:** The model directory must reside on an APFS volume with write permissions and cloning enabled. If cloning fails, the host aborts immediately.
- **External Resource Limits:** Requests referencing files outside the designated asset prefix are permitted only if every backing file exists and is smaller than 64 MiB (accommodating small framework auxiliary graphs).
- **Fail-Safe Operation:** Any unexpected resource size, altered preflight descriptor, missing Objective-C symbol, or compilation failure terminates the host process.
- **Privileges:** The hook operates entirely within user space and requires no elevated privileges.

## Deprecation Criteria

This workaround should be removed once AppleNeuralEngine natively deduplicates shared backing files or Core AI provides a supported staging API. Removal requires verifying that cold-cache specialization completes without redundant disk writes across the full model bundle.
