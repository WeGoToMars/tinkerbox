# Repository guidance

- This engine targets Apple Neural Engine, supplemental tasks can run on the CPU.
- Separate task APIs, model variants, architecture implementations, and runtime infrastructure; keep model-specific logic out of the app.
- Reuse architecture code across compatible weight variants and compose derived architectures from reusable blocks with matching contracts.
- Keep patches minimal. Prefer explicit composition over speculative abstractions, compatibility layers, or frameworks. Remove abandoned scaffolding.

## Tests

- Test public behavior: answers, ordering, batching, usage, limits, and request isolation.
- Do not test private helpers, attributes, tensor layouts, routing, masks, or host call counts.
- Do not mirror the implementation for expected results or mock the implementation being tested.
- Prefer a few meaningful end-to-end regressions; skip trivial reversible edits and exhaustive matrices.
- Leave unrelated tests alone; update fixtures when supported contracts change.
- Check affected Swift targets and real worker/App Intent behavior for app changes; Python tests alone are insufficient.
- Run appropriate checks; repeat or broaden only for failures or unresolved concerns.

## ANE validation

- Use the existing compiler and watchdog; retain full ANE placement verification.
- Compile only needed architectures. Standard M4 is `h16g`.
- Report the device and bundle tested; distinguish cross-compilation from on-device correctness/performance and existing bundles from changed exports.
