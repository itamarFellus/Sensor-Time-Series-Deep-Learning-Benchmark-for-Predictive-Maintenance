# AGENTS.md

- Make the smallest change that completes the requested task. Do not build future milestones speculatively.
- Preserve the existing `src/sensor_rul` structure and favor simple, readable Python with `pathlib`, type hints, and focused functions.
- For data processing, split by engine and fit preprocessing only on training data; never use test RUL values as model inputs.
- Inspect relevant files before editing. After editing, run the narrowest available check and report what changed and what was not tested.
- Ask before adding dependencies, changing public interfaces, or reorganizing files.
- Briefly explain non-obvious logic, assumptions, and tensor/DataFrame shapes so the code remains reviewable.
- Reuse code whenever possible
- Do not install anything without my permission.