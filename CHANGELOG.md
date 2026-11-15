# Changelog

## [0.1.0] - 2026-11-14
### Added
- Initialized `aska-physical/impedence-analyzers` baseline (`README.md`, `HANDOFF.md`, `CHANGELOG.md`, `verify.py`, `results/certificate.json`).
- Implemented scale-invariant Marquardt 7-frequency Cole+$T_d$ estimator (`7.14x` faster than 50-frequency sweeps with `0.9659` conditioning ratio), dual Hook-effect / stray-capacitance artifact gate, exact square-patch cell constant $2\pi a/(2-\sqrt{2})$, and 5 must-fail controls.
