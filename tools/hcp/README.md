# hcp

Headless parity checker for polygon geometry content hashing.

## What it checks
- `recompute(..., sequential)` as baseline
- `recompute(..., <selected mode>)` equals baseline hash
- `computeMissing(..., <selected mode>)` preserves pre-set hashes and equals baseline hash
- timing logs for baseline, recompute and computeMissing

## Args
- `--runtime-tuning <sequential|heuristic|hybrid>` (alias: `none` -> `sequential`, default: `heuristic`)
- `--buffer-bytes <N>` (minimum: `2097152`)
- `--seed <U64>` (deterministic payload seed)

## Example
`./hcp_d.exe --runtime-tuning heuristic --buffer-bytes 67108864 --seed 12345`

## CTest
`ctest --output-on-failure -C Debug -R NBL_HCP`
