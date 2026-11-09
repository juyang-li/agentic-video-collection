# Real-person multi-turn editing benchmark v1

This benchmark contains 30 fresh IVEBench real-video sources and 112 English edit instructions. It is person-centric and intended for autoregressive multi-turn evaluation: round `n` consumes the output from round `n-1`. It contains no target videos and does not claim ground-truth pixel supervision.

Frozen composition:

- lengths: 13 three-turn, 12 four-turn, 5 five-turn chains;
- chain structures: 18 progressive, 6 repeated-axis, 6 rollback/history chains;
- edit axes: appearance, outfit, action, and camera, exactly 28 turns each.

`benchmark.jsonl` is the canonical machine-readable benchmark, `turns.csv` is a flat view, and `source_inventory.csv` records source/media provenance and hashes. `metadata/` contains the frozen blueprints, API receipts, raw provider responses, and resumable build state. The API builder accepts only `LIJUYANG_ICLR`, Qwen3.8-Omni-Flash, and the Alibaba provider; it does not use a personal credential or silently fall back.

Build and validation use the existing project environment:

```bash
/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python scripts/build_benchmark.py --dry-run
/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python scripts/build_benchmark.py
/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python scripts/validate_benchmark.py
/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python scripts/build_review.py
# run test_review.py from an environment containing Playwright
/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python scripts/finalize_review.py
```

The review package is entirely offline. Its five labels default to `1` (pass); use `0` for failure and `2` for uncertainty.
