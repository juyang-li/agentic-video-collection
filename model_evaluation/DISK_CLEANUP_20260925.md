# Disk cleanup record — 2026-09-25

Purpose: free shared scratch space before the Bernini/OmniEdit evaluation while preserving reproducible inputs, inference weights, final recovery states, scores, and review artifacts.

## Capacity

- Project tree: about 296 GiB before cleanup, 248 GiB after cleanup.
- `/scratch`: 111 GiB available before cleanup, 159 GiB after cleanup.
- Node-local `/tmp`: about 28 TiB available; Bernini checkpoints and download caches should be staged there inside the Slurm job rather than persisted on `/scratch`.
- Inodes were not a constraint (`/scratch` inode use about 1%).

## Removed

- Full checkpoints `750`, `800`, `850`, and `900` from the single-4K, CoinVE-4K, and mixed-2K2K Lance runs. These were periodic saves, not the frozen `226/451/677/902` evaluation checkpoints.
- Checkpoint payloads from completed smoke/resume-verification runs; their small configs and logs remain.
- Optimizer shards from `226`, `451`, and `677` in the three formal runs. Their inference adapters and provenance files remain. Exact-resume state is retained at each run's final `902` checkpoint.
- Local `training_recovery_v1.tar` after confirming its previously recorded, size-matched Baidu copy at `/apps/bypy/lance_review/lance_4k_r64_seed42_e2_v1_training_recovery_v1.tar` (3,368,058,880 bytes, file ID `898522629005543`). Receipts and verification metadata remain locally.
- The inactive VEFX virtual environment/model cache, Stage-1 model caches, pip/CLIP caches, Python bytecode, pytest caches, and zero-byte stale lock files.

## Preserved and verified

- OmniEdit Spatial CSV has 240 data rows, and the official Spatial source tree has 240 MP4 files.
- All three formal Lance runs retain inference adapters at `226`, `451`, `677`, and `902` (12 adapters total).
- All three final `902` checkpoints retain their four optimizer shards and complete recovery metadata.
- Stage-1 reserve media, frozen training data, CoinVE ablation media, all official evaluation inputs, scores, generation outputs, and review packages were not removed.
- No active Lance, VACE, or Bernini training/inference process was present during cleanup.

## Bernini storage rule

Set Hugging Face, Torch, Triton, and temporary video caches to the Slurm job's `$TMPDIR`; copy only manifests, logs, generated videos, hashes, and final audit packages back to `/scratch`. Do not download the roughly 192 GB full Bernini-v2 checkpoint into the shared project tree.
