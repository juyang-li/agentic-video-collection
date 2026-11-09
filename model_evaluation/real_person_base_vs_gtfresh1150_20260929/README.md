# Real-person benchmark: Lance base vs gtfresh step1150

This run compares the official `Lance_3B_Video` base model with the senior
student's `gtfresh-v1 step1150` LoRA on the 30-chain real-person multi-turn
benchmark v2.

- Both models use identical preprocessing and inference settings: 30 diffusion
  steps, timestep shift 3.5, text CFG 1.0, global/noise seed 42.
- Evaluation is autoregressive: every generated round is fed into the next
  round by Lance's native `MultiTurnChatEngine`.
- Four H200 GPUs are used as two shards per model. Chain-index parity gives
  balanced shards of 55 and 57 generated videos per model.
- No Gemini or other automatic semantic judge is used. The final deliverable is
  an offline human-review package.

Outputs are resumable at completed-chain granularity. A completed chain has a
`case_status.json`, transcript, and every expected `turnNN.mp4`.

