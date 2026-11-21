# Real-person multi-turn editing benchmark v2

Version 2 keeps the 30 successful real-video sources from v1 and revises the instruction chains.

Key changes:

- action never appears in round 1; its 28 turns are distributed as R2=4, R3=14, R4=9, R5=1;
- vague leaning, slight-motion, gaze-only, and head-only action edits are forbidden;
- action edits use clear gestures, visible object interactions, or source-compatible full-body actions;
- small/medium faces no longer receive fine eye makeup, tiny scars, nose studs, or light stubble;
- appearance favors visible beard styles, hats, eyewear, scarves, and strong hairstyle silhouettes when appropriate.

The video sources, chain counts, chain types, and four 28-turn axis quotas are unchanged from v1.
