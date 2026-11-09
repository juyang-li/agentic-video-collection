#!/usr/bin/env bash
set -euo pipefail

ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")
DEST="$ROOT/models"
mkdir -p "$DEST/amt" "$DEST/vtss" "$DEST/videoclip" "$DEST/cotracker" "$DEST/groundingdino"

fetch() {
    local url=$1
    local destination=$2
    local expected_sha=$3
    if [[ -s "$destination" ]]; then
        local current_sha
        current_sha=$(sha256sum "$destination" | awk '{print $1}')
        if [[ -z "$expected_sha" || "$current_sha" == "$expected_sha" ]]; then
            printf 'reused %s sha256=%s\n' "$destination" "$current_sha"
            return
        fi
        mv "$destination" "$destination.bad.$(date +%s)"
    fi
    curl -L --fail --show-error --retry 5 --retry-all-errors --continue-at - \
        --output "$destination.part" "$url"
    local actual_sha
    actual_sha=$(sha256sum "$destination.part" | awk '{print $1}')
    if [[ -n "$expected_sha" && "$actual_sha" != "$expected_sha" ]]; then
        printf 'SHA256 mismatch for %s: %s != %s\n' "$destination" "$actual_sha" "$expected_sha" >&2
        exit 1
    fi
    mv "$destination.part" "$destination"
    printf 'downloaded %s sha256=%s\n' "$destination" "$actual_sha"
}

fetch \
  "https://huggingface.co/lalala125/AMT/resolve/main/amt-g.pth" \
  "$DEST/amt/amt-g.pth" \
  "a39af8744e8e6db16ebd29cb3dd09596fae38f360d8bb024288b3fe468256d07" &
pid_amt=$!
fetch \
  "https://huggingface.co/Koala-36M/Training_Suitability_Assessment/resolve/main/infer.pth" \
  "$DEST/vtss/infer.pth" \
  "1ddef099b07f63863091c46fc4992c58e5d18ee1663f26536ac04fb42e49d9f0" &
pid_vtss=$!
fetch \
  "https://huggingface.co/alibaba-pai/VideoCLIP-XL-v2/resolve/main/VideoCLIP-XL-v2.bin" \
  "$DEST/videoclip/VideoCLIP-XL-v2.bin" \
  "4b51b9311c2cd38a6f35f705a4cc6076764d0a9f6164b67b945b03a82fb79123" &
pid_videoclip=$!
fetch \
  "https://huggingface.co/facebook/cotracker3/resolve/main/baseline_offline.pth" \
  "$DEST/cotracker/baseline_offline.pth" \
  "da09bbac871f7398e5b29c4de5213652658949737bc158840b101678ba8ad1df" &
pid_cotracker=$!
fetch \
  "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth" \
  "$DEST/groundingdino/groundingdino_swinb_cogcoor.pth" \
  "" &
pid_grounding=$!

trap 'kill "$pid_amt" "$pid_vtss" "$pid_videoclip" "$pid_cotracker" "$pid_grounding" 2>/dev/null || true' INT TERM
wait "$pid_amt" "$pid_vtss" "$pid_videoclip" "$pid_cotracker" "$pid_grounding"
sha256sum \
  "$DEST/amt/amt-g.pth" \
  "$DEST/vtss/infer.pth" \
  "$DEST/videoclip/VideoCLIP-XL-v2.bin" \
  "$DEST/cotracker/baseline_offline.pth" \
  "$DEST/groundingdino/groundingdino_swinb_cogcoor.pth"
