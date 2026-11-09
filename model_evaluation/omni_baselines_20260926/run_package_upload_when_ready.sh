#!/usr/bin/env bash
set -euo pipefail

root=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/omni_baselines_20260926
env=/tmp/omni_bernini_20260926/env
remote=lance_review/omni_baselines_20260926

while [[ ! -f "$root/vace/.tier_a_4gpu_done" || \
         ! -f "$root/bernini_r14b/inference_tier_a.done" ]]; do
  printf '%s waiting for inference completion\n' "$(date -Is)"
  sleep 60
done

"$env/bin/python" "$root/bernini_r14b/scripts/persist_outputs.py" \
  --tasks "$root/bernini_r14b/manifests/tasks_tier_a.json" \
  --output-dir "$root/bernini_r14b/outputs_raw" \
  --marker "$root/bernini_r14b/outputs_persisted.done"

"$env/bin/python" "$root/scripts/finalize_vace_manifest.py"
"$env/bin/python" "$root/scripts/finalize_bernini_manifest.py"
vace_package="$root/vace/omni24_vace14b_mask_seed42_human_review_20260926"
bernini_package="$root/bernini_r14b/omni24_bernini_r14b_seed42_human_review_20260926"
if [[ ! -d "$vace_package" && ! -f "$vace_package.zip" ]]; then
  "$env/bin/python" "$root/scripts/build_generated_review.py" vace --count 24
fi
if [[ ! -d "$bernini_package" && ! -f "$bernini_package.zip" ]]; then
  "$env/bin/python" "$root/scripts/build_generated_review.py" bernini --count 24
fi

"$env/bin/pip" install playwright==1.55.0

# The compute image may omit libgbm even though the shared Chrome binary needs
# it.  Extract the matching Ubuntu package locally; no system install or root
# privilege is required.
chrome="$root/../ivebench_lance/runtime/google-chrome-stable/opt/google/chrome/chrome"
browser_lib_root="$runtime/browser-libs/root"
if ldd "$chrome" 2>&1 | grep -q 'libgbm.so.1 => not found'; then
  mkdir -p "$runtime/browser-libs/pkg" "$browser_lib_root"
  if [[ ! -e "$browser_lib_root/usr/lib/x86_64-linux-gnu/libgbm.so.1" ]]; then
    (
      cd "$runtime/browser-libs/pkg"
      apt-get download libgbm1
      deb=$(find . -maxdepth 1 -type f -name 'libgbm1*.deb' | head -n 1)
      test -n "$deb"
      dpkg-deb -x "$deb" "$browser_lib_root"
    )
  fi
  export LD_LIBRARY_PATH="$browser_lib_root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

for package in \
  "$root/lance/omni24_lance_base_seed42_human_review_20260926" \
  "$vace_package" \
  "$bernini_package"; do
  "$env/bin/python" "$root/scripts/test_human_review.py" "$package"
  if [[ ! -f "$package.zip" ]]; then
    "$env/bin/python" "$root/scripts/finalize_review.py" "$package"
  fi
done

bypy_direct() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u ALL_PROXY -u all_proxy bypy "$@"
}

bypy_direct mkdir "$remote" || true
for archive in \
  "$root/lance/omni24_lance_base_seed42_human_review_20260926.zip" \
  "$root/vace/omni24_vace14b_mask_seed42_human_review_20260926.zip" \
  "$root/bernini_r14b/omni24_bernini_r14b_seed42_human_review_20260926.zip"; do
  remote_file="$remote/$(basename "$archive")"
  # Baidu's tmpfile slice endpoint currently returns 31064 for this app, while
  # the regular upload endpoint is valid.  A slice threshold above these ZIPs
  # selects the verified whole-file path.
  bypy_direct -s 200m -r 2 -q -t 1200 -e -f upload \
    "$archive" "$remote_file" overwrite
  meta=$(bypy_direct meta "$remote_file" '$f $s $b')
  local_size=$(stat -c '%s' "$archive")
  local_md5=$(md5sum "$archive" | awk '{print $1}')
  grep -F "$local_size" <<<"$meta" >/dev/null
  grep -F "$local_md5" <<<"$meta" >/dev/null
done
bypy_direct list "$remote"
touch "$root/.packages_uploaded"
printf '%s packages built, validated, and uploaded\n' "$(date -Is)"
