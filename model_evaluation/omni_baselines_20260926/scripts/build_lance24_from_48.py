#!/usr/bin/env python3
"""Create a Tier-A Lance package aligned to the 24-sample model runs."""

import csv
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "lance/omni48_lance_base_seed42_human_review_20260926"
NAME = "omni24_lance_base_seed42_human_review_20260926"
DEST = ROOT / "lance" / NAME

if DEST.exists() or DEST.with_suffix(".zip").exists():
    raise FileExistsError(DEST)

with (SOURCE / "review_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
    rows = [row for row in csv.DictReader(stream) if row["tier"] == "A"]
if len(rows) != 24 or len({row["category"] for row in rows}) != 24:
    raise RuntimeError("Lance Tier-A selection is not 24 unique categories")
ids = {row["eval_id"] for row in rows}
generation = [json.loads(line) for line in (SOURCE / "generation_manifest.jsonl").read_text().splitlines() if line]
generation = [row for row in generation if row["eval_id"] in ids]
inventory = json.loads((SOURCE / "media_inventory.json").read_text())
inventory = [row for row in inventory if row["eval_id"] in ids]

(DEST / "media").mkdir(parents=True)
for row in rows:
    for key in ("review_source_path", "review_generated_path"):
        rel = Path(row[key])
        target = DEST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / rel, target)
with (DEST / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
with (DEST / "generation_manifest.jsonl").open("w", encoding="utf-8") as stream:
    for row in generation:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
(DEST / "media_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
(DEST / "summary.json").write_text(json.dumps({
    "status": "complete", "review_type": "human_only_unlabeled", "sample_count": 24,
    "category_count": 24, "samples_per_category": 1, "tiers": {"A": 24, "B": 0},
    "model_id": "Lance", "automated_judge_included": False,
}, ensure_ascii=False, indent=2) + "\n")
items = [{
    "eval_id": row["eval_id"], "tier": row["tier"], "category": row["category"],
    "instruction": row["instruction"],
    "panels": [
        {"label": "SOURCE · OFFICIAL INPUT", "path": row["review_source_path"]},
        {"label": "LANCE BASE · MODEL OUTPUT", "path": row["review_generated_path"]},
    ],
} for row in rows]
template = (ROOT / "scripts/human_review.html").read_text(encoding="utf-8")
payload = {"review_version": NAME, "title": "Lance Base · OmniEdit Spatial 24 · 人工审核", "items": items}
document = template.replace("__REVIEW_DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
(DEST / "index.html").write_text(document, encoding="utf-8")
(DEST / "README.md").write_text(
    "# Lance Base · OmniEdit Spatial 24 人工审核包\n\n完整解压后打开 `index.html`；本包完全离线。\n\n"
    "本包为与 VACE/Bernini 对齐的 Tier A 24 类子集。审核项初始均为空；1=通过，0=不通过，2=存疑。\n",
    encoding="utf-8",
)
print(json.dumps({"status": "built_pending_browser_validation", "package": str(DEST), "samples": 24}))
