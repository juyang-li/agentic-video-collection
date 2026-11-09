#!/usr/bin/env python3
"""Build a score-free, fully offline 30-card benchmark audit package."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "real_person_multiturn_benchmark_v1_review"
PACKAGE = ROOT / "review" / PACKAGE_NAME
FIELDS = ("person_relevance", "instruction_feasibility", "chain_coherence", "task_diversity", "technical_quality")


def main() -> None:
    validation = json.loads((ROOT / "validation_report.json").read_text())
    if validation.get("status") != "passed": raise ValueError("benchmark validation must pass first")
    if PACKAGE.exists(): raise FileExistsError(f"refusing to overwrite {PACKAGE}")
    (PACKAGE / "media").mkdir(parents=True)
    chains = [json.loads(line) for line in (ROOT / "benchmark.jsonl").read_text().splitlines() if line.strip()]
    cards, rows = [], []
    for index, chain in enumerate(chains, 1):
        source = ROOT / chain["source"]["media"]
        destination = PACKAGE / "media" / source.name
        shutil.copy2(source, destination)
        row = {"review_id": chain["chain_id"], "chain_id": chain["chain_id"],
               **{field: "1" for field in FIELDS}, "notes": ""}
        rows.append(row)
        cards.append({"index": index, "chain_id": chain["chain_id"], "chain_type": chain["chain_type"],
                      "source_caption": chain["source"]["official_source_caption"],
                      "source_dataset": f"IVEBench {chain['source']['subset']} #{chain['source']['official_id']}",
                      "video": f"media/{source.name}", "scene": chain["scene"], "turns": chain["turns"]})
    with (PACKAGE / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    payload = {"title": "Real-person multi-turn benchmark audit", "version": PACKAGE_NAME,
               "fields": list(FIELDS), "cards": cards, "defaults": rows}
    template = (Path(__file__).with_name("review_template.html")).read_text(encoding="utf-8")
    document = template.replace("__REVIEW_DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
    (PACKAGE / "index.html").write_text(document, encoding="utf-8")
    (PACKAGE / "README.md").write_text(
        "# Real-person multi-turn benchmark audit\n\n"
        "Keep this directory intact and open `index.html` in Chrome or Edge. It is fully offline.\n\n"
        "There are 30 cards. Each shows the real source video, original caption, chain type, and all edit instructions. "
        "The five labels default to `1` (pass); change failures to `0`, uncertain cases to `2`, and add notes. "
        "Use Export JSON/CSV for the completed review; Import JSON restores prior work.\n",
        encoding="utf-8")
    (PACKAGE / "package_summary.json").write_text(json.dumps({"status": "built_pending_browser_validation", "cards": 30,
        "review_units": 30, "media_files": 30, "offline": True, "network_dependencies": 0}, indent=2) + "\n")
    print(PACKAGE)


if __name__ == "__main__": main()
