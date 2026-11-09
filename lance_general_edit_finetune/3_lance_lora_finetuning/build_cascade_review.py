#!/usr/bin/env python3
"""Build a post-run stratified audit bundle from a completed rescue cascade."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


HERE = Path(__file__).resolve().parent
GENERAL_ROOT = HERE.parent
SPECIALIST_ROOT = GENERAL_ROOT / "1_specialist_filtering"
VIDEO_LINE_RE = re.compile(r"^\s*Stream .* Video:.*$", re.MULTILINE)
VIDEO_CODEC_RE = re.compile(r"Video:\s*([^,\s]+)")
YUV420_RE = re.compile(r"\byuvj?420p\b")
BROWSER_TRANSCODE = "libx264_crf17_yuv420p_faststart_noaudio"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def resolve_media(value: str) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [SPECIALIST_ROOT / path, GENERAL_ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"media not found: {value}")


def coverage_features(row: dict[str, str]) -> set[str]:
    """Return deterministic audit strata; sample identity is deliberately excluded."""
    features: set[str] = set()
    weighted_fields = {
        "dataset": 8,
        "branch": 16,
        "cascade_route": 8,
        "cascade_status": 8,
        "qwen_decision": 4,
        "gemini_decision": 4,
        "qwen_instruction_following": 2,
        "qwen_visible_artifact": 2,
        "qwen_content_preservation": 2,
        "gemini_instruction_following": 3,
        "gemini_visible_artifact": 3,
        "gemini_content_preservation": 3,
        "benchmark_stratum": 4,
        "evaluation_split": 4,
        "audit_category": 8,
    }
    for field, weight in weighted_fields.items():
        value = row.get(field, "").strip()
        if value:
            features.add(f"{weight}:{field}:{value}")
    for field in ("cascade_risk_reasons", "qwen_reason_codes_json", "gemini_reason_codes_json"):
        raw = row.get(field, "").strip()
        if not raw:
            continue
        try:
            values = json.loads(raw)
            if not isinstance(values, list):
                values = [values]
        except json.JSONDecodeError:
            values = [part.strip() for part in raw.split("|") if part.strip()]
        for value in values:
            features.add(f"3:{field}:{value}")
    return features


def coverage_pick(
    rows: list[dict[str, str]], count: int, excluded: set[str]
) -> list[dict[str, str]]:
    """Greedily maximize audit-stratum coverage with stable sample-id tie breaking."""
    remaining = sorted(
        (row for row in rows if row["sample_id"] not in excluded),
        key=lambda row: row["sample_id"],
    )
    selected: list[dict[str, str]] = []
    covered: set[str] = set()
    branch_counts: Counter[str] = Counter()
    while len(selected) < count and remaining:
        best_index = 0
        best_score: tuple[int, int, str] | None = None
        for index, row in enumerate(remaining):
            features = coverage_features(row)
            novelty = sum(int(feature.split(":", 1)[0]) for feature in features - covered)
            branch = row.get("branch", "") or row.get("dataset", "")
            score = (novelty - 4 * branch_counts[branch], -branch_counts[branch], row["sample_id"])
            if best_score is None or score > best_score:
                best_index, best_score = index, score
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        covered.update(coverage_features(chosen))
        branch_counts[chosen.get("branch", "") or chosen.get("dataset", "")] += 1
    if len(selected) != count:
        raise RuntimeError(f"need {count} rows but only found {len(selected)}")
    return selected


def review_group(row: dict[str, str]) -> str:
    route = row.get("cascade_route", "")
    final = row.get("cascade_status", "")
    qwen = row.get("qwen_decision", "")
    gemini = row.get("gemini_decision", "")
    if (
        final == "quarantine"
        or qwen in {"uncertain", "error"}
        or "stage1_boundary" in row.get("cascade_risk_reasons", "")
        or "robot_or_synthetic_subject" in row.get("cascade_risk_reasons", "")
    ):
        return "boundary_or_quarantine"
    if final == "accepted" and route == "qwen_accept_low_risk":
        return "direct_qwen_accept"
    if final == "accepted" and qwen == "rejected" and gemini == "accepted":
        return "qwen_reject_gemini_rescue"
    if final == "accepted":
        return "gemini_reviewed_accept"
    if final == "rejected":
        return "gemini_final_reject"
    return "boundary_or_quarantine"


def sha256_prefix(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def find_ffmpeg(explicit: str | None) -> str:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ffmpeg not found: {path}")
        return str(path.resolve())
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise RuntimeError(
            "ffmpeg is required for browser-compatible review media; "
            "pass --ffmpeg or install imageio-ffmpeg"
        ) from error


def probe_video(ffmpeg: str, path: Path) -> tuple[str, str]:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    line_match = VIDEO_LINE_RE.search(result.stderr)
    if line_match is None:
        raise RuntimeError(f"cannot probe video stream: {path}")
    line = line_match.group(0)
    codec_match = VIDEO_CODEC_RE.search(line)
    if codec_match is None:
        raise RuntimeError(f"cannot identify video codec: {path}")
    pixel_match = YUV420_RE.search(line)
    return codec_match.group(1).lower(), pixel_match.group(0).lower() if pixel_match else "other"


def prepare_review_media(
    task: tuple[int, dict[str, str], str], output: Path, ffmpeg: str
) -> tuple[dict[str, str], str, str, str, str, str]:
    index, row, role = task
    source = resolve_media(row[f"{role}_local"])
    digest = sha256_prefix(source)
    destination = output / "media" / f"{index:03d}_{role}_{digest}{source.suffix.lower()}"
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)
    source_codec, source_pixel_format = probe_video(ffmpeg, destination)
    transcode = "copied_browser_compatible"
    if source_codec != "h264" or source_pixel_format not in {"yuv420p", "yuvj420p"}:
        temporary = destination.with_name(destination.stem + ".browser.part.mp4")
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(destination),
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-threads",
                "2",
                str(temporary),
            ],
            check=True,
        )
        temporary.replace(destination)
        transcode = BROWSER_TRANSCODE
    review_codec, review_pixel_format = probe_video(ffmpeg, destination)
    if review_codec != "h264" or review_pixel_format not in {"yuv420p", "yuvj420p"}:
        raise RuntimeError(f"review media is not browser-compatible H.264: {destination}")
    return (
        row,
        role,
        f"media/{destination.name}",
        source_codec,
        source_pixel_format,
        transcode,
    )


def copy_media(
    output: Path, selected: list[dict[str, str]], ffmpeg: str, workers: int
) -> None:
    media = output / "media"
    media.mkdir(parents=True, exist_ok=True)
    tasks = [
        (index, row, role)
        for index, row in enumerate(selected, 1)
        for role in ("source", "target")
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(
            lambda task: prepare_review_media(task, output, ffmpeg), tasks
        )
        for row, role, review_path, source_codec, source_pixel_format, transcode in results:
            row[f"review_{role}_path"] = review_path
            row[f"review_{role}_source_codec"] = source_codec
            row[f"review_{role}_source_pixel_format"] = source_pixel_format
            row[f"review_{role}_codec"] = "h264"
            row[f"review_{role}_pixel_format"] = "yuv420p"
            row[f"review_{role}_transcode"] = transcode


def build_html(
    output: Path,
    rows: list[dict[str, str]],
    title: str,
    subtitle: str,
    review_version: str,
    model_evaluation: bool = False,
) -> None:
    survey = {
        "review_version": review_version,
        "title": title,
        "subtitle": subtitle,
        "items": rows,
        "model_evaluation": model_evaluation,
    }
    payload = json.dumps(survey, ensure_ascii=False).replace("</", "<\\/")
    document = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__PAGE_TITLE__</title>
  <style>
    :root { color-scheme:dark; --bg:#101218; --panel:#1a1f28; --panel2:#121720; --muted:#99a5b5; --line:#343c49; --good:#35a76f; --bad:#df5963; --warn:#d3a13c; --accent:#70a9ff; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:#edf2f8; font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
    header { position:sticky; top:0; z-index:10; display:flex; gap:10px; align-items:center; flex-wrap:wrap; padding:10px 16px; background:#131720f2; border-bottom:1px solid var(--line); backdrop-filter:blur(10px); }
    header strong { white-space:nowrap; }
    header .io { display:flex; gap:7px; margin-left:auto; }
    progress { width:210px; height:12px; }
    button,textarea { font:inherit; }
    button { color:#edf2f8; background:#282f3a; border:1px solid #424c5c; border-radius:7px; padding:7px 11px; cursor:pointer; }
    button:hover { background:#343d4a; }
    button.primary { background:#285fa9; border-color:#4380d3; }
    button.active { outline:2px solid var(--accent); }
    button.good.active { outline-color:var(--good); background:#1c5138; }
    button.bad.active { outline-color:var(--bad); background:#59272e; }
    button.warn.active { outline-color:var(--warn); background:#58451f; }
    main { max-width:1500px; margin:auto; padding:14px 16px; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:13px 14px; margin-bottom:11px; }
    .headline { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:8px; }
    .headline h1 { font-size:17px; margin:0; }
    .subtitle { color:var(--muted); font-size:13px; }
    .tags { display:flex; gap:6px; flex-wrap:wrap; }
    .tag { padding:3px 8px; border-radius:999px; border:1px solid #465163; background:#242b36; font-size:13px; }
    .tag.good { border-color:#277b50; color:#8fe0b5; }
    .tag.bad { border-color:#87424a; color:#ff9ea5; }
    .tag.warn { border-color:#80662e; color:#f2c96c; }
    .instruction { padding:11px 13px; background:var(--panel2); border-radius:7px; white-space:pre-wrap; font-size:17px; }
    .instruction .label { color:var(--muted); font-size:12px; margin-right:9px; }
    .video-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    .video-card { min-width:0; }
    .video-label { display:flex; justify-content:space-between; gap:8px; margin:0 2px 5px; color:var(--muted); font-weight:650; letter-spacing:.03em; }
    .video-role { font-size:12px; font-weight:500; color:#cad4e2; }
    video { display:block; width:100%; height:min(58vh,560px); object-fit:contain; background:#000; border-radius:8px; }
    .models { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:9px; }
    .model { background:var(--panel2); border:1px solid var(--line); border-radius:8px; padding:10px; min-width:0; }
    .model h3 { margin:0 0 7px; font-size:14px; }
    .checks { color:#cbd5e1; font-size:13px; margin-bottom:7px; }
    .evidence { color:#aeb8c7; white-space:pre-wrap; overflow-wrap:anywhere; font-size:12px; max-height:145px; overflow:auto; }
    .gold { border-color:#725fa7; }
    .status-row { display:flex; gap:8px; margin:5px 0 9px; flex-wrap:wrap; align-items:center; }
    textarea { width:100%; min-height:68px; color:#edf2f8; background:#10151d; border:1px solid var(--line); border-radius:7px; padding:8px; }
    .footer-actions { display:flex; align-items:center; justify-content:space-between; gap:10px; margin:14px 0 34px; }
    .muted { color:var(--muted); }
    .timer { font-variant-numeric:tabular-nums; white-space:nowrap; }
    code { color:#c9d7e9; }
    @media (max-width:950px) { .video-grid,.models { grid-template-columns:1fr; } video { height:auto; max-height:62vh; } progress { width:120px; } header .io { margin-left:0; } }
  </style>
</head>
<body>
<header>
  <strong id="headerTitle"></strong>
  <progress id="progress" value="0" max="1"></progress>
  <span id="progressText"></span>
  <span class="timer" id="timer">00:00:00</span>
  <button id="jumpBtn">跳转</button>
  <div class="io">
    <button id="exportCsv">导出 CSV</button>
    <button id="exportJson">备份 JSON</button>
    <button id="importBtn">导入</button>
    <input id="importFile" type="file" accept=".csv,.json" hidden>
  </div>
</header>

<main>
  <section class="panel">
    <div class="headline"><h1 id="sampleTitle"></h1><span id="sampleId" class="subtitle"></span></div>
    <div id="metaTags" class="tags"></div>
    <div id="subtitle" class="subtitle" style="margin-top:7px"></div>
  </section>

  <section class="panel">
    <div class="instruction"><span class="label">INSTRUCTION</span><span id="instruction"></span></div>
  </section>

  <section class="panel">
    <div class="video-grid">
      <div class="video-card">
        <div class="video-label"><span>SRC</span><span id="sourceRole" class="video-role"></span></div>
        <video id="sourceVideo" controls muted loop playsinline preload="metadata"></video>
      </div>
      <div class="video-card">
        <div class="video-label"><span id="rightVideoLabel">TARGET</span><span id="targetRole" class="video-role"></span></div>
        <video id="targetVideo" controls muted loop playsinline preload="metadata"></video>
      </div>
    </div>
  </section>

  <section class="panel">
    <div id="modelCards" class="models"></div>
  </section>

  <section class="panel">
    <div class="status-row">
      <strong>人工结论</strong>
      <button class="status good" data-status="accepted">1 accepted</button>
      <button class="status bad" data-status="rejected">0 rejected</button>
      <button class="status warn" data-status="uncertain">2 uncertain</button>
      <span class="muted">空格同步播放；左右方向键翻页</span>
    </div>
    <textarea id="notes" placeholder="可选备注"></textarea>
  </section>

  <div class="footer-actions">
    <button id="previous">← 上一条</button>
    <span id="saveState" class="muted">已自动保存</span>
    <button id="next" class="primary">完成本条并继续 →</button>
  </div>
</main>

<script id="surveyData" type="application/json">__SURVEY_DATA__</script>
<script>
(() => {
  "use strict";
  const survey = JSON.parse(document.getElementById("surveyData").textContent);
  const items = survey.items;
  const storageKey = `dark-pair-review:${survey.review_version}`;
  const $ = id => document.getElementById(id);
  const sourceVideo = $("sourceVideo"), targetVideo = $("targetVideo");
  let state = loadState();
  let index = Math.min(Number(state.index || 0), Math.max(0, items.length - 1));
  let enteredAt = Date.now();
  let syncLock = false;

  function esc(value) { return String(value ?? "").replace(/[&<>\"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
  function emptyReview() { return {human_decision:"",human_notes:"",seconds:0,updated_at:""}; }
  function freshState() { return {version:survey.review_version,index:0,reviews:{}}; }
  function loadState() {
    try {
      const parsed = JSON.parse(localStorage.getItem(storageKey) || "null");
      return parsed && parsed.version === survey.review_version ? parsed : freshState();
    } catch (_) { return freshState(); }
  }
  function currentReview() {
    const id = items[index].sample_id;
    if (!state.reviews[id]) state.reviews[id] = emptyReview();
    return state.reviews[id];
  }
  function save() {
    state.index = index;
    localStorage.setItem(storageKey, JSON.stringify(state));
    $("saveState").textContent = "已自动保存 " + new Date().toLocaleTimeString();
    updateProgress();
  }
  function accountTime() {
    const review = currentReview();
    review.seconds = Number(review.seconds || 0) + Math.max(0, Math.round((Date.now() - enteredAt) / 1000));
    enteredAt = Date.now();
  }
  function collect() {
    const review = currentReview();
    review.human_notes = $("notes").value;
    review.updated_at = new Date().toISOString();
  }
  function decisionClass(value) {
    if (value === "accepted" || value === "pass") return "good";
    if (value === "rejected" || value === "fail") return "bad";
    return "warn";
  }
  function tag(label, value, css="") {
    if (value === undefined || value === null || value === "") return "";
    return `<span class="tag ${css}">${esc(label)}=${esc(value)}</span>`;
  }
  function sideRole(item, side) {
    const explicit = side === "source" ? item.audit_left_label : item.audit_right_label;
    if (explicit) return explicit;
    const provenance = item[`${side}_provenance`] || "unresolved";
    if (provenance === "original") return "REFERENCE_VIDEO";
    if (provenance === "edited") return "GENERATED_VIDEO";
    return "ROLE_UNRESOLVED";
  }
  function generatedPosition(item) {
    if (item.audit_generated_position) return item.audit_generated_position;
    if (item.generated_role) return String(item.generated_role).toUpperCase();
    const src = item.source_provenance, tar = item.target_provenance;
    if (src === "edited" && tar === "original") return "SRC";
    if (src === "original" && tar === "edited") return "TARGET";
    if (src === "edited" && tar === "edited") return "BOTH";
    return "UNRESOLVED";
  }
  function evidence(item, prefix) {
    const raw = item[`${prefix}_evidence_json`] || "";
    let parsed = raw;
    try {
      const object = JSON.parse(raw);
      parsed = Object.entries(object).map(([key,value]) => `${key}: ${value}`).join("\n");
    } catch (_) {}
    const reasons = item[`${prefix}_reason_codes_json`] || "";
    return [reasons ? `reason_codes: ${reasons}` : "", parsed].filter(Boolean).join("\n");
  }
  function modelCard(name, decision, checks, body, extraClass="") {
    const shown = decision || "not_routed";
    return `<div class="model ${extraClass}"><h3>${esc(name)} ${tag("decision",shown,decisionClass(shown))}</h3><div class="checks">${checks}</div><div class="evidence">${esc(body || "—")}</div></div>`;
  }
  function renderModels(item) {
    if (survey.model_evaluation) {
      const checks = ["accuracy", "preservation", "realism", "consistency"]
        .map(key => `${esc(key)}=${esc(item[key] ?? "pending")}/5`).join(" · ");
      $("modelCards").innerHTML = `<div class="model" style="grid-column:1/-1"><h3>Gemini 3.8 Flash (low) · overall=${esc(item.overall_score ?? "pending")}/5</h3><div class="checks">${checks}</div><div class="evidence">${esc(item.reason || "尚未评审")}</div></div>`;
      return;
    }
    const cascadeChecks = `route=${esc(item.cascade_route || "—")}<br>risk=${esc(item.cascade_risk_reasons || "—")}`;
    const qwenChecks = `IF=${esc(item.qwen_instruction_following || "—")} · artifact=${esc(item.qwen_visible_artifact || "—")} · preservation=${esc(item.qwen_content_preservation || "—")}`;
    const geminiChecks = `IF=${esc(item.gemini_instruction_following || "—")} · artifact=${esc(item.gemini_visible_artifact || "—")} · preservation=${esc(item.gemini_content_preservation || "—")}`;
    const cards = [
      modelCard("Cascade", item.cascade_status, cascadeChecks, item.cascade_risk_reasons || "无额外风险说明"),
      modelCard("Qwen 72B", item.qwen_decision, qwenChecks, evidence(item,"qwen")),
      modelCard("Gemini", item.gemini_decision, geminiChecks, evidence(item,"gemini")),
    ];
    if (item.benchmark_human_decision) {
      cards.push(modelCard("冻结人工标签", item.benchmark_human_decision, `split=${esc(item.evaluation_split || "—")}`, item.benchmark_human_notes || "无原人工备注", "gold"));
    }
    $("modelCards").innerHTML = cards.join("");
  }
  function render() {
    const item = items[index], review = currentReview();
    enteredAt = Date.now();
    sourceVideo.pause(); targetVideo.pause();
    $("headerTitle").textContent = survey.title;
    $("sampleTitle").textContent = `${index + 1}/${items.length} · ${item.dataset || "unknown"} / ${item.branch || item.operation || "unknown"}`;
    $("sampleId").textContent = item.sample_id;
    $("rightVideoLabel").textContent = survey.model_evaluation ? "MODEL OUTPUT" : "TARGET";
    $("subtitle").textContent = survey.subtitle;
    $("metaTags").innerHTML = [
      tag("dataset",item.dataset), tag("task",item.branch || item.operation), tag("scope",item.target_scope),
      tag("group",item.review_group), tag("final",item.cascade_status,decisionClass(item.cascade_status)),
      tag("audit_category",item.audit_category), tag("selected_4k",item.selected_4k),
      tag("route",item.cascade_route), tag("provenance",`${item.source_provenance || "?"} → ${item.target_provenance || "?"}`),
      tag("generated",generatedPosition(item)), tag("split",item.evaluation_split),
      tag("checkpoint",item.checkpoint_step), tag("seed",item.generation_seed),
      tag("benchmark_result",item.benchmark_error_type,item.benchmark_error_type === "correct" ? "good" : "bad"),
      item.benchmark_human_decision ? tag("gold",item.benchmark_human_decision,decisionClass(item.benchmark_human_decision)) : ""
    ].join("");
    $("instruction").textContent = item.instruction || item.audit_instruction || "";
    $("sourceRole").textContent = `${sideRole(item,"source")} · provenance=${item.source_provenance || "unresolved"}`;
    $("targetRole").textContent = `${sideRole(item,"target")} · provenance=${item.target_provenance || "unresolved"}`;
    sourceVideo.src = item.review_source_path;
    targetVideo.src = item.review_target_path;
    sourceVideo.load(); targetVideo.load();
    renderModels(item);
    document.querySelectorAll(".status").forEach(button => button.classList.toggle("active", button.dataset.status === review.human_decision));
    $("notes").value = review.human_notes || "";
    $("next").textContent = index === items.length - 1 ? "保存当前条" : "下一条 →";
    updateProgress();
  }
  function setStatus(status) {
    accountTime();
    currentReview().human_decision = status;
    collect(); save(); render();
  }
  function move(delta) {
    collect(); accountTime();
    index = Math.max(0, Math.min(items.length - 1, index + delta));
    save(); render();
  }
  function updateProgress() {
    const completed = items.filter(item => state.reviews[item.sample_id]?.human_decision).length;
    $("progress").max = items.length; $("progress").value = completed;
    $("progressText").textContent = `${completed}/${items.length} 完成 · 当前 ${index + 1}/${items.length}`;
    const seconds = Object.values(state.reviews).reduce((sum, review) => sum + Number(review.seconds || 0), 0);
    $("timer").textContent = new Date(seconds * 1000).toISOString().slice(11,19);
  }
  function download(name, text, type) {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([text], {type})); link.download = name; link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }
  function csvCell(value) {
    const valueText = typeof value === "object" && value !== null ? JSON.stringify(value) : String(value ?? "");
    return /[",\n]/.test(valueText) ? `"${valueText.replaceAll('"','""')}"` : valueText;
  }
  function reviewRows() {
    return items.map(item => {
      const review = state.reviews[item.sample_id] || emptyReview();
      return {...item,human_decision:review.human_decision,overall_status:review.human_decision,human_notes:review.human_notes,review_seconds:review.seconds,review_updated_at:review.updated_at};
    });
  }
  function exportCsv() {
    collect(); save();
    const rows = reviewRows(), fields = [...new Set(rows.flatMap(row => Object.keys(row)))];
    const text = [fields.join(","), ...rows.map(row => fields.map(field => csvCell(row[field])).join(","))].join("\n") + "\n";
    download(`${survey.review_version}_review.csv`, "\ufeff" + text, "text/csv;charset=utf-8");
  }
  function parseCsv(text) {
    const rows=[]; let row=[],cell="",quoted=false;
    for(let i=0;i<text.length;i++){
      const c=text[i];
      if(quoted){ if(c==='"'&&text[i+1]==='"'){cell+='"';i++;} else if(c==='"')quoted=false; else cell+=c; }
      else if(c==='"')quoted=true; else if(c===','){row.push(cell);cell="";} else if(c==='\n'){row.push(cell);rows.push(row);row=[];cell="";} else if(c!=='\r')cell+=c;
    }
    if(cell||row.length){row.push(cell);rows.push(row);}
    const headers=rows.shift()||[];
    return rows.filter(row=>row.length).map(row=>Object.fromEntries(headers.map((header,i)=>[header,row[i]||""])));
  }
  function importRows(rows) {
    const known = new Set(items.map(item => item.sample_id)); let count=0;
    for (const row of rows) {
      const decision = row.human_decision || row.overall_status || row.status || "";
      if (!known.has(row.sample_id) || !decision) continue;
      state.reviews[row.sample_id] = {human_decision:decision,human_notes:row.human_notes || row.notes || "",seconds:Number(row.review_seconds || row.seconds || 0),updated_at:row.review_updated_at || row.updated_at || ""}; count++;
    }
    state.version=survey.review_version; save(); render(); return count;
  }
  function importJson(parsed) {
    if (Array.isArray(parsed?.rows)) return importRows(parsed.rows);
    const reviews = parsed?.state?.reviews || parsed?.reviews;
    if (!reviews || typeof reviews !== "object") throw new Error("JSON 中没有可识别的审核结果");
    return importRows(Object.entries(reviews).map(([sample_id,r]) => ({sample_id,...r})));
  }
  function align(primary, secondary, action) {
    if (syncLock) return;
    syncLock=true;
    try {
      if (Number.isFinite(primary.currentTime) && Math.abs(primary.currentTime-secondary.currentTime)>.08) secondary.currentTime=primary.currentTime;
      if (action==="play") secondary.play().catch(()=>{});
      if (action==="pause") secondary.pause();
      secondary.playbackRate=primary.playbackRate;
    } finally { setTimeout(()=>{syncLock=false;},0); }
  }

  document.querySelectorAll(".status").forEach(button => button.addEventListener("click",()=>setStatus(button.dataset.status)));
  $("notes").addEventListener("input",()=>{collect();save();});
  $("next").addEventListener("click",()=>move(1));
  $("previous").addEventListener("click",()=>move(-1));
  $("jumpBtn").addEventListener("click",()=>{
    const raw=prompt(`跳转到 1-${items.length}`,String(index+1)), nextIndex=Number(raw);
    if(Number.isInteger(nextIndex)&&nextIndex>=1&&nextIndex<=items.length){collect();accountTime();index=nextIndex-1;save();render();}
  });
  $("exportCsv").addEventListener("click",exportCsv);
  $("exportJson").addEventListener("click",()=>{collect();save();download(`${survey.review_version}_backup.json`,JSON.stringify({review_version:survey.review_version,rows:reviewRows(),state},null,2),"application/json");});
  $("importBtn").addEventListener("click",()=>$("importFile").click());
  $("importFile").addEventListener("change",async event=>{
    const file=event.target.files[0]; if(!file)return;
    try {const text=await file.text();const count=file.name.toLowerCase().endsWith(".json")?importJson(JSON.parse(text)):importRows(parseCsv(text));alert(`已按 sample_id 导入 ${count} 条审核。`);}
    catch(error){alert("导入失败："+(error?.message||error));}
    finally{event.target.value="";}
  });
  [sourceVideo,targetVideo].forEach(video=>{
    const other=video===sourceVideo?targetVideo:sourceVideo;
    video.addEventListener("play",()=>align(video,other,"play"));
    video.addEventListener("pause",()=>align(video,other,"pause"));
    video.addEventListener("seeking",()=>align(video,other,"seek"));
    video.addEventListener("ratechange",()=>align(video,other,"rate"));
    video.addEventListener("timeupdate",()=>{if(!syncLock&&!video.paused&&Math.abs(video.currentTime-other.currentTime)>.20)align(video,other,"seek");});
  });
  document.addEventListener("keydown",event=>{
    if(event.target.matches("textarea,input"))return;
    if(event.key==="1")setStatus("accepted");
    else if(event.key==="0")setStatus("rejected");
    else if(event.key==="2")setStatus("uncertain");
    else if(event.key===" "){event.preventDefault();if(sourceVideo.paused&&targetVideo.paused)sourceVideo.play().catch(()=>{});else{sourceVideo.pause();targetVideo.pause();}}
    else if(event.key==="ArrowRight")move(1);
    else if(event.key==="ArrowLeft")move(-1);
  });
  setInterval(updateProgress,1000);
  window.addEventListener("beforeunload",()=>{collect();accountTime();save();});
  render();
})();
</script>
</body>
</html>
"""
    document = document.replace("__PAGE_TITLE__", html.escape(title)).replace("__SURVEY_DATA__", payload)
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cascade",
        type=Path,
        action="append",
        required=True,
        help="cascade_rows.csv; repeat for every production round",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ffmpeg",
        help="ffmpeg executable used to probe and normalize review media to H.264",
    )
    parser.add_argument("--media-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--per-group", type=int, default=20)
    parser.add_argument(
        "--all",
        action="store_true",
        help="include every input row instead of selecting per review group",
    )
    parser.add_argument(
        "--filtered-out",
        action="store_true",
        help="keep only cascade rejected/quarantine rows",
    )
    parser.add_argument(
        "--benchmark-errors",
        action="store_true",
        help="keep only rows where cascade status differs from the frozen human label",
    )
    parser.add_argument(
        "--benchmark-errors-first",
        action="store_true",
        help="include all benchmark rows but put false accepts/rejects before correct rows",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="select this many rows by deterministic maximum-stratum coverage",
    )
    parser.add_argument(
        "--use-rescue-variant",
        action="store_true",
        help="for benchmark rows, display the frozen rescue variant as the final cascade",
    )
    parser.add_argument("--title", default="Stage-2 rescue cascade 最终抽查")
    parser.add_argument(
        "--subtitle",
        default=(
            "包含直接通过、二审通过、Qwen 拒绝后被 Gemini 救回、"
            "最终拒绝和边界样本；审核发生在整个 Stage 2 完成之后。"
        ),
    )
    parser.add_argument("--review-version", default="stage2_rescue_cascade_review_v2")
    args = parser.parse_args()
    rows: list[dict[str, str]] = []
    for path in args.cascade:
        for row in read_csv(path):
            copied = dict(row)
            if copied.get("human_decision"):
                copied["benchmark_human_decision"] = copied.pop("human_decision")
                copied["benchmark_human_notes"] = copied.pop("human_notes", "")
            if args.use_rescue_variant:
                copied["hard_reject_cascade_route"] = copied.get("cascade_route", "")
                copied["hard_reject_cascade_status"] = copied.get("cascade_status", "")
                copied["cascade_route"] = copied.get("rescue_variant_route", "")
                copied["cascade_status"] = copied.get("rescue_variant_status", "")
            copied["cascade_input_file"] = str(path.resolve())
            rows.append(copied)
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("cascade rows must have unique sample IDs")
    if args.filtered_out:
        rows = [row for row in rows if row.get("cascade_status") != "accepted"]
    if args.benchmark_errors or args.benchmark_errors_first:
        errors: list[dict[str, str]] = []
        correct: list[dict[str, str]] = []
        for row in rows:
            gold = row.get("benchmark_human_decision", "")
            predicted = row.get("cascade_status", "")
            if gold == "accepted" and predicted == "rejected":
                row["benchmark_error_type"] = "false_reject"
                errors.append(row)
            elif gold == "rejected" and predicted == "accepted":
                row["benchmark_error_type"] = "false_accept"
                errors.append(row)
            elif gold in {"accepted", "rejected"} and predicted == gold:
                row["benchmark_error_type"] = "correct"
                correct.append(row)
            else:
                raise ValueError(
                    f"cannot compare benchmark row {row.get('sample_id')}: "
                    f"gold={gold!r} cascade={predicted!r}"
                )
        if args.benchmark_errors:
            rows = errors
        else:
            rows = errors + correct
    if not rows:
        raise ValueError("no cascade rows remain after applying review subset filters")
    names = [
        "direct_qwen_accept",
        "gemini_reviewed_accept",
        "qwen_reject_gemini_rescue",
        "gemini_final_reject",
        "boundary_or_quarantine",
    ]
    selected: list[dict[str, str]] = []
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(rows):
            raise ValueError(f"--limit must be in 1..{len(rows)}")
        for row in coverage_pick(rows, args.limit, set()):
            copied = dict(row)
            copied["review_group"] = review_group(row)
            selected.append(copied)
    elif args.all:
        error_order = {"false_accept": 0, "false_reject": 1, "correct": 2}
        ordered = sorted(
            rows,
            key=lambda item: (
                error_order.get(item.get("benchmark_error_type", ""), 3),
                item["sample_id"],
            ),
        )
        for row in ordered:
            copied = dict(row)
            copied["review_group"] = review_group(row)
            selected.append(copied)
    else:
        used: set[str] = set()
        for name in names:
            candidates = [row for row in rows if review_group(row) == name]
            group = coverage_pick(candidates, args.per_group, used)
            used.update(row["sample_id"] for row in group)
            for row in group:
                copied = dict(row)
                copied["review_group"] = name
                selected.append(copied)
    if args.media_workers < 1:
        raise ValueError("--media-workers must be positive")
    ffmpeg = find_ffmpeg(args.ffmpeg)
    args.output.mkdir(parents=True, exist_ok=False)
    copy_media(args.output, selected, ffmpeg, args.media_workers)
    build_html(args.output, selected, args.title, args.subtitle, args.review_version)
    fields = list(dict.fromkeys(key for row in selected for key in row))
    with (args.output / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(selected)
    summary = {
        "review_version": args.review_version,
        "selection_seed": args.seed,
        "selection_policy": (
            "deterministic_maximum_stratum_coverage"
            if args.limit is not None or not args.all
            else "all_rows"
        ),
        "subset": (
            "benchmark_errors"
            if args.benchmark_errors
            else "benchmark_all_errors_first"
            if args.benchmark_errors_first
            else "filtered_out"
            if args.filtered_out
            else "all"
        ),
        "cascade_inputs": [str(path.resolve()) for path in args.cascade],
        "rows": len(selected),
        "groups": dict(Counter(row["review_group"] for row in selected)),
        "datasets": dict(Counter(row.get("dataset", "") for row in selected)),
        "branches": dict(Counter(row.get("branch", "") for row in selected)),
        "cascade_status": dict(Counter(row.get("cascade_status", "") for row in selected)),
        "benchmark_error_type": dict(
            Counter(row.get("benchmark_error_type", "") for row in selected)
        ),
        "review_media": {
            "codec": "h264",
            "pixel_format": "yuv420p",
            "browser_compatible": True,
            "transcoded_files": sum(
                row.get(f"review_{role}_transcode") == BROWSER_TRANSCODE
                for row in selected
                for role in ("source", "target")
            ),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
