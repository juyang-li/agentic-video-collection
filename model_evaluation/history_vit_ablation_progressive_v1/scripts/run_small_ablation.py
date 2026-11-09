#!/usr/bin/env python3
"""Run normal chains or fixed-history ablations on the frozen 3/2/4-turn benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
MODEL_EVAL = ROOT.parent
SOURCE_RUN = MODEL_EVAL / "gtfresh1150_vs_coinve677_20260927/multiturn/gtfresh_step1150"
OLD_RUNNER = MODEL_EVAL / "history_vit_ablation_v1/scripts/run_ablation.py"
BENCHMARK = ROOT / "benchmark.json"
BENCHMARK_SHA256 = "04e4322dacf75738b2016ec4eca07f8f408b66be3081adcb8025513ac5949d7c"
CHAIN_IDS = ("c_3071d965660823b5", "c_575347cc8db313eb", "c_e98974f1c66a9300")
ROUND_MAPPINGS = {
    "c_3071d965660823b5": [1, 3, 4],
    "c_575347cc8db313eb": [1, 2],
    "c_e98974f1c66a9300": [1, 2, 3, 5],
}
MODELS = ("lance_base", "gtfresh_step1150")
CONDITIONS = (
    "vit_zero_text_keep",
    "vit_gaussian_text_keep",
    "vit_zero_text_zero",
    "vit_gaussian_text_zero",
)


def load_shared():
    spec = importlib.util.spec_from_file_location("history_vit_shared", OLD_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import shared runner: {OLD_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SHARED = load_shared()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_cases() -> list[dict]:
    if sha256(BENCHMARK) != BENCHMARK_SHA256:
        raise RuntimeError(f"frozen benchmark hash mismatch: {BENCHMARK}")
    benchmark = json.loads(BENCHMARK.read_text())
    expected_header = {
        "schema_version": "ordinary_chain_no_camera_benchmark_v1",
        "status": "frozen",
        "chain_count": 3,
        "total_turns": 9,
        "formal_checkpoints": 6,
    }
    for field, expected in expected_header.items():
        if benchmark.get(field) != expected:
            raise RuntimeError(f"unexpected benchmark {field}: {benchmark.get(field)!r}")
    if benchmark.get("derivation", {}).get("round_mappings") != ROUND_MAPPINGS:
        raise RuntimeError("frozen benchmark round mappings changed")
    if benchmark.get("derivation", {}).get("ground_truth_available") is not False:
        raise RuntimeError("derived benchmark must not claim source-run GT")
    source_manifest_path = SOURCE_RUN / "manifest.json"
    if sha256(source_manifest_path) != benchmark["source_manifest_sha256"]:
        raise RuntimeError("source run manifest changed")
    source_manifest = json.loads(source_manifest_path.read_text())
    source_axes = {
        item["name"].split("/")[0]: [output["edit_axis"] for output in item["outputs"]]
        for item in source_manifest["cases"]
        if item["name"].endswith("/autoregressive")
    }

    cases = benchmark["cases"]
    if [case.get("chain_id") for case in cases] != list(CHAIN_IDS):
        raise RuntimeError("unexpected frozen chain order")
    forbidden = ("restore", "return to", "revert", "undo", "back to", "previous state", "original state")
    for case in cases:
        chain_id = case["chain_id"]
        mapping = ROUND_MAPPINGS[chain_id]
        if case.get("kept_source_rounds") != mapping:
            raise RuntimeError(f"round mapping mismatch: {chain_id}")
        input_path = Path(case["source_input"])
        if sha256(input_path) != case["source_input_sha256"]:
            raise RuntimeError(f"source input changed: {input_path}")
        original = json.loads(input_path.read_text())
        removed = sorted(set(range(1, len(original["turns"]) + 1)) - set(mapping))
        if case.get("removed_source_rounds") != removed:
            raise RuntimeError(f"removed-round metadata mismatch: {chain_id}")
        axes = source_axes[chain_id]
        if (any(axes[index - 1] == "camera_id" for index in mapping)
                or any(axes[index - 1] != "camera_id" for index in removed)):
            raise RuntimeError(f"derived chain did not remove exactly the camera rounds: {chain_id}")
        conversation = case["conversation"]
        turns = conversation["turns"]
        if len(turns) != len(mapping) or any(turn.get("role") != "user" for turn in turns):
            raise RuntimeError(f"unexpected derived chain shape: {chain_id}")
        for turn, source_round in zip(turns, mapping):
            source_turn = original["turns"][source_round - 1]
            for field in ("role", "action", "content"):
                if turn.get(field) != source_turn.get(field):
                    raise RuntimeError(f"derived turn differs from source: {chain_id} R{source_round} {field}")
        source = Path(case["source"])
        if not source.is_file() or sha256(source) != case["source_sha256"]:
            raise RuntimeError(f"source video mismatch: {source}")
        if turns[0]["content"][0] != {"type": "video", "value": str(source)}:
            raise RuntimeError(f"first-turn source mismatch: {chain_id}")
        outputs = case["outputs"]
        if len(outputs) != len(turns):
            raise RuntimeError(f"output metadata count mismatch: {chain_id}")
        for index, (turn, output, source_round) in enumerate(zip(turns, outputs, mapping), 1):
            texts = [item["value"] for item in turn["content"] if item["type"] == "text"]
            if len(texts) != 1 or any(word in texts[0].lower() for word in forbidden):
                raise RuntimeError(f"chain contains rollback/history wording: {chain_id} R{index}: {texts}")
            expected = {"round": index, "source_round": source_round,
                        "axis": turn["meta"]["axis"], "instruction": texts[0]}
            if output != expected or output["axis"] == "camera_id":
                raise RuntimeError(f"invalid derived output metadata: {chain_id} R{index}")
            if any(key in output for key in ("gt_target", "gt_target_sha256", "gt_source")):
                raise RuntimeError(f"derived benchmark must not reuse GT: {chain_id} R{index}")
    if sum(len(case["outputs"]) for case in cases) != 9:
        raise RuntimeError("expected 9 total derived turns")
    if sum(max(0, len(case["outputs"]) - 1) for case in cases) != 6:
        raise RuntimeError("expected 6 formal R2+ checkpoints")
    return cases


def normal_complete(root: Path, case: dict) -> bool:
    directory = root / case["chain_id"]
    status = directory / "case_status.json"
    state = json.loads(status.read_text()) if status.is_file() else {}
    return (state.get("status") == "generated"
            and state.get("benchmark_sha256") == BENCHMARK_SHA256
            and all((directory / f"turn{index:02d}.mp4").is_file()
                    and (directory / f"turn{index:02d}.mp4").stat().st_size > 1000
                    for index in range(len(case["outputs"]))))


def normal_video(model: str, chain_id: str, zero_based_round: int) -> Path:
    path = ROOT / "outputs" / model / "normal" / chain_id / f"turn{zero_based_round:02d}.mp4"
    if not path.is_file() or path.stat().st_size <= 1000:
        raise FileNotFoundError(f"normal history is missing: {path}")
    return path.resolve()


def verify_normal_run(model: str, cases: list[dict]) -> None:
    root = ROOT / "outputs" / model / "normal"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"normal manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != "generated"
            or manifest.get("benchmark_sha256") != BENCHMARK_SHA256
            or manifest.get("total_turns") != 9):
        raise RuntimeError(f"normal run does not match frozen benchmark: {manifest_path}")
    for case in cases:
        if not normal_complete(root, case):
            raise RuntimeError(f"normal chain is incomplete: {model} {case['chain_id']}")


def make_fixed_history(case: dict, model: str, target_round: int) -> tuple[dict, list[str]]:
    turns, old_texts = [], []
    for index, user_turn in enumerate(case["conversation"]["turns"][:target_round]):
        turns.append(copy.deepcopy(user_turn))
        if index + 1 < target_round:
            text = next(item["value"] for item in user_turn["content"] if item["type"] == "text")
            old_texts.append(text)
            turns.append({"role": "assistant", "content": [{"type": "video",
                "value": str(normal_video(model, case["chain_id"], index))}],
                "meta": {"normal_history": True, "turn_no": index}})
    return {"system_prompt": case["conversation"].get("system_prompt"), "turns": turns}, old_texts


def build_engine(model: str, preprocessing: dict):
    import torch
    from transformers import set_seed
    sys.path.insert(0, str(SHARED.LANCE / "lance"))
    os.chdir(SHARED.LANCE / "lance")
    from config.config_factory import DataArguments, InferenceArguments, ModelArguments
    SHARED.install_adapter(model)
    from multiturn.engine import MultiTurnChatEngine
    model_args = ModelArguments(
        model_path=str(SHARED.BASE_MODEL), llm_path=str(SHARED.BASE_MODEL),
        vit_path=str(SHARED.LANCE / "models/Lance/Qwen2.5-VL-ViT"),
        vit_type="qwen_2_5_vl_original", llm_qk_norm=True, llm_qk_norm_und=True,
        llm_qk_norm_gen=True, tie_word_embeddings=False, max_num_frames=121, max_latent_size=64,
        latent_patch_size=[1, 1, 1], cfg_text_scale=1.0, cfg_vit_scale=1.0)
    data_args = DataArguments()
    inference_args = InferenceArguments(
        validation_num_timesteps=30, validation_timestep_shift=3.5, copy_init_moe=True,
        visual_und=True, visual_gen=True, vae_model_type="wan", apply_qwen_2_5_vl_pos_emb=True,
        apply_chat_template=False, cfg_type=0, validation_data_seed=42, video_height=480,
        video_width=480, num_frames=50, resolution="video_360p", text_template=True, use_KVcache=True)
    inference_args.global_seed = 42
    inference_args.validation_noise_seed = 42
    set_seed(42)
    engine = MultiTurnChatEngine(model_args, data_args, inference_args, device=0, log=print)
    SHARED.apply_preprocessing(engine.builder, preprocessing)
    adapter_check = SHARED.verify_loaded_adapter(engine, model)
    return engine, model_args, data_args, inference_args, adapter_check


def base_manifest(model: str, cases: list[dict], preprocessing: dict, adapter_check, model_args, data_args,
                  inference_args) -> dict:
    return {"schema_version": "ordinary_chain_no_camera_history_vit_ablation_v1", "status": "running",
            "model_id": model, "source_run": str(SOURCE_RUN), "chain_ids": list(CHAIN_IDS),
            "benchmark": str(BENCHMARK.resolve()), "benchmark_sha256": BENCHMARK_SHA256,
            "chain_count": 3, "turns_per_chain": [3, 2, 4], "total_turns": 9, "formal_checkpoints": 6,
            "ground_truth_available": False,
            "base_model": str(SHARED.BASE_MODEL),
            "checkpoint": str(SHARED.GTFRESH) if model == "gtfresh_step1150" else None,
            "adapter_check": adapter_check, "preprocessing": preprocessing,
            "inference": {"diffusion_steps": 30, "timestep_shift": 3.5, "cfg_text_scale": 1.0,
                          "cfg_vit_scale": 1.0, "global_seed": 42, "noise_seed": 42},
            "arguments": {"model": asdict(model_args), "data": asdict(data_args),
                          "inference": asdict(inference_args)},
            "cases": cases, "hostname": socket.gethostname(), "pid": os.getpid(),
            "started_at": datetime.now().astimezone().isoformat()}


def run_normal(engine, model: str, cases: list[dict], manifest: dict) -> None:
    import torch
    from transformers import set_seed
    from multiturn.conversation import Conversation
    output_root = ROOT / "outputs" / model / "normal"
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for field in ("model_id", "benchmark_sha256"):
            if previous.get(field) != manifest[field]:
                raise RuntimeError(f"normal resume manifest mismatch: {field}")
    write_json(manifest_path, {**manifest, "condition": "normal"})
    for case in cases:
        if normal_complete(output_root, case):
            print(f"[skip] normal {model} {case['chain_id']}", flush=True); continue
        final_dir = output_root / case["chain_id"]
        partial = output_root / f".{case['chain_id']}.partial"
        if partial.exists(): shutil.rmtree(partial)
        partial.mkdir(parents=True)
        write_json(partial / "input.json", case["conversation"])
        conversation = Conversation.from_dict(case["conversation"])
        set_seed(42); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
        engine.run_conversation(conversation, save_dir=str(partial))
        write_json(partial / "transcript.json", conversation.to_dict())
        torch.cuda.synchronize()
        if any(not (partial / f"turn{index:02d}.mp4").is_file() for index in range(len(case["outputs"]))):
            raise RuntimeError(f"normal chain incomplete: {case['chain_id']}")
        write_json(partial / "case_status.json", {"status": "generated", "chain_id": case["chain_id"],
                   "turns": len(case["outputs"]), "elapsed_seconds": time.monotonic() - started,
                   "benchmark_sha256": BENCHMARK_SHA256,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "finished_at": datetime.now().astimezone().isoformat()})
        if final_dir.exists(): shutil.rmtree(final_dir)
        os.replace(partial, final_dir)
        print(f"[done] normal {model} {case['chain_id']}", flush=True)
    completed = {**manifest, "condition": "normal", "status": "generated",
                 "finished_at": datetime.now().astimezone().isoformat()}
    write_json(manifest_path, completed)


def run_ablation(engine, model: str, cases: list[dict], conditions: list[str], manifest: dict) -> None:
    import torch
    from transformers import set_seed
    from data.dataset_base import simple_custom_collate
    from multiturn.conversation import Conversation
    checkpoints = [(case, round_no) for case in cases for round_no in range(2, len(case["outputs"]) + 1)]
    if len(checkpoints) != 6:
        raise RuntimeError("expected 6 formal checkpoints")
    for condition in conditions:
        condition_root = ROOT / "outputs" / model / condition
        manifest_path = condition_root / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            for field in ("model_id", "benchmark_sha256", "condition"):
                expected = condition if field == "condition" else manifest[field]
                if previous.get(field) != expected:
                    raise RuntimeError(f"ablation resume manifest mismatch: {field}")
        write_json(manifest_path, {**manifest, "condition": condition, "checkpoint_count": 6,
                                  "fixed_normal_history": True})
        for number, (case, target_round) in enumerate(checkpoints, 1):
            case_dir = condition_root / case["chain_id"]
            output = case_dir / f"turn{target_round - 1:02d}.mp4"
            receipt = output.with_suffix(".json")
            if output.is_file() and output.stat().st_size > 1000 and receipt.is_file():
                print(f"[skip] {condition} {number}/6", flush=True); continue
            case_dir.mkdir(parents=True, exist_ok=True)
            conversation_dict, old_texts = make_fixed_history(case, model, target_round)
            conversation = Conversation.from_dict(conversation_dict)
            target_idx = len(conversation.turns) - 1
            set_seed(42)
            sample = engine.builder.build(conversation, target_idx)
            val_data = simple_custom_collate([sample]).cuda(0).to_dict()
            seed = SHARED.stable_seed("ordinary_chain_no_camera_history_vit_ablation_v1", model, condition,
                                      case["chain_id"], target_round)
            handles, trace = SHARED.install_intervention_hooks(engine, val_data, condition, old_texts, seed)
            started = time.monotonic()
            try:
                generated = engine._run_visual(val_data, conversation.turns[target_idx].action,
                                               str(case_dir), target_round - 1)
            finally:
                for handle in handles: handle.remove()
            if trace.get("vit_hook_calls") != 1 or trace.get("direct_source_unchanged") is not True:
                raise RuntimeError(f"ViT hook failed: {trace}")
            if condition.endswith("text_zero") and trace.get("text_hook_calls") != 1:
                raise RuntimeError(f"text hook failed: {trace}")
            if Path(generated).resolve() != output.resolve():
                raise RuntimeError(f"unexpected output: {generated}")
            target = case["outputs"][target_round - 1]
            write_json(receipt, {"status": "generated", "model_id": model, "condition": condition,
                       "chain_id": case["chain_id"], "round": target_round,
                       "source_round": target["source_round"], "axis": target["axis"],
                       "instruction": target["instruction"], "ground_truth_available": False,
                       "output": str(output.resolve()), "output_sha256": sha256(output),
                       "elapsed_seconds": time.monotonic() - started, "hook_seed": seed, "trace": trace,
                       "finished_at": datetime.now().astimezone().isoformat()})
            print(f"[done] {condition} {number}/6 {case['chain_id']} R{target_round}", flush=True)
        write_json(manifest_path, {**manifest, "condition": condition, "checkpoint_count": 6,
                                  "fixed_normal_history": True, "status": "generated",
                                  "finished_at": datetime.now().astimezone().isoformat()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--mode", choices=("normal", "ablation"), required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.mode == "ablation" and not args.conditions:
        parser.error("--conditions are required in ablation mode")
    cases = load_cases()
    if args.mode == "ablation":
        verify_normal_run(args.model, cases)
    if args.dry_run:
        print(json.dumps({"status": "validated", "model": args.model, "mode": args.mode,
                          "chains": 3, "turns_per_chain": [3, 2, 4], "turns": 9,
                          "formal_checkpoints": 6, "benchmark_sha256": BENCHMARK_SHA256})); return
    _, preprocessing = SHARED.common_preprocessing()
    engine, model_args, data_args, inference_args, adapter_check = build_engine(args.model, preprocessing)
    manifest = base_manifest(args.model, cases, preprocessing, adapter_check, model_args, data_args, inference_args)
    print(f"[adapter] {adapter_check}", flush=True)
    if args.mode == "normal":
        run_normal(engine, args.model, cases, manifest)
    else:
        run_ablation(engine, args.model, cases, args.conditions, manifest)


if __name__ == "__main__":
    main()

