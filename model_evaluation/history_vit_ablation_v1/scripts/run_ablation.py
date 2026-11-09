#!/usr/bin/env python3
"""Run fixed-history ViT/text ablations at the 24 frozen drift checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DRIFT = ROOT.parent / "real_person_drift_stress_v1"
DATAMAKE = Path("/home/users/astar/cfar/stuchengyou/jcy/xks/DataMake")
LANCE = DATAMAKE / "10_research/model_experiments/lance"
BASE_MODEL = Path("/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/lance/models/Lance/Lance_3B_Video")
GTFRESH = LANCE / "outputs/training_runs/datamake_coinve_mixed_ddp8_lora64_both_no_region_gtfresh_v1/checkpoints/0001150"
NORMAL_ROOT = DRIFT / "outputs"
MODELS = {"lance_base": None, "gtfresh_step1150": GTFRESH}
FORMAL_CONDITIONS = (
    "vit_zero_text_keep",
    "vit_gaussian_text_keep",
    "vit_zero_text_zero",
    "vit_gaussian_text_zero",
)
ALL_CONDITIONS = ("normal_replay",) + FORMAL_CONDITIONS


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stable_seed(*values: object) -> int:
    digest = hashlib.sha256("\0".join(map(str, values)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_checkpoint(path: Path, expected_step: int) -> dict:
    complete = json.loads((path / "checkpoint_complete.json").read_text())
    if complete.get("status") != "complete" or complete.get("completed_steps") != expected_step:
        raise RuntimeError(f"incomplete checkpoint: {path}")
    for name in ("adapter.safetensors", "datamake_lora_config.json", "datamake_online_training_contract.json"):
        if sha256(path / name) != complete["artifacts"][name]["sha256"]:
            raise RuntimeError(f"checkpoint artifact mismatch: {path / name}")
    return json.loads((path / "datamake_online_training_contract.json").read_text())


def common_preprocessing() -> tuple[dict, dict]:
    contract = verify_checkpoint(GTFRESH, 1150)
    if Path(contract["base_model"]).resolve() != BASE_MODEL.resolve():
        raise RuntimeError("gtfresh contract uses a different base model")
    config = Path(contract["dataset_config"])
    if sha256(config) != contract["dataset_config_sha256"]:
        raise RuntimeError("training dataset config changed")
    datasets = yaml.safe_load(config.read_text(encoding="utf-8"))["D1"]
    key = next(name for name in ("datamake_multiturn_v2v_local", "datamake_mixed_v2v_local") if name in datasets)
    dataset = datasets[key]
    preprocessing = {name: dataset[name] for name in (
        "video_transform_args", "vit_video_transform_args", "video_frame_sampler_args")}
    preprocessing.update(dataset_config=str(config), dataset_config_sha256=sha256(config))
    return contract, preprocessing


def apply_preprocessing(builder, preprocessing: dict) -> None:
    from data.transforms import VideoTransform
    from data.video.sampler.utils import FRAME_SAMPLER_TYPES
    builder.transform = VideoTransform(**preprocessing["video_transform_args"])
    builder.vit_transform = VideoTransform(**preprocessing["vit_video_transform_args"])
    sampler = preprocessing["video_frame_sampler_args"]
    builder.frame_sampler = FRAME_SAMPLER_TYPES[sampler["type"]](**sampler["params"])


def install_adapter(model_id: str) -> None:
    if model_id == "gtfresh_step1150":
        sys.path.insert(0, str(LANCE / "runtime/peft_0_15_2"))
        from train.datamake_peft import install_inference_loader
        if not install_inference_loader(str(GTFRESH)):
            raise RuntimeError("failed to install gtfresh PEFT adapter loader")


def verify_loaded_adapter(engine, model_id: str) -> dict | None:
    if model_id == "lance_base":
        return None
    import torch
    from safetensors.torch import load_file
    from train.datamake_peft import canonical_name
    state = load_file(str(GTFRESH / "adapter.safetensors"), device="cpu")
    live = {canonical_name(key): value for key, value in engine.model.named_parameters()}
    missing = [key for key in state if key not in live]
    matching = sum(torch.equal(live[key].detach().cpu(), value.to(live[key].dtype))
                   for key, value in state.items() if key in live)
    nonzero_b = sum(bool(torch.count_nonzero(value)) for key, value in state.items()
                    if key.endswith(("lora_B", "lora_B.weight")))
    result = {"tensor_count": len(state), "matching_after_cast": matching,
              "nonzero_lora_B": nonzero_b, "missing_live_keys": len(missing)}
    if (len(state), matching, nonzero_b, len(missing)) != (1016, 1016, 503, 0):
        raise RuntimeError(f"adapter load verification failed: {result}")
    return result


def load_benchmark() -> tuple[dict[str, dict], list[dict]]:
    records = {
        row["drift_id"]: row
        for row in (json.loads(line) for line in (DRIFT / "benchmark.jsonl").read_text().splitlines() if line.strip())
    }
    import csv
    with (DRIFT / "checkpoints.csv").open(encoding="utf-8-sig", newline="") as stream:
        checkpoints = list(csv.DictReader(stream))
    if len(records) != 12 or len(checkpoints) != 24:
        raise RuntimeError("expected frozen 12-chain / 24-checkpoint benchmark")
    return records, checkpoints


def normal_video(model_id: str, drift_id: str, zero_based_turn: int) -> Path:
    path = NORMAL_ROOT / model_id / drift_id / f"turn{zero_based_turn:02d}.mp4"
    if not path.is_file() or path.stat().st_size <= 1000:
        raise FileNotFoundError(f"missing normal-history video: {path}")
    return path.resolve()


def make_fixed_history(record: dict, model_id: str, target_round: int) -> tuple[dict, list[str]]:
    """Build user/current turn plus canonical normal assistant history (round is 1-based)."""
    source = (DRIFT / record["source"]["media"]).resolve()
    turns: list[dict[str, Any]] = []
    old_texts: list[str] = []
    for index, turn in enumerate(record["turns"][:target_round]):
        content: list[dict[str, str]] = []
        if index == 0:
            content.append({"type": "video", "value": str(source)})
        content.append({"type": "text", "value": turn["instruction"]})
        turns.append({"role": "user", "action": "edit_video", "content": content,
                      "meta": {"drift_id": record["drift_id"], "round": index + 1, "axis": turn["axis"]}})
        if index + 1 < target_round:
            old_texts.append(turn["instruction"])
            turns.append({"role": "assistant", "content": [{"type": "video", "value": str(normal_video(
                model_id, record["drift_id"], index))}], "meta": {"normal_history": True, "turn_no": index}})
    return {"system_prompt": None, "turns": turns}, old_texts


def find_exact(haystack: list[int], needle: list[int]) -> list[int]:
    if not needle:
        return []
    matches = [i for i in range(len(haystack) - len(needle) + 1) if haystack[i:i + len(needle)] == needle]
    if len(matches) != 1:
        raise RuntimeError(f"historical instruction token match count is {len(matches)}, expected 1")
    return list(range(matches[0], matches[0] + len(needle)))


def intervention_flags(condition: str) -> tuple[str | None, bool]:
    if condition == "normal_replay":
        return None, False
    vit_mode = "gaussian" if "gaussian" in condition else "zero"
    return vit_mode, condition.endswith("text_zero")


def install_intervention_hooks(engine, val_data: dict, condition: str, old_texts: list[str], seed: int):
    """Install one-shot hooks and return (handles, mutable trace)."""
    import torch
    vit_mode, text_zero = intervention_flags(condition)
    trace: dict[str, Any] = {"condition": condition, "vit_mode": vit_mode, "text_zero": text_zero,
                            "history_instruction_count": len(old_texts)}
    handles = []
    if vit_mode is not None:
        grid = val_data["vit_video_grid_thw"].detach().cpu().reshape(-1, 3)
        merge_unit = int(engine.model.vit_model.spatial_merge_unit)
        raw_counts = [int(row.prod().item()) for row in grid]
        if any(count % merge_unit for count in raw_counts):
            raise RuntimeError(f"ViT grid is not divisible by merge factor {merge_unit}: {raw_counts}")
        block_counts = [count // merge_unit for count in raw_counts]
        if len(block_counts) < 2:
            raise RuntimeError("checkpoint has no historical ViT block to ablate")
        history_tokens = sum(block_counts[:-1])
        trace.update(vit_grid=grid.tolist(), vit_block_tokens=block_counts,
                     history_vit_tokens=history_tokens, direct_source_vit_tokens=block_counts[-1],
                     vit_hook_calls=0)

        def vit_hook(_module, _args, _kwargs, output):
            if trace["vit_hook_calls"]:
                raise RuntimeError("ViT hook called more than once for one isolated checkpoint")
            trace["vit_hook_calls"] += 1
            if not isinstance(output, torch.Tensor) or output.ndim != 2:
                raise RuntimeError(f"unexpected ViT output: {type(output)} shape={getattr(output, 'shape', None)}")
            if output.shape[0] != sum(block_counts):
                raise RuntimeError(f"ViT output length {output.shape[0]} != expected {sum(block_counts)}")
            result = output.clone()
            direct_source_before = output[history_tokens:].clone()
            if vit_mode == "zero":
                result[:history_tokens].zero_()
            else:
                generator = torch.Generator(device=output.device).manual_seed(seed)
                left = 0
                stats = []
                for count in block_counts[:-1]:
                    block = output[left:left + count]
                    mean = block.float().mean(dim=0, keepdim=True)
                    std = block.float().std(dim=0, unbiased=False, keepdim=True)
                    noise = torch.randn(block.shape, generator=generator, device=block.device, dtype=torch.float32)
                    result[left:left + count] = (noise * std + mean).to(output.dtype)
                    stats.append({"tokens": count, "feature_mean_mean": float(mean.mean()),
                                  "feature_std_mean": float(std.mean())})
                    left += count
                trace["gaussian_block_stats"] = stats
            if not torch.equal(result[history_tokens:], direct_source_before):
                raise RuntimeError("latest direct-source ViT block changed during history ablation")
            trace["direct_source_unchanged"] = True
            return result

        handles.append(engine.model.vit_model.register_forward_hook(vit_hook, with_kwargs=True))

    if text_zero:
        ids_tensor = val_data["packed_text_ids"]
        ids = ids_tensor.detach().cpu().reshape(-1).tolist()
        indexes: list[int] = []
        for old_text in old_texts:
            needle = engine.tokenizer(old_text, add_special_tokens=False)["input_ids"]
            indexes.extend(find_exact(ids, needle))
        if not indexes:
            raise RuntimeError("text-zero condition found no historical instruction tokens")
        mask = torch.zeros(len(ids), dtype=torch.bool)
        mask[sorted(set(indexes))] = True
        trace.update(history_text_tokens=int(mask.sum()), text_hook_calls=0)

        def text_hook(_module, args, output):
            input_ids = args[0]
            if input_ids.numel() != mask.numel() or trace["text_hook_calls"]:
                return output
            trace["text_hook_calls"] += 1
            result = output.clone()
            local_mask = mask.to(output.device)
            if output.ndim == 2:
                result[local_mask] = 0
            elif output.ndim == 3 and output.shape[0] == 1:
                result[0, local_mask] = 0
            else:
                raise RuntimeError(f"unexpected text embedding shape: {tuple(output.shape)}")
            return result

        handles.append(engine.model.language_model.model.embed_tokens.register_forward_hook(text_hook))
    return handles, trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--conditions", nargs="+", choices=ALL_CONDITIONS, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--limit-checkpoints", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit_checkpoints is not None and args.limit_checkpoints < 1:
        parser.error("--limit-checkpoints must be positive")

    contract, preprocessing = common_preprocessing()
    records, checkpoints = load_benchmark()
    if args.limit_checkpoints:
        checkpoints = checkpoints[:args.limit_checkpoints]
    for checkpoint in checkpoints:
        normal_video(args.model, checkpoint["drift_id"], int(checkpoint["round"]) - 2)
        make_fixed_history(records[checkpoint["drift_id"]], args.model, int(checkpoint["round"]))
    if args.dry_run:
        print(json.dumps({"status": "cpu_inputs_validated", "model": args.model,
                          "conditions": args.conditions, "checkpoints": len(checkpoints)}))
        return

    os.chdir(LANCE / "lance")
    sys.path.insert(0, str(LANCE / "lance"))
    import torch
    from transformers import set_seed
    from config.config_factory import DataArguments, InferenceArguments, ModelArguments
    from data.dataset_base import simple_custom_collate
    from multiturn.conversation import Conversation
    install_adapter(args.model)
    from multiturn.engine import MultiTurnChatEngine

    torch.cuda.set_device(0)
    model_args = ModelArguments(
        model_path=str(BASE_MODEL), llm_path=str(BASE_MODEL), vit_path=str(LANCE / "models/Lance/Qwen2.5-VL-ViT"),
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
    apply_preprocessing(engine.builder, preprocessing)
    adapter_check = verify_loaded_adapter(engine, args.model)
    print(f"[adapter] {adapter_check}", flush=True)

    for condition in args.conditions:
        condition_root = args.output_root.resolve() / args.model / condition
        manifest_path = condition_root / "manifest.json"
        manifest = {
            "schema_version": "history_vit_ablation_v1", "status": "running",
            "model_id": args.model, "condition": condition, "checkpoint_count": len(checkpoints),
            "benchmark": str((DRIFT / "benchmark.jsonl").resolve()),
            "benchmark_sha256": sha256(DRIFT / "benchmark.jsonl"),
            "normal_history_root": str((NORMAL_ROOT / args.model).resolve()),
            "base_model": str(BASE_MODEL), "checkpoint": str(MODELS[args.model]) if MODELS[args.model] else None,
            "adapter_check": adapter_check, "inference": {"diffusion_steps": 30, "timestep_shift": 3.5,
            "cfg_text_scale": 1.0, "cfg_vit_scale": 1.0, "noise_seed": 42, "fixed_normal_history": True},
            "arguments": {"model": asdict(model_args), "data": asdict(data_args), "inference": asdict(inference_args)},
            "hostname": socket.gethostname(), "pid": os.getpid(),
            "started_at": datetime.now().astimezone().isoformat(),
        }
        if manifest_path.exists():
            prior = json.loads(manifest_path.read_text())
            for field in ("model_id", "condition", "benchmark_sha256"):
                if prior.get(field) != manifest[field]:
                    raise RuntimeError(f"resume manifest mismatch: {field}")
        write_json(manifest_path, manifest)

        for number, checkpoint in enumerate(checkpoints, 1):
            drift_id, target_round = checkpoint["drift_id"], int(checkpoint["round"])
            case_dir = condition_root / drift_id
            output = case_dir / f"turn{target_round - 1:02d}.mp4"
            receipt = case_dir / f"turn{target_round - 1:02d}.json"
            if output.is_file() and output.stat().st_size > 1000 and receipt.is_file():
                print(f"[skip] {condition} {drift_id} round={target_round}", flush=True)
                continue
            case_dir.mkdir(parents=True, exist_ok=True)
            conversation_dict, old_texts = make_fixed_history(records[drift_id], args.model, target_round)
            conversation = Conversation.from_dict(conversation_dict)
            target_idx = len(conversation.turns) - 1
            set_seed(42)
            sample = engine.builder.build(conversation, target_idx)
            val_data = simple_custom_collate([sample]).cuda(0).to_dict()
            hook_seed = stable_seed("history_vit_ablation_v1", args.model, condition, drift_id, target_round)
            handles, trace = install_intervention_hooks(engine, val_data, condition, old_texts, hook_seed)
            started = time.monotonic()
            try:
                generated = engine._run_visual(val_data, conversation.turns[target_idx].action,
                                               str(case_dir), target_round - 1)
            finally:
                for handle in handles:
                    handle.remove()
            if trace.get("vit_mode") is not None and trace.get("vit_hook_calls") != 1:
                raise RuntimeError(f"ViT hook did not fire exactly once: {trace}")
            if trace.get("text_zero") and trace.get("text_hook_calls") != 1:
                raise RuntimeError(f"text hook did not fire exactly once: {trace}")
            if Path(generated).resolve() != output.resolve() or not output.is_file():
                raise RuntimeError(f"unexpected output path: {generated}")
            elapsed = time.monotonic() - started
            write_json(receipt, {"status": "generated", "model_id": args.model, "condition": condition,
                "drift_id": drift_id, "round": target_round, "instruction": record_instruction(records[drift_id], target_round),
                "output": str(output.resolve()), "output_sha256": sha256(output), "elapsed_seconds": elapsed,
                "hook_seed": hook_seed, "trace": trace, "finished_at": datetime.now().astimezone().isoformat()})
            print(f"[done] {condition} {number}/{len(checkpoints)} {drift_id} round={target_round} {elapsed:.1f}s", flush=True)

        manifest.update(status="generated", finished_at=datetime.now().astimezone().isoformat())
        write_json(manifest_path, manifest)


def record_instruction(record: dict, target_round: int) -> str:
    return record["turns"][target_round - 1]["instruction"]


if __name__ == "__main__":
    main()

