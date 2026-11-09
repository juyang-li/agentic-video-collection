#!/usr/bin/env python3
"""Run sharded normal chains or fixed-normal-history History-ViT ablations."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark.json"
DATAMAKE = Path("/home/users/astar/cfar/stuchengyou/jcy/xks/DataMake")
LANCE = DATAMAKE / "10_research/model_experiments/lance"
BASE_MODEL = Path(
    "/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/lance/"
    "models/Lance/Lance_3B_Video"
)
GTFRESH = LANCE / (
    "outputs/training_runs/datamake_coinve_mixed_ddp8_lora64_both_no_region_"
    "gtfresh_v1/checkpoints/0001150"
)
MODELS = {"lance_base": None, "gtfresh_step1150": GTFRESH}
CONDITIONS = ("vit_zero_text_keep", "vit_gaussian_text_keep")
EXPECTED_AXES = {"appearance": 23, "action": 20, "outfit": 22}
SCHEMA_VERSION = "ivebench_history_vit_ablation_nonrollback_nocamera_v1"
BENCHMARK_SCHEMA = "ivebench_nonrollback_nocamera_history_vit_v1"
SOURCE_BENCHMARK_SHA256 = "1f9edd319806e546eef367d82facc14cdb2240b872ad41135e980cd2fe62998a"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stable_seed(*values: object) -> int:
    digest = hashlib.sha256("\0".join(map(str, values)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (BENCHMARK.parent / path).resolve()


def load_cases() -> tuple[list[dict], str]:
    if not BENCHMARK.is_file():
        raise FileNotFoundError(f"derived benchmark is missing: {BENCHMARK}")
    benchmark_sha = sha256(BENCHMARK)
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    if benchmark.get("schema_version") != BENCHMARK_SCHEMA or benchmark.get("status") != "frozen":
        raise RuntimeError("benchmark schema/status does not identify the frozen IVEBench derivative")
    if benchmark.get("source_benchmark_sha256") != SOURCE_BENCHMARK_SHA256:
        raise RuntimeError("source IVEBench v2 hash changed")
    if benchmark.get("derivation", {}).get("ground_truth_available") is not False:
        raise RuntimeError("derived benchmark must not claim ground-truth targets")
    cases = benchmark.get("cases")
    if not isinstance(cases, list) or len(cases) != 24:
        raise RuntimeError("expected benchmark.json with exactly 24 cases")

    chain_ids: set[str] = set()
    axis_counts = {key: 0 for key in EXPECTED_AXES}
    total_turns = 0
    checkpoints = 0
    for case in cases:
        chain_id = case.get("chain_id")
        if not isinstance(chain_id, str) or chain_id in chain_ids:
            raise RuntimeError(f"invalid or duplicate chain_id: {chain_id!r}")
        chain_ids.add(chain_id)
        source = resolve_path(case["source"])
        if not source.is_file() or sha256(source) != case.get("source_sha256"):
            raise RuntimeError(f"source video hash mismatch: {chain_id}: {source}")
        conversation = case.get("conversation", {})
        turns = conversation.get("turns")
        outputs = case.get("outputs")
        if not isinstance(turns, list) or not isinstance(outputs, list) or len(turns) != len(outputs):
            raise RuntimeError(f"conversation/output count mismatch: {chain_id}")
        if len(turns) < 2:
            raise RuntimeError(f"chain must contain at least two turns: {chain_id}")
        for index, (turn, output) in enumerate(zip(turns, outputs), 1):
            if turn.get("role") != "user" or turn.get("action") != "edit_video":
                raise RuntimeError(f"unexpected turn role/action: {chain_id} R{index}")
            texts = [item.get("value") for item in turn.get("content", []) if item.get("type") == "text"]
            if texts != [output.get("instruction")]:
                raise RuntimeError(f"instruction mismatch: {chain_id} R{index}")
            if output.get("round") != index or output.get("axis") not in EXPECTED_AXES:
                raise RuntimeError(f"invalid output metadata: {chain_id} R{index}")
            axis_counts[output["axis"]] += 1
            visuals = [item for item in turn.get("content", []) if item.get("type") in ("video", "image")]
            if index == 1:
                if len(visuals) != 1 or resolve_path(visuals[0]["value"]) != source:
                    raise RuntimeError(f"first turn must contain the frozen source: {chain_id}")
            elif visuals:
                raise RuntimeError(f"later user turn unexpectedly contains a visual: {chain_id} R{index}")
            if output.get("axis") == "camera" or output.get("history_reference_round") is not None:
                raise RuntimeError(f"camera/history-reference turn survived derivation: {chain_id} R{index}")
        total_turns += len(outputs)
        checkpoints += len(outputs) - 1

    if total_turns != 65 or checkpoints != 41 or axis_counts != EXPECTED_AXES:
        raise RuntimeError(
            f"unexpected benchmark counts: turns={total_turns}, checkpoints={checkpoints}, "
            f"axes={axis_counts}"
        )
    if benchmark.get("axis_counts") != EXPECTED_AXES:
        raise RuntimeError(f"benchmark header axis_counts={benchmark.get('axis_counts')!r}")
    declared = {
        "chain_count": 24,
        "total_turns": 65,
        "formal_checkpoints": 41,
    }
    for field, expected in declared.items():
        if field in benchmark and benchmark[field] != expected:
            raise RuntimeError(f"benchmark header {field}={benchmark[field]!r}, expected {expected}")
    return cases, benchmark_sha


def partition_cases(cases: list[dict], count: int) -> list[list[dict]]:
    """Stable longest-first greedy partition, balanced by normal turn count."""
    if count < 1:
        raise ValueError("shard count must be positive")
    shards: list[list[dict]] = [[] for _ in range(count)]
    loads = [0] * count
    for case in sorted(cases, key=lambda row: (-len(row["outputs"]), row["chain_id"])):
        target = min(range(count), key=lambda index: (loads[index], index))
        shards[target].append(case)
        loads[target] += len(case["outputs"])
    for shard in shards:
        shard.sort(key=lambda row: row["chain_id"])
    return shards


def verify_checkpoint(path: Path, expected_step: int) -> dict:
    complete = json.loads((path / "checkpoint_complete.json").read_text())
    if complete.get("status") != "complete" or complete.get("completed_steps") != expected_step:
        raise RuntimeError(f"incomplete checkpoint: {path}")
    for name in (
        "adapter.safetensors",
        "datamake_lora_config.json",
        "datamake_online_training_contract.json",
    ):
        if sha256(path / name) != complete["artifacts"][name]["sha256"]:
            raise RuntimeError(f"checkpoint artifact mismatch: {path / name}")
    return json.loads((path / "datamake_online_training_contract.json").read_text())


def common_preprocessing() -> dict:
    contract = verify_checkpoint(GTFRESH, 1150)
    if Path(contract["base_model"]).resolve() != BASE_MODEL.resolve():
        raise RuntimeError("gtfresh contract uses a different base model")
    config = Path(contract["dataset_config"])
    if sha256(config) != contract["dataset_config_sha256"]:
        raise RuntimeError("training dataset config changed")
    datasets = yaml.safe_load(config.read_text(encoding="utf-8"))["D1"]
    key = next(
        name for name in ("datamake_multiturn_v2v_local", "datamake_mixed_v2v_local")
        if name in datasets
    )
    dataset = datasets[key]
    result = {name: dataset[name] for name in (
        "video_transform_args", "vit_video_transform_args", "video_frame_sampler_args"
    )}
    result.update(dataset_config=str(config), dataset_config_sha256=sha256(config))
    return result


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
    matching = sum(
        torch.equal(live[key].detach().cpu(), value.to(live[key].dtype))
        for key, value in state.items() if key in live
    )
    nonzero_b = sum(
        bool(torch.count_nonzero(value))
        for key, value in state.items() if key.endswith(("lora_B", "lora_B.weight"))
    )
    result = {
        "tensor_count": len(state), "matching_after_cast": matching,
        "nonzero_lora_B": nonzero_b, "missing_live_keys": len(missing),
    }
    if (len(state), matching, nonzero_b, len(missing)) != (1016, 1016, 503, 0):
        raise RuntimeError(f"adapter load verification failed: {result}")
    return result


def build_engine(model_id: str, preprocessing: dict):
    import torch
    from transformers import set_seed

    os.chdir(LANCE / "lance")
    sys.path.insert(0, str(LANCE / "lance"))
    from config.config_factory import DataArguments, InferenceArguments, ModelArguments

    install_adapter(model_id)
    from multiturn.engine import MultiTurnChatEngine

    torch.cuda.set_device(0)
    model_args = ModelArguments(
        model_path=str(BASE_MODEL), llm_path=str(BASE_MODEL),
        vit_path=str(LANCE / "models/Lance/Qwen2.5-VL-ViT"),
        vit_type="qwen_2_5_vl_original", llm_qk_norm=True, llm_qk_norm_und=True,
        llm_qk_norm_gen=True, tie_word_embeddings=False, max_num_frames=121,
        max_latent_size=64, latent_patch_size=[1, 1, 1], cfg_text_scale=1.0,
        cfg_vit_scale=1.0,
    )
    data_args = DataArguments()
    inference_args = InferenceArguments(
        validation_num_timesteps=30, validation_timestep_shift=3.5,
        copy_init_moe=True, visual_und=True, visual_gen=True, vae_model_type="wan",
        apply_qwen_2_5_vl_pos_emb=True, apply_chat_template=False, cfg_type=0,
        validation_data_seed=42, video_height=480, video_width=480, num_frames=50,
        resolution="video_360p", text_template=True, use_KVcache=True,
    )
    inference_args.global_seed = 42
    inference_args.validation_noise_seed = 42
    set_seed(42)
    engine = MultiTurnChatEngine(model_args, data_args, inference_args, device=0, log=print)
    apply_preprocessing(engine.builder, preprocessing)
    adapter_check = verify_loaded_adapter(engine, model_id)
    return engine, model_args, data_args, inference_args, adapter_check


def base_manifest(
    model_id: str, benchmark_sha: str, preprocessing: dict, adapter_check: dict | None,
    model_args, data_args, inference_args, output_root: Path, normal_root: Path,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "status": "running", "model_id": model_id,
        "benchmark": str(BENCHMARK.resolve()), "benchmark_sha256": benchmark_sha,
        "chain_count": 24, "total_turns": 65, "formal_checkpoints": 41,
        "fixed_normal_history": True, "history_text_preserved": True,
        "direct_source_vit_preserved": True, "direct_source_vae_preserved": True,
        "base_model": str(BASE_MODEL),
        "checkpoint": str(GTFRESH) if model_id == "gtfresh_step1150" else None,
        "output_root": str(output_root.resolve()),
        "normal_history_root": str(normal_root.resolve()),
        "adapter_check": adapter_check, "preprocessing": preprocessing,
        "inference": {
            "diffusion_steps": 30, "timestep_shift": 3.5,
            "cfg_text_scale": 1.0, "cfg_vit_scale": 1.0,
            "global_seed": 42, "noise_seed": 42, "height": 480, "width": 480,
            "num_frames": 50, "use_KVcache": True,
        },
        "arguments": {
            "model": asdict(model_args), "data": asdict(data_args),
            "inference": asdict(inference_args),
        },
        "hostname": socket.gethostname(), "pid": os.getpid(),
        "started_at": datetime.now().astimezone().isoformat(),
    }


def normal_paths(
    output_root: Path, model_id: str, chain_id: str, turn_index: int,
) -> tuple[Path, Path]:
    output = output_root / model_id / "normal" / chain_id / f"turn{turn_index:02d}.mp4"
    return output, output.with_suffix(".json")


def valid_receipt(
    output: Path, receipt: Path, *, model_id: str, benchmark_sha: str,
    chain_id: str, round_no: int, condition: str,
) -> bool:
    if not output.is_file() or output.stat().st_size <= 1000 or not receipt.is_file():
        return False
    try:
        record = json.loads(receipt.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "status": "generated", "model_id": model_id, "condition": condition,
        "benchmark_sha256": benchmark_sha, "chain_id": chain_id, "round": round_no,
    }
    direct_source = Path(record.get("direct_source", ""))
    direct_source_valid = (
        direct_source.is_file()
        and record.get("direct_source_sha256") == sha256(direct_source)
    )
    return (
        all(record.get(key) == value for key, value in expected.items())
        and record.get("output_sha256") == sha256(output)
        and direct_source_valid
    )


def normal_video(
    normal_root: Path, model_id: str, case: dict, zero_based_turn: int, benchmark_sha: str,
) -> Path:
    output, receipt = normal_paths(normal_root, model_id, case["chain_id"], zero_based_turn)
    if not valid_receipt(
        output, receipt, model_id=model_id, benchmark_sha=benchmark_sha,
        chain_id=case["chain_id"], round_no=zero_based_turn + 1, condition="normal",
    ):
        raise RuntimeError(f"missing or invalid normal output: {output}")
    return output.resolve()


def verify_normal_run(
    normal_root: Path, model_id: str, cases: list[dict], benchmark_sha: str,
) -> None:
    for case in cases:
        for turn_index in range(len(case["outputs"])):
            normal_video(normal_root, model_id, case, turn_index, benchmark_sha)
        case_status = normal_root / model_id / "normal" / case["chain_id"] / "case_status.json"
        if not case_status.is_file():
            raise RuntimeError(f"normal case status missing: {case_status}")
        state = json.loads(case_status.read_text())
        if (
            state.get("status") != "generated"
            or state.get("model_id") != model_id
            or state.get("benchmark_sha256") != benchmark_sha
            or state.get("turn_count") != len(case["outputs"])
        ):
            raise RuntimeError(f"normal case status invalid: {case_status}")
    if len(cases) == 24:
        expected_shards = partition_cases(cases, 2)
        for shard_index, shard_cases in enumerate(expected_shards):
            path = (
                normal_root / model_id / "normal" / "shards"
                / f"shard{shard_index:02d}-of-02.json"
            )
            if not path.is_file():
                raise RuntimeError(f"normal shard manifest missing: {path}")
            record = json.loads(path.read_text())
            if (
                record.get("status") != "generated"
                or record.get("model_id") != model_id
                or record.get("benchmark_sha256") != benchmark_sha
                or record.get("chain_ids") != [case["chain_id"] for case in shard_cases]
            ):
                raise RuntimeError(f"normal shard manifest invalid: {path}")


def run_normal(
    engine, model_id: str, cases: list[dict], benchmark_sha: str, manifest: dict,
    shard_index: int, shard_count: int, output_root: Path,
) -> None:
    import torch
    from transformers import set_seed
    from multiturn.conversation import Conversation, Turn

    run_root = output_root
    output_root = run_root / model_id / "normal"
    shard_cases = partition_cases(cases, shard_count)[shard_index]
    shard_manifest = output_root / "shards" / f"shard{shard_index:02d}-of-{shard_count:02d}.json"
    contract = {
        **manifest, "mode": "normal", "shard_index": shard_index,
        "shard_count": shard_count, "chain_ids": [case["chain_id"] for case in shard_cases],
        "turn_count": sum(len(case["outputs"]) for case in shard_cases),
    }
    if shard_manifest.is_file():
        old = json.loads(shard_manifest.read_text())
        for field in ("model_id", "benchmark_sha256", "shard_index", "shard_count", "chain_ids"):
            if old.get(field) != contract[field]:
                raise RuntimeError(f"normal resume contract mismatch: {field}")
    write_json(shard_manifest, contract)

    for case in shard_cases:
        chain_id = case["chain_id"]
        case_dir = output_root / chain_id
        case_dir.mkdir(parents=True, exist_ok=True)
        built = Conversation(turns=[], system_prompt=case["conversation"].get("system_prompt"))
        for turn_index, (turn_dict, metadata) in enumerate(
            zip(case["conversation"]["turns"], case["outputs"])
        ):
            built.append(Turn.from_dict(copy.deepcopy(turn_dict)))
            output, receipt = normal_paths(run_root, model_id, chain_id, turn_index)
            if valid_receipt(
                output, receipt, model_id=model_id, benchmark_sha=benchmark_sha,
                chain_id=chain_id, round_no=turn_index + 1, condition="normal",
            ):
                built.add_assistant_visual(
                    str(output.resolve()), "video",
                    meta={"turn_no": turn_index, "resumed_normal": True},
                )
                print(f"[skip] normal {model_id} {chain_id} R{turn_index + 1}", flush=True)
                continue

            partial = case_dir / f".turn{turn_index:02d}.partial"
            if partial.exists():
                shutil.rmtree(partial)
            partial.mkdir()
            source = resolve_path(case["source"]) if turn_index == 0 else normal_video(
                run_root, model_id, case, turn_index - 1, benchmark_sha
            )
            set_seed(42)
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            generated = engine.generate_turn(
                built, len(built.turns) - 1, str(partial), turn_index
            )
            generated_path = Path(generated).resolve()
            expected_partial = (partial / f"turn{turn_index:02d}.mp4").resolve()
            if generated_path != expected_partial or generated_path.stat().st_size <= 1000:
                raise RuntimeError(f"unexpected normal output: {generated_path}")
            os.replace(generated_path, output)
            shutil.rmtree(partial)
            built.turns[-1].content[0].value = str(output.resolve())
            torch.cuda.synchronize()
            write_json(receipt, {
                "status": "generated", "schema_version": SCHEMA_VERSION,
                "model_id": model_id, "condition": "normal", "chain_id": chain_id,
                "round": turn_index + 1, "source_round": metadata.get("source_round"),
                "axis": metadata["axis"], "instruction": metadata["instruction"],
                "benchmark_sha256": benchmark_sha, "direct_source": str(source),
                "direct_source_sha256": sha256(source), "output": str(output.resolve()),
                "output_sha256": sha256(output), "elapsed_seconds": time.monotonic() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "finished_at": datetime.now().astimezone().isoformat(),
            })
            write_json(case_dir / "transcript.json", built.to_dict())
            print(f"[done] normal {model_id} {chain_id} R{turn_index + 1}", flush=True)

        write_json(case_dir / "case_status.json", {
            "status": "generated", "model_id": model_id, "chain_id": chain_id,
            "turn_count": len(case["outputs"]), "benchmark_sha256": benchmark_sha,
            "finished_at": datetime.now().astimezone().isoformat(),
        })

    write_json(shard_manifest, {
        **contract, "status": "generated",
        "finished_at": datetime.now().astimezone().isoformat(),
    })


def make_fixed_history(
    case: dict, model_id: str, target_round: int, benchmark_sha: str, normal_root: Path,
) -> tuple[dict, list[str], Path]:
    turns: list[dict] = []
    old_texts: list[str] = []
    for index, user_turn in enumerate(case["conversation"]["turns"][:target_round]):
        turns.append(copy.deepcopy(user_turn))
        if index + 1 < target_round:
            old_texts.append(case["outputs"][index]["instruction"])
            history = normal_video(normal_root, model_id, case, index, benchmark_sha)
            turns.append({
                "role": "assistant", "content": [{"type": "video", "value": str(history)}],
                "meta": {"normal_history": True, "turn_no": index},
            })
    direct_source = normal_video(normal_root, model_id, case, target_round - 2, benchmark_sha)
    return {
        "system_prompt": case["conversation"].get("system_prompt"), "turns": turns,
    }, old_texts, direct_source


def tensor_digest(value: Any) -> dict:
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        raw = tensor.view(torch.uint8).numpy().tobytes()
        return {
            "kind": "tensor", "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if isinstance(value, (list, tuple)):
        return {"kind": "sequence", "items": [tensor_digest(item) for item in value]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return {"kind": "scalar", "value": value}
    return {"kind": type(value).__name__, "repr": repr(value)}


def condition_fingerprints(val_data: dict) -> dict:
    keys = (
        "packed_text_ids", "packed_text_indexes", "padded_videos",
        "vae_video_grid_thw", "packed_vae_token_indexes",
    )
    return {key: tensor_digest(val_data.get(key)) for key in keys}


def install_history_vit_hook(engine, val_data: dict, condition: str, seed: int):
    import torch

    if condition not in CONDITIONS:
        raise ValueError(condition)
    vit_mode = "zero" if condition == "vit_zero_text_keep" else "gaussian"
    grid = val_data["vit_video_grid_thw"].detach().cpu().reshape(-1, 3)
    merge_unit = int(engine.model.vit_model.spatial_merge_unit)
    raw_counts = [int(row.prod().item()) for row in grid]
    if any(count % merge_unit for count in raw_counts):
        raise RuntimeError(f"ViT grid not divisible by merge unit {merge_unit}: {raw_counts}")
    block_counts = [count // merge_unit for count in raw_counts]
    if len(block_counts) < 2:
        raise RuntimeError("R2+ checkpoint contains no historical ViT block")
    history_tokens = sum(block_counts[:-1])
    trace: dict[str, Any] = {
        "condition": condition, "vit_mode": vit_mode, "text_mode": "keep",
        "vit_grid": grid.tolist(), "vit_block_tokens": block_counts,
        "history_vit_tokens": history_tokens,
        "direct_source_vit_tokens": block_counts[-1], "vit_hook_calls": 0,
    }

    def hook(_module, _args, _kwargs, output):
        if trace["vit_hook_calls"]:
            raise RuntimeError("ViT hook called more than once for an isolated checkpoint")
        trace["vit_hook_calls"] += 1
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise RuntimeError(f"unexpected ViT output shape: {getattr(output, 'shape', None)}")
        if output.shape[0] != sum(block_counts):
            raise RuntimeError(
                f"ViT output length {output.shape[0]} != expected {sum(block_counts)}"
            )
        result = output.clone()
        direct_before = output[history_tokens:].clone()
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
                noise = torch.randn(
                    block.shape, generator=generator, device=block.device, dtype=torch.float32
                )
                result[left:left + count] = (noise * std + mean).to(output.dtype)
                stats.append({
                    "tokens": count, "feature_mean_mean": float(mean.mean()),
                    "feature_std_mean": float(std.mean()),
                })
                left += count
            trace["gaussian_block_stats"] = stats
        if not torch.equal(result[history_tokens:], direct_before):
            raise RuntimeError("latest direct-source ViT block changed")
        trace["direct_source_vit_unchanged"] = True
        return result

    handle = engine.model.vit_model.register_forward_hook(hook, with_kwargs=True)
    return handle, trace


def run_ablation(
    engine, model_id: str, condition: str, cases: list[dict], benchmark_sha: str,
    manifest: dict, limit: int | None, output_root: Path, normal_root: Path,
    target_round: int | None,
) -> None:
    import torch
    from transformers import set_seed
    from data.dataset_base import simple_custom_collate
    from multiturn.conversation import Conversation

    checkpoints = [
        (case, round_no) for case in cases
        for round_no in range(2, len(case["outputs"]) + 1)
        if target_round is None or round_no == target_round
    ]
    if target_round is None and len(cases) == 24 and len(checkpoints) != 41:
        raise RuntimeError("expected 41 formal R2+ checkpoints")
    if not checkpoints:
        raise RuntimeError(f"no ablation checkpoint matches target round {target_round}")
    if limit is not None:
        checkpoints = checkpoints[:limit]
    condition_root = output_root / model_id / condition
    manifest_path = condition_root / "manifest.json"
    contract = {
        **manifest, "condition": condition, "checkpoint_count": len(checkpoints),
    }
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text())
        for field in ("model_id", "benchmark_sha256", "condition", "checkpoint_count"):
            if old.get(field) != contract[field]:
                raise RuntimeError(f"ablation resume contract mismatch: {field}")
    write_json(manifest_path, contract)

    for number, (case, target_round) in enumerate(checkpoints, 1):
        chain_id = case["chain_id"]
        case_dir = condition_root / chain_id
        output = case_dir / f"turn{target_round - 1:02d}.mp4"
        receipt = output.with_suffix(".json")
        if valid_receipt(
            output, receipt, model_id=model_id, benchmark_sha=benchmark_sha,
            chain_id=chain_id, round_no=target_round, condition=condition,
        ):
            print(f"[skip] {condition} {number}/{len(checkpoints)} {chain_id} R{target_round}", flush=True)
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        partial = case_dir / f".turn{target_round - 1:02d}.partial"
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir()

        conversation_dict, old_texts, direct_source = make_fixed_history(
            case, model_id, target_round, benchmark_sha, normal_root
        )
        conversation = Conversation.from_dict(conversation_dict)
        target_index = len(conversation.turns) - 1
        set_seed(42)
        sample = engine.builder.build(conversation, target_index)
        val_data = simple_custom_collate([sample]).cuda(0).to_dict()
        before = condition_fingerprints(val_data)
        hook_seed = stable_seed(
            SCHEMA_VERSION, model_id, condition, chain_id, target_round
        )
        handle, trace = install_history_vit_hook(engine, val_data, condition, hook_seed)
        started = time.monotonic()
        try:
            generated = engine._run_visual(
                val_data, conversation.turns[target_index].action,
                str(partial), target_round - 1,
            )
        finally:
            handle.remove()
        after = condition_fingerprints(val_data)
        if before != after:
            raise RuntimeError("text/VAE input tensors changed during History-ViT intervention")
        trace.update(
            history_instruction_count=len(old_texts), history_instructions=old_texts,
            history_text_unchanged=before["packed_text_ids"] == after["packed_text_ids"],
            direct_source_vae_unchanged=(
                before["padded_videos"] == after["padded_videos"]
                and before["vae_video_grid_thw"] == after["vae_video_grid_thw"]
                and before["packed_vae_token_indexes"] == after["packed_vae_token_indexes"]
            ),
            input_fingerprints=before,
        )
        if (
            trace.get("vit_hook_calls") != 1
            or trace.get("direct_source_vit_unchanged") is not True
            or trace.get("history_text_unchanged") is not True
            or trace.get("direct_source_vae_unchanged") is not True
        ):
            raise RuntimeError(f"History-ViT intervention audit failed: {trace}")

        generated_path = Path(generated).resolve()
        expected_partial = (partial / f"turn{target_round - 1:02d}.mp4").resolve()
        if generated_path != expected_partial or generated_path.stat().st_size <= 1000:
            raise RuntimeError(f"unexpected ablation output: {generated_path}")
        os.replace(generated_path, output)
        shutil.rmtree(partial)
        metadata = case["outputs"][target_round - 1]
        write_json(receipt, {
            "status": "generated", "schema_version": SCHEMA_VERSION,
            "model_id": model_id, "condition": condition, "chain_id": chain_id,
            "round": target_round, "source_round": metadata.get("source_round"),
            "axis": metadata["axis"], "instruction": metadata["instruction"],
            "benchmark_sha256": benchmark_sha, "fixed_normal_history": True,
            "direct_source": str(direct_source),
            "direct_source_sha256": sha256(direct_source),
            "output": str(output.resolve()), "output_sha256": sha256(output),
            "elapsed_seconds": time.monotonic() - started, "hook_seed": hook_seed,
            "trace": trace, "finished_at": datetime.now().astimezone().isoformat(),
        })
        print(
            f"[done] {condition} {number}/{len(checkpoints)} {chain_id} R{target_round}",
            flush=True,
        )

    write_json(manifest_path, {
        **contract, "status": "generated",
        "finished_at": datetime.now().astimezone().isoformat(),
    })


def compare_videos(candidate: Path, reference: Path) -> dict:
    """Compare encoded files and decoded RGB bytes; smoke passes only on pixel identity."""
    import cv2
    import numpy as np

    left_capture = cv2.VideoCapture(str(candidate))
    right_capture = cv2.VideoCapture(str(reference))
    if not left_capture.isOpened() or not right_capture.isOpened():
        raise RuntimeError(f"OpenCV could not open control/reference: {candidate}, {reference}")
    left_hash = hashlib.sha256()
    right_hash = hashlib.sha256()
    frames = 0
    decoded_bytes = 0
    absolute_sum = 0
    maximum = 0
    changed = 0
    shape_equal = True
    while True:
        left_ok, left_frame = left_capture.read()
        right_ok, right_frame = right_capture.read()
        if left_ok != right_ok:
            shape_equal = False
            break
        if not left_ok:
            break
        if left_frame.shape != right_frame.shape:
            shape_equal = False
            break
        left_raw = left_frame.tobytes()
        right_raw = right_frame.tobytes()
        left_hash.update(left_raw)
        right_hash.update(right_raw)
        difference = np.abs(left_frame.astype(np.int16) - right_frame.astype(np.int16))
        frames += 1
        decoded_bytes += len(left_raw)
        absolute_sum += int(difference.sum())
        maximum = max(maximum, int(difference.max()))
        changed += int(np.count_nonzero(difference))
    left_capture.release()
    right_capture.release()
    if frames == 0:
        raise RuntimeError(f"OpenCV decoded no frames: {candidate}, {reference}")
    comparison = {
        "candidate": str(candidate.resolve()), "reference": str(reference.resolve()),
        "candidate_file_sha256": sha256(candidate),
        "reference_file_sha256": sha256(reference),
        "encoded_file_identical": sha256(candidate) == sha256(reference),
        "candidate_rgb_sha256": left_hash.hexdigest(),
        "reference_rgb_sha256": right_hash.hexdigest(),
        "decoded_frame_count": frames, "decoded_byte_count": decoded_bytes,
        "decoded_shape_equal": shape_equal,
    }
    if not shape_equal:
        comparison.update(decoded_rgb_identical=False, rgb_mae=None, rgb_max_abs=None,
                          changed_channel_ratio=None)
        return comparison
    comparison.update(
        decoded_rgb_identical=changed == 0,
        rgb_mae=float(absolute_sum / decoded_bytes),
        rgb_max_abs=maximum,
        changed_channel_ratio=float(changed / decoded_bytes),
    )
    return comparison


def run_control(
    engine, model_id: str, case: dict, target_round: int, benchmark_sha: str,
    manifest: dict, output_root: Path, normal_root: Path,
) -> None:
    """Replay one fixed-normal-history checkpoint with no hook and gate on pixel identity."""
    from transformers import set_seed
    from data.dataset_base import simple_custom_collate
    from multiturn.conversation import Conversation

    chain_id = case["chain_id"]
    condition = "normal_replay"
    condition_root = output_root / model_id / condition
    case_dir = condition_root / chain_id
    output = case_dir / f"turn{target_round - 1:02d}.mp4"
    receipt = output.with_suffix(".json")
    if valid_receipt(
        output, receipt, model_id=model_id, benchmark_sha=benchmark_sha,
        chain_id=chain_id, round_no=target_round, condition=condition,
    ):
        record = json.loads(receipt.read_text())
        if record.get("comparison", {}).get("decoded_rgb_identical") is not True:
            raise RuntimeError(f"existing control receipt did not pass pixel identity: {receipt}")
        print(f"[skip] control {model_id} {chain_id} R{target_round}", flush=True)
        return

    case_dir.mkdir(parents=True, exist_ok=True)
    partial = case_dir / f".turn{target_round - 1:02d}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    conversation_dict, old_texts, direct_source = make_fixed_history(
        case, model_id, target_round, benchmark_sha, normal_root
    )
    reference = normal_video(normal_root, model_id, case, target_round - 1, benchmark_sha)
    conversation = Conversation.from_dict(conversation_dict)
    target_index = len(conversation.turns) - 1
    set_seed(42)
    sample = engine.builder.build(conversation, target_index)
    val_data = simple_custom_collate([sample]).cuda(0).to_dict()
    before = condition_fingerprints(val_data)
    started = time.monotonic()
    generated = engine._run_visual(
        val_data, conversation.turns[target_index].action, str(partial), target_round - 1
    )
    after = condition_fingerprints(val_data)
    if before != after:
        raise RuntimeError("control replay mutated packed text/VAE inputs")
    generated_path = Path(generated).resolve()
    expected_partial = (partial / f"turn{target_round - 1:02d}.mp4").resolve()
    if generated_path != expected_partial or generated_path.stat().st_size <= 1000:
        raise RuntimeError(f"unexpected control output: {generated_path}")
    os.replace(generated_path, output)
    shutil.rmtree(partial)
    comparison = compare_videos(output, reference)
    metadata = case["outputs"][target_round - 1]
    passed = comparison["decoded_rgb_identical"] is True
    record = {
        "status": "generated" if passed else "comparison_failed",
        "schema_version": SCHEMA_VERSION, "model_id": model_id,
        "condition": condition, "chain_id": chain_id, "round": target_round,
        "source_round": metadata.get("source_round"), "axis": metadata["axis"],
        "instruction": metadata["instruction"], "benchmark_sha256": benchmark_sha,
        "fixed_normal_history": True, "intervention": "none", "hook_calls": 0,
        "history_instruction_count": len(old_texts), "history_instructions": old_texts,
        "history_text_unchanged": before["packed_text_ids"] == after["packed_text_ids"],
        "direct_source_vae_unchanged": (
            before["padded_videos"] == after["padded_videos"]
            and before["vae_video_grid_thw"] == after["vae_video_grid_thw"]
            and before["packed_vae_token_indexes"] == after["packed_vae_token_indexes"]
        ),
        "direct_source": str(direct_source), "direct_source_sha256": sha256(direct_source),
        "output": str(output.resolve()), "output_sha256": sha256(output),
        "normal_reference": str(reference), "normal_reference_sha256": sha256(reference),
        "comparison": comparison, "input_fingerprints": before,
        "elapsed_seconds": time.monotonic() - started,
        "finished_at": datetime.now().astimezone().isoformat(),
    }
    write_json(receipt, record)
    write_json(
        condition_root / "manifests" / f"{chain_id}_R{target_round:02d}.json",
        {**manifest, "status": record["status"], "condition": condition,
         "chain_id": chain_id, "round": target_round, "comparison": comparison,
         "finished_at": record["finished_at"]},
    )
    if not passed:
        raise RuntimeError(f"normal replay is not pixel-identical to normal: {comparison}")
    print(f"[done] control {model_id} {chain_id} R{target_round} pixel-identical", flush=True)


def verify_ablation_run(
    output_root: Path, model_id: str, condition: str, cases: list[dict], benchmark_sha: str,
    selected_round: int | None = None,
) -> None:
    for case in cases:
        for target_round in range(2, len(case["outputs"]) + 1):
            if selected_round is not None and target_round != selected_round:
                continue
            output = output_root / model_id / condition / case["chain_id"] / f"turn{target_round - 1:02d}.mp4"
            if not valid_receipt(
                output, output.with_suffix(".json"), model_id=model_id,
                benchmark_sha=benchmark_sha, chain_id=case["chain_id"],
                round_no=target_round, condition=condition,
            ):
                raise RuntimeError(f"missing or invalid ablation output: {output}")
            receipt = json.loads(output.with_suffix(".json").read_text())
            trace = receipt.get("trace", {})
            if (
                trace.get("vit_hook_calls") != 1
                or trace.get("direct_source_vit_unchanged") is not True
                or trace.get("history_text_unchanged") is not True
                or trace.get("direct_source_vae_unchanged") is not True
                or trace.get("text_mode") != "keep"
                or trace.get("history_vit_tokens", 0) <= 0
            ):
                raise RuntimeError(f"ablation receipt trace invalid: {output.with_suffix('.json')}")


def verify_control_run(
    output_root: Path, model_id: str, case: dict, target_round: int, benchmark_sha: str,
) -> None:
    output = (
        output_root / model_id / "normal_replay" / case["chain_id"]
        / f"turn{target_round - 1:02d}.mp4"
    )
    receipt = output.with_suffix(".json")
    if not valid_receipt(
        output, receipt, model_id=model_id, benchmark_sha=benchmark_sha,
        chain_id=case["chain_id"], round_no=target_round, condition="normal_replay",
    ):
        raise RuntimeError(f"missing or invalid control replay: {output}")
    record = json.loads(receipt.read_text())
    if record.get("hook_calls") != 0 or record.get("comparison", {}).get("decoded_rgb_identical") is not True:
        raise RuntimeError(f"control replay audit failed: {receipt}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--mode", choices=("normal", "ablation", "control"), required=True)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--normal-root", type=Path, default=ROOT / "outputs",
        help="root containing canonical normal history (independent of --output-root)",
    )
    parser.add_argument("--chain-id")
    parser.add_argument("--target-round", type=int)
    parser.add_argument("--last-round", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.mode == "ablation" and args.condition is None:
        parser.error("--condition is required in ablation mode")
    if args.mode != "ablation" and args.condition is not None:
        parser.error("--condition is only valid in ablation mode")
    if args.mode == "control" and not args.chain_id:
        parser.error("control mode requires --chain-id")
    if args.mode == "control" and not (args.target_round or args.last_round):
        parser.error("control mode requires --target-round or --last-round")
    if args.target_round is not None and args.last_round:
        parser.error("choose only one of --target-round and --last-round")
    if args.target_round is not None and not args.chain_id:
        parser.error("--target-round requires --chain-id")
    if args.last_round and not args.chain_id:
        parser.error("--last-round requires --chain-id")
    if args.mode == "normal" and (args.target_round is not None or args.last_round):
        parser.error("normal mode supports --chain-id but not checkpoint selectors")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.limit is not None and args.mode != "ablation":
        parser.error("--limit is only valid in ablation mode")

    output_root = args.output_root.resolve()
    normal_root = args.normal_root.resolve()
    formal_root = (ROOT / "outputs").resolve()
    subset_ablation = args.mode == "ablation" and (
        args.limit is not None or args.chain_id is not None
        or args.target_round is not None or args.last_round
    )
    if subset_ablation and output_root == formal_root:
        parser.error("subset/limited ablation requires a non-formal --output-root")
    if args.mode == "control" and output_root == formal_root:
        parser.error("control smoke requires a non-formal --output-root")

    cases, benchmark_sha = load_cases()
    if args.chain_id:
        cases = [case for case in cases if case["chain_id"] == args.chain_id]
        if not cases:
            parser.error(f"unknown --chain-id: {args.chain_id}")
    selected_round = args.target_round
    if args.last_round:
        selected_round = len(cases[0]["outputs"])
    if selected_round is not None:
        if selected_round < 2 or selected_round > len(cases[0]["outputs"]):
            parser.error(
                f"selected round must be in [2, {len(cases[0]['outputs'])}] for {cases[0]['chain_id']}"
            )
    common_preprocessing()
    if args.mode == "ablation":
        verify_normal_run(normal_root, args.model, cases, benchmark_sha)
    elif args.mode == "control":
        for turn_index in range(selected_round):
            normal_video(normal_root, args.model, cases[0], turn_index, benchmark_sha)
    if args.verify_only:
        if args.mode == "normal":
            verify_normal_run(output_root, args.model, cases, benchmark_sha)
        elif args.mode == "ablation":
            verify_ablation_run(
                output_root, args.model, args.condition, cases, benchmark_sha, selected_round
            )
        else:
            verify_control_run(output_root, args.model, cases[0], selected_round, benchmark_sha)
        print(json.dumps({"status": "verified", "model": args.model, "mode": args.mode,
                          "condition": args.condition, "benchmark_sha256": benchmark_sha}))
        return
    if args.dry_run:
        shards = partition_cases(cases, args.num_shards)
        print(json.dumps({
            "status": "validated", "model": args.model, "mode": args.mode,
            "condition": args.condition, "benchmark_sha256": benchmark_sha,
            "chains": len(cases),
            "turns": sum(len(case["outputs"]) for case in cases),
            "formal_checkpoints": sum(len(case["outputs"]) - 1 for case in cases),
            "selected_round": selected_round, "output_root": str(output_root),
            "normal_root": str(normal_root),
            "shard_index": args.shard_index, "shard_count": args.num_shards,
            "shard_chain_ids": [case["chain_id"] for case in shards[args.shard_index]],
            "shard_turns": sum(len(case["outputs"]) for case in shards[args.shard_index]),
        }))
        return

    preprocessing = common_preprocessing()
    engine, model_args, data_args, inference_args, adapter_check = build_engine(
        args.model, preprocessing
    )
    manifest = base_manifest(
        args.model, benchmark_sha, preprocessing, adapter_check,
        model_args, data_args, inference_args, output_root, normal_root,
    )
    print(f"[adapter] {adapter_check}", flush=True)
    if args.mode == "normal":
        run_normal(
            engine, args.model, cases, benchmark_sha, manifest,
            args.shard_index, args.num_shards, output_root,
        )
    elif args.mode == "ablation":
        run_ablation(
            engine, args.model, args.condition, cases, benchmark_sha, manifest, args.limit,
            output_root, normal_root, selected_round,
        )
    else:
        run_control(
            engine, args.model, cases[0], selected_round, benchmark_sha,
            manifest, output_root, normal_root,
        )


if __name__ == "__main__":
    main()
