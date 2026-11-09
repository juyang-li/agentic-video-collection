#!/usr/bin/env python3
"""Run one autoregressive real-person benchmark shard with Lance."""

from __future__ import annotations

import argparse
import csv
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

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
DATAMAKE = Path("/home/users/astar/cfar/stuchengyou/jcy/xks/DataMake")
LANCE = DATAMAKE / "10_research/model_experiments/lance"
BASE_MODEL = Path(
    "/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/lance/"
    "models/Lance/Lance_3B_Video"
)
GTFRESH = LANCE / (
    "outputs/training_runs/datamake_coinve_mixed_ddp8_lora64_both_no_region_gtfresh_v1/"
    "checkpoints/0001150"
)
BENCHMARK = ROOT.parent / "real_person_multiturn_benchmark_v2/benchmark.jsonl"
BENCHMARK_ROOT = BENCHMARK.parent
SOURCE_INVENTORY = BENCHMARK_ROOT / "source_inventory.csv"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_records(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with SOURCE_INVENTORY.open(encoding="utf-8-sig", newline="") as stream:
        media_hashes = {row["chain_id"]: row["media_sha256"] for row in csv.DictReader(stream)}
    if len(records) != 30 or sum(len(record["turns"]) for record in records) != 112:
        raise ValueError("expected the frozen 30-chain / 112-turn benchmark")
    if len({record["chain_id"] for record in records}) != len(records):
        raise ValueError("duplicate chain_id")
    for record in records:
        media = BENCHMARK_ROOT / record["source"]["media"]
        if not media.is_file() or sha256(media) != media_hashes.get(record["chain_id"]):
            raise ValueError(f"source media mismatch: {media}")
        if [turn["round"] for turn in record["turns"]] != list(range(1, len(record["turns"]) + 1)):
            raise ValueError(f"non-contiguous rounds: {record['chain_id']}")
    return records


def training_contract() -> dict:
    complete = json.loads((GTFRESH / "checkpoint_complete.json").read_text())
    if complete["status"] != "complete" or complete["completed_steps"] != 1150:
        raise ValueError("gtfresh step1150 is not complete")
    for name in ("adapter.safetensors", "datamake_lora_config.json", "datamake_online_training_contract.json"):
        artifact = GTFRESH / name
        if sha256(artifact) != complete["artifacts"][name]["sha256"]:
            raise ValueError(f"checkpoint artifact mismatch: {name}")
    contract = json.loads((GTFRESH / "datamake_online_training_contract.json").read_text())
    if Path(contract["base_model"]).resolve() != BASE_MODEL.resolve():
        raise ValueError("training contract points to a different base model")
    config = Path(contract["dataset_config"])
    if sha256(config) != contract["dataset_config_sha256"]:
        raise ValueError("training dataset config changed")
    return contract


def load_preprocessing(contract: dict) -> dict:
    config = Path(contract["dataset_config"])
    datasets = yaml.safe_load(config.read_text(encoding="utf-8"))["D1"]
    supported = [key for key in ("datamake_multiturn_v2v_local", "datamake_mixed_v2v_local") if key in datasets]
    if len(supported) != 1:
        raise ValueError("expected one supported training dataset")
    dataset = datasets[supported[0]]
    return {
        "dataset_config": str(config),
        "dataset_config_sha256": sha256(config),
        "video_transform_args": dataset["video_transform_args"],
        "vit_video_transform_args": dataset["vit_video_transform_args"],
        "video_frame_sampler_args": dataset["video_frame_sampler_args"],
    }


def apply_preprocessing(builder, preprocessing: dict) -> None:
    from data.transforms import VideoTransform
    from data.video.sampler.utils import FRAME_SAMPLER_TYPES

    builder.transform = VideoTransform(**preprocessing["video_transform_args"])
    builder.vit_transform = VideoTransform(**preprocessing["vit_video_transform_args"])
    frame_sampler = preprocessing["video_frame_sampler_args"]
    builder.frame_sampler = FRAME_SAMPLER_TYPES[frame_sampler["type"]](**frame_sampler["params"])


def make_case(record: dict, max_rounds: int | None = None) -> dict:
    turns = record["turns"][:max_rounds]
    source = (BENCHMARK_ROOT / record["source"]["media"]).resolve()
    conversation_turns = []
    outputs = []
    for index, turn in enumerate(turns):
        content = []
        if index == 0:
            content.append({"type": "video", "value": str(source)})
        content.append({"type": "text", "value": turn["instruction"]})
        conversation_turns.append({
            "role": "user", "action": "edit_video", "content": content,
            "meta": {"chain_id": record["chain_id"], "round": turn["round"], "axis": turn["axis"]},
        })
        outputs.append({
            "round": turn["round"], "axis": turn["axis"], "instruction": turn["instruction"],
            "target_state": turn["target_state"], "output": f"{record['chain_id']}/turn{index:02d}.mp4",
        })
    return {
        "chain_id": record["chain_id"], "chain_type": record["chain_type"],
        "source": str(source), "source_sha256": record["source"]["official_sha256"],
        "conversation": {"system_prompt": None, "turns": conversation_turns}, "outputs": outputs,
    }


def case_complete(output: Path, case: dict) -> bool:
    case_dir = output / case["chain_id"]
    status = case_dir / "case_status.json"
    if not status.is_file() or json.loads(status.read_text()).get("status") != "generated":
        return False
    return (case_dir / "transcript.json").is_file() and all(
        (case_dir / f"turn{index:02d}.mp4").stat().st_size > 1000 for index in range(len(case["outputs"]))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("lance_base", "gtfresh_step1150"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-chains", type=int)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard index")
    if args.limit_chains is not None and args.limit_chains < 1:
        parser.error("--limit-chains must be positive")
    if args.max_rounds is not None and args.max_rounds < 1:
        parser.error("--max-rounds must be positive")

    contract = training_contract()
    preprocessing = load_preprocessing(contract)
    records = load_records(BENCHMARK)
    records = records[args.shard_index :: args.num_shards]
    if args.limit_chains:
        records = records[:args.limit_chains]
    cases = [make_case(record, args.max_rounds) for record in records]
    output = args.output_dir.resolve()
    manifest_path = output / "manifest.json"
    manifest = {
        "schema_version": "real_person_lance_multiturn_shard_v1", "status": "prepared",
        "model_id": args.model, "shard_index": args.shard_index, "num_shards": args.num_shards,
        "benchmark": str(BENCHMARK), "benchmark_sha256": sha256(BENCHMARK),
        "expected_chains": len(cases), "expected_videos": sum(len(case["outputs"]) for case in cases),
        "inference": {"diffusion_steps": 30, "timestep_shift": 3.5, "cfg_text_scale": 1.0,
                      "cfg_vit_scale": 1.0, "global_seed": 42, "noise_seed": 42,
                      "use_KVcache": True, "autoregressive": True},
        "base_model": str(BASE_MODEL), "base_model_sha256": contract["base_model_sha256"],
        "checkpoint": None if args.model == "lance_base" else str(GTFRESH),
        "adapter_sha256": None if args.model == "lance_base" else sha256(GTFRESH / "adapter.safetensors"),
        "preprocessing": preprocessing, "cases": cases,
    }
    if args.dry_run:
        print(json.dumps({"status": "cpu_inputs_validated", "model": args.model,
                          "chains": len(cases), "videos": manifest["expected_videos"]}))
        return

    output.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for field in ("model_id", "shard_index", "num_shards", "benchmark_sha256", "expected_chains", "expected_videos"):
            if previous.get(field) != manifest[field]:
                raise ValueError(f"resume manifest mismatch: {field}")
        manifest = previous
        cases = manifest["cases"]
    else:
        manifest.update(status="running", started_at=datetime.now().astimezone().isoformat(),
                        hostname=socket.gethostname(), pid=os.getpid())
        write_json(manifest_path, manifest)

    pending = [case for case in cases if not case_complete(output, case)]
    if not pending:
        manifest.update(status="generated", finished_at=datetime.now().astimezone().isoformat())
        write_json(manifest_path, manifest)
        print(f"[done] already complete {output}", flush=True)
        return

    os.chdir(LANCE / "lance")
    sys.path.insert(0, str(LANCE / "lance"))
    import torch
    from transformers import set_seed
    from config.config_factory import DataArguments, InferenceArguments, ModelArguments
    from multiturn.conversation import Conversation

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")
    torch.cuda.set_device(0)
    if args.model == "gtfresh_step1150":
        sys.path.insert(0, str(LANCE / "runtime/peft_0_15_2"))
        from train.datamake_peft import install_inference_loader
        if not install_inference_loader(str(GTFRESH)):
            raise RuntimeError("failed to install gtfresh adapter loader")
    from multiturn.engine import MultiTurnChatEngine

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
    manifest["arguments"] = {"model": asdict(model_args), "data": asdict(data_args),
                             "inference": asdict(inference_args)}
    write_json(manifest_path, manifest)
    set_seed(42)
    engine = MultiTurnChatEngine(model_args, data_args, inference_args, device=0, log=print)
    apply_preprocessing(engine.builder, preprocessing)

    if args.model == "gtfresh_step1150":
        from safetensors.torch import load_file
        from train.datamake_peft import canonical_name
        adapter = load_file(str(GTFRESH / "adapter.safetensors"), device="cpu")
        live = {canonical_name(key): value for key, value in engine.model.named_parameters()}
        matching = sum(torch.equal(live[key].detach().cpu(), value.to(live[key].dtype))
                       for key, value in adapter.items())
        nonzero_b = sum(bool(torch.count_nonzero(value)) for key, value in adapter.items()
                        if key.endswith(("lora_B", "lora_B.weight")))
        manifest["adapter_load_check"] = {
            "tensor_count": len(adapter), "matching_after_bf16_cast": matching,
            "nonzero_lora_B": nonzero_b,
        }
        if matching != len(adapter) or nonzero_b == 0:
            raise RuntimeError("adapter load verification failed")
        del adapter, live
        write_json(manifest_path, manifest)
        print(f"[adapter] {manifest['adapter_load_check']}", flush=True)

    for case in pending:
        chain_id = case["chain_id"]
        final_dir = output / chain_id
        partial = output / f".{chain_id}.partial"
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir()
        input_path = partial / "input.json"
        write_json(input_path, case["conversation"])
        print(f"[case] {chain_id} started rounds={len(case['outputs'])}", flush=True)
        start = time.monotonic()
        set_seed(42)
        torch.cuda.reset_peak_memory_stats()
        conversation = Conversation.from_dict(case["conversation"])
        engine.run_conversation(conversation, save_dir=str(partial))
        write_json(partial / "transcript.json", conversation.to_dict())
        torch.cuda.synchronize()
        missing = [index for index in range(len(case["outputs"]))
                   if not (partial / f"turn{index:02d}.mp4").is_file()]
        if missing:
            raise RuntimeError(f"missing generated turns for {chain_id}: {missing}")
        elapsed = time.monotonic() - start
        write_json(partial / "case_status.json", {
            "status": "generated", "chain_id": chain_id, "rounds": len(case["outputs"]),
            "elapsed_seconds": elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "finished_at": datetime.now().astimezone().isoformat(),
        })
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(partial, final_dir)
        print(f"[case] {chain_id} finished in {elapsed:.1f}s", flush=True)

    manifest.update(status="generated", finished_at=datetime.now().astimezone().isoformat())
    write_json(manifest_path, manifest)
    print(f"[done] {output}", flush=True)


if __name__ == "__main__":
    main()
