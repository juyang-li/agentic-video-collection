#!/usr/bin/env python3
"""Run one model autoregressively over the 12-chain drift benchmark."""

from __future__ import annotations

import argparse
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
DATAMAKE = Path("/home/users/astar/cfar/stuchengyou/jcy/xks/DataMake")
LANCE = DATAMAKE / "10_research/model_experiments/lance"
BASE_MODEL = Path("/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/lance/models/Lance/Lance_3B_Video")
COINVE = ROOT.parent / "gtfresh1150_vs_coinve677_20260927/inference_checkpoints/coinve_real_step677"
GTFRESH = LANCE / "outputs/training_runs/datamake_coinve_mixed_ddp8_lora64_both_no_region_gtfresh_v1/checkpoints/0001150"
BENCHMARK = ROOT / "benchmark.jsonl"
SOURCE_INVENTORY = ROOT / "source_inventory.csv"
CHECKPOINTS = {"lance_base": None, "coinve_real_step677": COINVE, "gtfresh_step1150": GTFRESH}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_records() -> list[dict]:
    import csv
    records = [json.loads(line) for line in BENCHMARK.read_text(encoding="utf-8").splitlines() if line.strip()]
    with SOURCE_INVENTORY.open(encoding="utf-8-sig", newline="") as stream:
        hashes = {row["drift_id"]: row["media_sha256"] for row in csv.DictReader(stream)}
    if len(records) != 12 or sum(len(record["turns"]) for record in records) != 48:
        raise RuntimeError("expected frozen 12-chain / 48-turn benchmark")
    if sum(len(record["checkpoints"]) for record in records) != 24:
        raise RuntimeError("expected 24 drift checkpoints")
    for record in records:
        source = ROOT / record["source"]["media"]
        if not source.is_file() or sha256(source) != hashes.get(record["drift_id"]):
            raise RuntimeError(f"source media mismatch: {source}")
    return records


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


def make_case(record: dict) -> dict:
    source = (ROOT / record["source"]["media"]).resolve()
    conversation, outputs = [], []
    for index, turn in enumerate(record["turns"]):
        content = []
        if index == 0:
            content.append({"type": "video", "value": str(source)})
        content.append({"type": "text", "value": turn["instruction"]})
        conversation.append({"role": "user", "action": "edit_video", "content": content,
                             "meta": {"drift_id": record["drift_id"], "round": turn["round"], "axis": turn["axis"]}})
        outputs.append({"round": turn["round"], "axis": turn["axis"], "instruction": turn["instruction"],
                        "target_state": turn["target_state"], "output": f"{record['drift_id']}/turn{index:02d}.mp4"})
    return {"drift_id": record["drift_id"], "chain_id": record["drift_id"], "pattern": record["pattern"],
            "source": str(source), "source_sha256": record["source"]["official_sha256"],
            "conversation": {"system_prompt": None, "turns": conversation}, "outputs": outputs}


def case_complete(output: Path, case: dict) -> bool:
    directory = output / case["drift_id"]
    status = directory / "case_status.json"
    if not status.is_file() or json.loads(status.read_text()).get("status") != "generated":
        return False
    return (directory / "transcript.json").is_file() and all(
        (directory / f"turn{index:02d}.mp4").is_file() and (directory / f"turn{index:02d}.mp4").stat().st_size > 1000
        for index in range(4))


def install_adapter(model_id: str) -> None:
    if model_id == "coinve_real_step677":
        verify_checkpoint(COINVE, 677)
        from train.datamake_generation_lora import install_lora_adapter_inference_loader
        if not install_lora_adapter_inference_loader(str(COINVE)):
            raise RuntimeError("failed to install CoinVE legacy adapter loader")
    elif model_id == "gtfresh_step1150":
        sys.path.insert(0, str(LANCE / "runtime/peft_0_15_2"))
        from train.datamake_peft import install_inference_loader
        if not install_inference_loader(str(GTFRESH)):
            raise RuntimeError("failed to install gtfresh PEFT adapter loader")


def verify_loaded_adapter(engine, model_id: str) -> dict | None:
    if model_id == "lance_base":
        return None
    import torch
    from safetensors.torch import load_file
    checkpoint = CHECKPOINTS[model_id]
    state = load_file(str(checkpoint / "adapter.safetensors"), device="cpu")
    if model_id == "gtfresh_step1150":
        from train.datamake_peft import canonical_name
        live = {canonical_name(key): value for key, value in engine.model.named_parameters()}
    else:
        live = dict(engine.model.named_parameters())
    missing = [key for key in state if key not in live]
    matching = sum(torch.equal(live[key].detach().cpu(), value.to(live[key].dtype))
                   for key, value in state.items() if key in live)
    nonzero_b = sum(bool(torch.count_nonzero(value)) for key, value in state.items()
                    if key.endswith(("lora_B", "lora_B.weight")))
    expected = {"coinve_real_step677": (504, 252), "gtfresh_step1150": (1016, 503)}[model_id]
    result = {"tensor_count": len(state), "matching_after_cast": matching,
              "nonzero_lora_B": nonzero_b, "missing_live_keys": len(missing)}
    if (len(state), matching, nonzero_b, len(missing)) != (expected[0], expected[0], expected[1], 0):
        raise RuntimeError(f"adapter load verification failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-chains", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit_chains is not None and args.limit_chains < 1:
        parser.error("--limit-chains must be positive")

    contract, preprocessing = common_preprocessing()
    records = load_records()
    if args.limit_chains:
        records = records[:args.limit_chains]
    cases = [make_case(record) for record in records]
    output = args.output_dir.resolve()
    manifest_path = output / "manifest.json"
    checkpoint = CHECKPOINTS[args.model]
    manifest = {
        "schema_version": "real_person_drift_lance_run_v1", "status": "prepared", "model_id": args.model,
        "benchmark": str(BENCHMARK), "benchmark_sha256": sha256(BENCHMARK),
        "expected_chains": len(cases), "expected_videos": len(cases) * 4,
        "inference": {"diffusion_steps": 30, "timestep_shift": 3.5, "cfg_text_scale": 1.0,
                      "cfg_vit_scale": 1.0, "global_seed": 42, "noise_seed": 42,
                      "use_KVcache": True, "autoregressive": True},
        "base_model": str(BASE_MODEL), "base_model_sha256": contract["base_model_sha256"],
        "checkpoint": str(checkpoint) if checkpoint else None,
        "adapter_sha256": sha256(checkpoint / "adapter.safetensors") if checkpoint else None,
        "preprocessing": preprocessing, "cases": cases,
    }
    if args.dry_run:
        print(json.dumps({"status": "cpu_inputs_validated", "model": args.model,
                          "chains": len(cases), "videos": len(cases) * 4}))
        return

    output.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for field in ("model_id", "benchmark_sha256", "expected_chains", "expected_videos"):
            if previous.get(field) != manifest[field]:
                raise RuntimeError(f"resume manifest mismatch: {field}")
        manifest, cases = previous, previous["cases"]
    else:
        manifest.update(status="running", started_at=datetime.now().astimezone().isoformat(),
                        hostname=socket.gethostname(), pid=os.getpid())
        write_json(manifest_path, manifest)
    pending = [case for case in cases if not case_complete(output, case)]
    if not pending:
        manifest.update(status="generated", finished_at=datetime.now().astimezone().isoformat())
        write_json(manifest_path, manifest)
        print(f"[done] already complete {args.model}", flush=True)
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
    install_adapter(args.model)
    from multiturn.engine import MultiTurnChatEngine

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
    manifest["arguments"] = {"model": asdict(model_args), "data": asdict(data_args), "inference": asdict(inference_args)}
    write_json(manifest_path, manifest)
    set_seed(42)
    engine = MultiTurnChatEngine(model_args, data_args, inference_args, device=0, log=print)
    apply_preprocessing(engine.builder, preprocessing)
    manifest["adapter_load_check"] = verify_loaded_adapter(engine, args.model)
    write_json(manifest_path, manifest)
    print(f"[adapter] {manifest['adapter_load_check']}", flush=True)

    for case in pending:
        final_dir = output / case["drift_id"]
        partial = output / f".{case['drift_id']}.partial"
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir()
        write_json(partial / "input.json", case["conversation"])
        print(f"[case] {case['drift_id']} started", flush=True)
        started = time.monotonic()
        set_seed(42); torch.cuda.reset_peak_memory_stats()
        conversation = Conversation.from_dict(case["conversation"])
        engine.run_conversation(conversation, save_dir=str(partial))
        write_json(partial / "transcript.json", conversation.to_dict())
        torch.cuda.synchronize()
        missing = [index for index in range(4) if not (partial / f"turn{index:02d}.mp4").is_file()]
        if missing:
            raise RuntimeError(f"missing generated turns for {case['drift_id']}: {missing}")
        elapsed = time.monotonic() - started
        write_json(partial / "case_status.json", {"status": "generated", "drift_id": case["drift_id"],
                   "rounds": 4, "elapsed_seconds": elapsed,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "finished_at": datetime.now().astimezone().isoformat()})
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(partial, final_dir)
        print(f"[case] {case['drift_id']} finished in {elapsed:.1f}s", flush=True)
    manifest.update(status="generated", finished_at=datetime.now().astimezone().isoformat())
    write_json(manifest_path, manifest)
    print(f"[done] {args.model}", flush=True)


if __name__ == "__main__":
    main()
