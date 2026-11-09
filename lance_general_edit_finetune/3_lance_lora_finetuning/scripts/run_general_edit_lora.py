#!/usr/bin/env python3
"""Validate and launch the frozen 4K general-edit Lance LoRA profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
from typing import Any

from safetensors import safe_open


STAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = STAGE_ROOT / "configs/lance_4k_r64_seed42_e2_v1.json"
DEFAULT_ENV = Path("/scratch/prj0000000275/jcy/.conda/envs/xks_lance")
PROXY_KEYS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def target_audit(model_file: Path, rank: int = 64) -> dict[str, Any]:
    suffixes = (
        "self_attn.q_proj_moe_gen", "self_attn.k_proj_moe_gen",
        "self_attn.v_proj_moe_gen", "self_attn.o_proj_moe_gen",
        "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj",
    )
    targets = []
    with safe_open(model_file, framework="pt", device="cpu") as stream:
        for key in stream.keys():
            if not key.endswith(".weight") or not any(key[:-7].endswith(x) for x in suffixes):
                continue
            shape = list(stream.get_slice(key).get_shape())
            targets.append(
                {"name": key[:-7], "in_features": shape[1], "out_features": shape[0]}
            )
    targets.sort(key=lambda item: item["name"])
    canonical = json.dumps(targets, sort_keys=True, separators=(",", ":")).encode()
    return {
        "modules": len(targets),
        "signature": hashlib.sha256(canonical).hexdigest(),
        "trainable_parameters": sum(
            rank * (item["in_features"] + item["out_features"]) for item in targets
        ),
    }


def validate_profile(path: Path, verify_base_hash: bool) -> dict[str, Any]:
    profile = read_json(path)
    require(profile["schema_version"], "datamake_lance_one_step_x0_lora_profile_v3", "schema")
    require(profile["status"], "training_authorized", "status")
    require(profile["training_authorized"], True, "authorization")
    require(profile["source"]["inherit_any_finetuned_weights"], False, "fresh base")
    require(profile["source"]["senior_checkpoint_used"], False, "senior exclusion")
    runtime = Path(profile["source"]["runtime"]).resolve()
    commit = subprocess.run(
        ["git", "-C", str(runtime), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    require(commit, profile["source"]["lance_commit"], "Lance commit")
    data = profile["data"]
    manifest_path = Path(data["train_chain_manifest"]).resolve()
    manifest = read_json(manifest_path)
    require(manifest["chain_count"], 3601, "train pairs")
    require(manifest["total_optimizer_steps"], 902, "schedule steps")
    require(manifest["seed"], 42, "data seed")
    require(manifest["real_sample_exposures"], 7202, "sample exposures")
    require(sha256_file(Path(data["train_config"])), data["train_config_sha256"], "dataset YAML hash")
    require(sha256_file(Path(data["training_schedule"])), data["training_schedule_sha256"], "schedule hash")
    if data["train_chain_manifest_sha256"] != "TO_BE_FROZEN":
        require(sha256_file(manifest_path), data["train_chain_manifest_sha256"], "manifest hash")
    model_file = Path(profile["model"]["model_path"]) / "model.safetensors"
    if (Path(profile["model"]["model_path"]) / "datamake_lora_config.json").exists():
        raise ValueError("base path contains a LoRA config; official plain base required")
    if verify_base_hash:
        require(sha256_file(model_file), profile["model"]["model_safetensors_sha256"], "base hash")
    audit = target_audit(model_file, profile["lora"]["rank"])
    require(audit["modules"], profile["lora"]["target_modules"], "LoRA modules")
    require(audit["signature"], profile["lora"]["target_signature_sha256"], "LoRA signature")
    require(audit["trainable_parameters"], profile["lora"]["trainable_parameters"], "trainable parameters")
    return profile


def run_name(profile: dict[str, Any], smoke_steps: int, smoke_label: str) -> str:
    name = profile["run"]["run_name"]
    if smoke_steps:
        name += "_smoke" + (f"_{smoke_label}" if smoke_label else "")
    return name


def checkpoint_root(profile: dict[str, Any], smoke_steps: int, smoke_label: str) -> Path:
    return Path(profile["run"]["output_root"]) / run_name(profile, smoke_steps, smoke_label) / "checkpoints"


def complete_checkpoints(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        [
            path for path in root.iterdir()
            if path.is_dir() and path.name.isdigit()
            and (path / "checkpoint_complete.json").is_file()
        ],
        key=lambda path: int(path.name),
    )


def resolve_resume(profile: dict[str, Any], value: str, smoke_steps: int, smoke_label: str) -> Path | None:
    root = checkpoint_root(profile, smoke_steps, smoke_label)
    checkpoints = complete_checkpoints(root)
    if value == "none":
        if checkpoints:
            raise ValueError(f"checkpoints already exist under {root}; resume or use a new label")
        return None
    if value == "latest":
        if not checkpoints:
            raise ValueError(f"no complete checkpoint under {root}")
        return checkpoints[-1]
    result = Path(value).resolve()
    if result.parent != root.resolve() or not (result / "checkpoint_complete.json").is_file():
        raise ValueError(f"invalid checkpoint for this run: {result}")
    return result


def command(
    profile_path: Path,
    profile: dict[str, Any],
    resume: Path | None,
    smoke_steps: int,
    smoke_label: str,
    master_port: int,
) -> list[str]:
    data, run, model = profile["data"], profile["run"], profile["model"]
    lora, rollout = profile["lora"], profile["rollout"]
    optim, loss = profile["optimization"], profile["loss"]
    distributed, checkpoint = profile["distributed"], profile["checkpoint"]
    wandb = profile["wandb"]
    total_steps = smoke_steps or run["total_steps"]
    name = run_name(profile, smoke_steps, smoke_label)
    run_id = wandb["run_id"] + (
        "-smoke" + (f"-{smoke_label}" if smoke_label else "") if smoke_steps else ""
    )
    result = [
        str(DEFAULT_ENV / "bin/torchrun"), "--nnodes", "1", "--nproc-per-node", "4",
        "--node-rank", "0", "--master-addr", "127.0.0.1", "--master-port", str(master_port),
        "train/datamake_online_rollout_lora_train.py",
        "--model_path", model["model_path"], "--vit_path", model["vit_path"],
        "--vit_type", model["vit_type"], "--llm_qk_norm", "True",
        "--llm_qk_norm_und", "True", "--llm_qk_norm_gen", "True",
        "--tie_word_embeddings", "False", "--max_num_frames", "121",
        "--max_latent_size", "64", "--latent_patch_size", "1", "1", "1",
        "--load_from_lance_checkpoint", "True", "--init_from_vlm_checkpoint", "False",
        "--copy_init_moe", "True", "--visual_und", "True", "--visual_gen", "True",
        "--vae_model_type", "wan", "--freeze_llm", "False",
        "--freeze_llm_embed_tokens", "True", "--freeze_vit", "True",
        "--freeze_vit_connector", "False", "--freeze_vae", "True",
        "--freeze_und_params", "True", "--freeze_und", "True",
        "--dataset_config_file", data["train_config"], "--num_workers", "0",
        "--prefetch_factor", "2", "--expected_num_tokens", "65536",
        "--max_num_tokens", "65536", "--max_num_tokens_per_sample", "65536",
        "--prefer_buffer_before", "65536", "--max_buffer_size", "0",
        "--require_und_gen", "True", "--data_seed", "42", "--global_seed", "42",
        "--validation_data_seed", "42", "--validation_noise_seed", "42",
        "--text_cond_dropout_prob", "0", "--vae_cond_dropout_prob", "0",
        "--vit_cond_dropout_prob", "0", "--cfg_type", "0",
        "--warmup_steps", str(optim["warmup_steps"]), "--lr_scheduler", optim["scheduler"],
        "--lr_num_cycles", str(optim["cosine_cycles"]), "--lr", str(optim["learning_rate"]),
        "--min_lr", str(optim["minimum_learning_rate"]), "--beta1", str(optim["beta1"]),
        "--beta2", str(optim["beta2"]), "--eps", str(optim["epsilon"]),
        "--max_grad_norm", str(optim["max_gradient_norm"]), "--ce_weight", "0",
        "--mse_weight", "1", "--edit_region_weighting", "False",
        "--timestep_shift", "4", "--use_flex", "True",
        "--sharding_strategy", distributed["sharding_strategy"],
        "--num_replicate", "1", "--num_shard", "4", "--cpu_offload", "False",
        "--use_ema", "False", "--auto_resume", "False", "--resume_model_only", "False",
        "--load_data_status", "False", "--outputs_dir", run["output_root"],
        "--total_steps", str(total_steps), "--save_every", str(total_steps if smoke_steps else 50),
        "--save_final_checkpoint", "True", "--checkpoint_completed_steps", "True",
        "--keep_last_checkpoints", "4", "--ckpt_debug_steps", "-1", "--log_every", "1",
        "--wandb_project", wandb["project"], "--wandb_name", name,
        "--wandb_runid", run_id, "--wandb_resume", "allow",
        "--wandb_offline", str(bool(smoke_steps)), "--validation_step", "-1",
        "--apply_qwen_2_5_vl_pos_emb", "True", "--apply_chat_template", "False",
        "--datamake_chain_manifest", data["train_chain_manifest"],
        "--datamake_profile_path", str(profile_path),
        "--datamake_profile_sha256", sha256_file(profile_path),
        "--datamake_history_mode", data["history_mode"],
        "--datamake_gradient_accumulation_steps", "2", "--datamake_num_epochs", "2",
        "--datamake_rollout_diffusion_steps", "0", "--datamake_rollout_cfg_text_scale", "1",
        "--datamake_rollout_timestep_shift", "4", "--datamake_rollout_cfg_interval", "0.4", "1.0",
        "--datamake_rollout_cfg_renorm_min", "0", "--datamake_rollout_cfg_renorm_type", "global",
        "--datamake_rollout_use_kv_cache", "False", "--datamake_rollout_retry_count", "1",
        "--datamake_lora_rank", "64", "--datamake_lora_alpha", "64",
        "--datamake_lora_dropout", "0.05", "--datamake_lora_target_scope", "generation_experts",
        "--datamake_lora_expected_target_modules", "252",
        "--datamake_lora_expected_target_signature_sha256", lora["target_signature_sha256"],
        "--datamake_base_model_sha256", model["model_safetensors_sha256"],
        "--datamake_smoke_mode", str(bool(smoke_steps)), "--datamake_smoke_label", smoke_label,
    ]
    if resume is not None:
        result.extend(["--resume_from", str(resume)])
    return result


def environment(profile: dict[str, Any], smoke: bool) -> dict[str, str]:
    value = dict(os.environ)
    # Accept the user's private alias without ever persisting it in the profile,
    # command line, logs, or child environment under the non-standard name.
    wan_db_key = value.pop("WAN_DB", "").strip()
    if wan_db_key:
        value["WANDB_API_KEY"] = wan_db_key
    # Without an environment key, allow W&B's normal `wandb login` credentials.
    for key in PROXY_KEYS:
        value.pop(key, None)
    value.update(
        {
            "HF_HOME": value.get("HF_HOME", "/scratch/prj0000000275/jcy/.cache/huggingface"),
            "NCCL_DEBUG": value.get("NCCL_DEBUG", "WARN"),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "WANDB_MODE": "offline" if smoke else profile["wandb"]["mode"],
            "WANDB_BASE_URL": "https://api.wandb.ai",
            "WANDB_ENTITY": profile["wandb"]["entity"],
            "WANDB_PROJECT": profile["wandb"]["project"],
            "WANDB_RUN_GROUP": profile["wandb"]["group"],
            "WANDB_JOB_TYPE": "smoke" if smoke else profile["wandb"]["job_type"],
            "WANDB_TAGS": ",".join(profile["wandb"]["tags"]),
            "WANDB_DISABLE_CODE": "true", "WANDB_LOG_MODEL": "false",
            "WANDB_CONSOLE": "off", "PYTHONUNBUFFERED": "1",
        }
    )
    runtime = str(Path(profile["source"]["runtime"]).resolve())
    previous_pythonpath = value.get("PYTHONPATH")
    value["PYTHONPATH"] = (
        f"{runtime}:{previous_pythonpath}" if previous_pythonpath else runtime
    )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--resume-from", default="none")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--smoke-label", default="")
    parser.add_argument("--master-port", type=int, default=29642)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-base-hash", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.smoke_steps <= 32:
        raise ValueError("smoke steps must be 0..32")
    if args.smoke_steps and not args.smoke_label:
        raise ValueError("smoke runs require a stable --smoke-label")
    profile_path = args.profile.resolve()
    profile = validate_profile(profile_path, verify_base_hash=not args.skip_base_hash)
    resume = resolve_resume(profile, args.resume_from, args.smoke_steps, args.smoke_label)
    if resume is not None and int(resume.name) >= (args.smoke_steps or profile["run"]["total_steps"]):
        raise ValueError("resume checkpoint already reached the requested target step")
    launch = command(profile_path, profile, resume, args.smoke_steps, args.smoke_label, args.master_port)
    print(shlex.join(launch), flush=True)
    if args.dry_run:
        return
    import torch
    require(torch.cuda.device_count(), 4, "visible CUDA devices")
    run_dir = Path(profile["run"]["output_root"]) / run_name(profile, args.smoke_steps, args.smoke_label)
    segment_dir = run_dir / "segments" / f"job_{os.environ.get('SLURM_JOB_ID', 'manual')}_{os.getpid()}"
    segment_dir.mkdir(parents=True, exist_ok=False)
    (segment_dir / "launch_manifest.json").write_text(
        json.dumps(
            {
                "profile": str(profile_path), "profile_sha256": sha256_file(profile_path),
                "resume_from": str(resume) if resume else None, "command": launch,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "smoke_steps": args.smoke_steps, "smoke_label": args.smoke_label or None,
            },
            indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    child = subprocess.Popen(
        launch,
        cwd=Path(profile["source"]["runtime"]),
        env=environment(profile, bool(args.smoke_steps)),
        start_new_session=True,
    )
    previous_handlers = {}

    def forward_signal(signum: int, _frame: object) -> None:
        if child.poll() is None:
            os.killpg(child.pid, signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, forward_signal)
    try:
        returncode = child.wait()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if returncode:
        raise subprocess.CalledProcessError(returncode, launch)


if __name__ == "__main__":
    main()
