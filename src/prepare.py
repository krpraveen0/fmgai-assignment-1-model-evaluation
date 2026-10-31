"""Preparation commands and pretrained baseline diagnostics."""  # MODIFIED

import argparse  # ADDED
import json  # ADDED
import shutil  # ADDED
from pathlib import Path  # ADDED
from core import (  # MODIFIED
    pair_inventory,  # MODIFIED
    validate_pair,  # MODIFIED
    group_splits,  # MODIFIED
    write_manifest,  # MODIFIED
    save_schedule,  # MODIFIED
    fit_stats,  # MODIFIED
    render_examples,  # MODIFIED
)  # MODIFIED
from config import load_config  # MODIFIED

NOISE_POWER = 2.00  # ADDED


def gpu_report(outdir):  # ADDED
    import torch  # ADDED

    cuda = torch.cuda.is_available()  # MODIFIED
    mps = torch.backends.mps.is_available()  # ADDED
    info = {  # MODIFIED
        "gpu_available": cuda,  # MODIFIED
        "mps_available": mps,  # MODIFIED
        "selected_device": "cuda" if cuda else ("mps" if mps else "cpu"),  # MODIFIED
        "pytorch": torch.__version__,  # MODIFIED
        "torch_cuda": torch.version.cuda,  # MODIFIED
        "cuda_bf16_supported": torch.cuda.is_bf16_supported() if cuda else False,  # MODIFIED
        "disk_free_gib": round(shutil.disk_usage(Path.cwd()).free / 2**30, 2),  # MODIFIED
    }  # MODIFIED
    if cuda:  # MODIFIED
        p = torch.cuda.get_device_properties(0)  # MODIFIED
        info.update(gpu_name=p.name, vram_gib=round(p.total_memory / 2**30, 2))  # MODIFIED
    out = Path(outdir)  # MODIFIED
    out.mkdir(parents=True, exist_ok=True)  # MODIFIED
    (out / "gpu_preflight.json").write_text(json.dumps(info, indent=2))  # MODIFIED
    print(json.dumps(info, indent=2))  # MODIFIED
    return info  # ADDED


def training_scheduler(which, config="experiment.json"):  # MODIFIED
    from experiment import scheduler_for  # MODIFIED

    return scheduler_for(which, load_config(config))  # MODIFIED


def validate_dataset(args):  # ADDED
    c = load_config(args.config)  # MODIFIED
    records = pair_inventory(args.data_root)  # ADDED
    if not args.make_splits:  # MODIFIED
        for row in records[: args.check_pairs]:  # MODIFIED
            validate_pair(row)  # MODIFIED
        print("Inspected pairs:", min(args.check_pairs, len(records)))  # MODIFIED
        return  # MODIFIED
    valid, invalid = [], []  # ADDED
    for i, row in enumerate(records):  # MODIFIED
        try:  # ADDED
            validate_pair(row)  # ADDED
            valid.append(row)  # ADDED
        except (ValueError, OSError) as exc:  # MODIFIED
            invalid.append({"id": row["id"], "reason": str(exc)})  # MODIFIED
        if (i + 1) % 1000 == 0:  # MODIFIED
            print("Validated pairs:", i + 1, flush=True)  # MODIFIED
    out = Path(args.outdir)  # MODIFIED
    out.mkdir(parents=True, exist_ok=True)  # MODIFIED
    (out / "excluded_invalid.json").write_text(json.dumps(invalid, indent=2))  # MODIFIED
    splits = group_splits(  # MODIFIED
        valid,  # MODIFIED
        train_n=c["train_patches"],  # MODIFIED
        val_n=c["val_patches"],  # MODIFIED
        test_n=c["test_patches"],  # MODIFIED
        seed=c["seed"],  # MODIFIED
    )  # MODIFIED
    write_manifest(splits, out)  # MODIFIED
    print({split: len(rows) for split, rows in splits.items()})  # MODIFIED


def checkpoint_inspect(outdir, config="experiment.json"):  # MODIFIED
    from huggingface_hub import hf_hub_download  # MODIFIED

    c = load_config(config)  # MODIFIED
    names = [  # MODIFIED
        "scheduler/scheduler_config.json",  # MODIFIED
        "unet/config.json",  # MODIFIED
        "tokenizer/tokenizer_config.json",  # MODIFIED
    ]  # MODIFIED
    loaded = {  # MODIFIED
        name: json.loads(  # MODIFIED
            Path(  # MODIFIED
                hf_hub_download(  # MODIFIED
                    c["model_id"], filename=name, revision=c["model_revision"]  # MODIFIED
                )  # MODIFIED
            ).read_text()  # MODIFIED
        )  # MODIFIED
        for name in names  # MODIFIED
    }  # MODIFIED
    s = loaded[names[0]]  # MODIFIED
    for key in ("beta_schedule", "prediction_type"):  # MODIFIED
        expected = c["beta_schedule"] if key == "beta_schedule" else "epsilon"  # MODIFIED
        if s[key] != expected:  # MODIFIED
            raise ValueError(f"Checkpoint {key} mismatch")  # MODIFIED
    if (  # MODIFIED
        s["beta_start"] != c["beta_start"]  # MODIFIED
        or s["beta_end"] != c["beta_end"]  # MODIFIED
        or s["num_train_timesteps"] != c["noise_steps"]  # MODIFIED
        or loaded[names[1]]["in_channels"] != 8  # MODIFIED
    ):  # MODIFIED
        raise ValueError("Checkpoint conflicts with declared model/schedule")  # MODIFIED
    out = Path(outdir)  # MODIFIED
    out.mkdir(parents=True, exist_ok=True)  # MODIFIED
    (out / "checkpoint_config_audit.json").write_text(  # MODIFIED
        json.dumps(  # MODIFIED
            {"repo": c["model_id"], "revision": c["model_revision"], "loaded": loaded},  # MODIFIED
            indent=2,  # MODIFIED
        )  # MODIFIED
    )  # MODIFIED
    print("Checkpoint configurations verified; weights were not loaded.")  # MODIFIED


def main():  # ADDED
    p = argparse.ArgumentParser(description="Phase 2 data and schedule preparation")  # MODIFIED
    p.add_argument("--config", default="experiment.json")  # MODIFIED
    sub = p.add_subparsers(dest="command", required=True)  # MODIFIED
    for command in ("preflight", "schedule", "checkpoint"):  # MODIFIED
        s = sub.add_parser(command)  # MODIFIED
        s.add_argument("--outdir", default=f"outputs/{command}")  # MODIFIED
    s = sub.add_parser("dataset")  # MODIFIED
    s.add_argument("--data-root", required=True)  # MODIFIED
    s.add_argument("--outdir", default="outputs/manifests")  # MODIFIED
    s.add_argument("--check-pairs", type=int, default=12)  # MODIFIED
    s.add_argument("--make-splits", action="store_true")  # MODIFIED
    s = sub.add_parser("stats")  # MODIFIED
    s.add_argument("--train-manifest", required=True)  # MODIFIED
    s.add_argument("--out", default="outputs/train_stats.json")  # MODIFIED
    s.add_argument("--max-patches", type=int)  # MODIFIED
    s.add_argument("--pixels-per-patch", type=int, default=128)  # MODIFIED
    s = sub.add_parser("render")  # MODIFIED
    s.add_argument("--manifest", required=True)  # MODIFIED
    s.add_argument("--stats", required=True)  # MODIFIED
    s.add_argument("--outdir", required=True)  # MODIFIED
    s.add_argument("--max-patches", type=int, default=10000)  # MODIFIED
    args = p.parse_args()  # MODIFIED
    if args.command == "preflight":  # MODIFIED
        gpu_report(args.outdir)  # MODIFIED
    elif args.command == "schedule":  # MODIFIED
        save_schedule(args.outdir)  # MODIFIED
    elif args.command == "checkpoint":  # MODIFIED
        checkpoint_inspect(args.outdir, args.config)  # MODIFIED
    elif args.command == "dataset":  # MODIFIED
        validate_dataset(args)  # MODIFIED
    elif args.command == "stats":  # MODIFIED
        fit_stats(  # MODIFIED
            args.train_manifest,  # MODIFIED
            args.out,  # MODIFIED
            args.max_patches,  # MODIFIED
            args.pixels_per_patch,  # MODIFIED
            seed=load_config(args.config)["seed"],  # MODIFIED
        )  # MODIFIED
    elif args.command == "render":  # MODIFIED
        render_examples(  # MODIFIED
            args.manifest,  # MODIFIED
            args.stats,  # MODIFIED
            args.outdir,  # MODIFIED
            args.max_patches,  # MODIFIED
            seed=load_config(args.config)["seed"],  # MODIFIED
        )  # MODIFIED


if __name__ == "__main__":  # MODIFIED
    main()  # MODIFIED
