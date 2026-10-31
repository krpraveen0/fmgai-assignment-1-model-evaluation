"""Experiment settings and split provenance."""  # ADDED

import json  # ADDED
from pathlib import Path  # ADDED
from core import (  # ADDED
    MODEL_ID,  # ADDED
    MODEL_REVISION,  # ADDED
    NOISE_POWER,  # ADDED
    SEED,  # ADDED
    checked_manifest,  # ADDED
    file_hash,  # ADDED
    load_jsonl,  # ADDED
)  # ADDED


def load_config(path="experiment.json"):  # ADDED
    c = json.loads(Path(path).read_text())  # ADDED
    if (  # ADDED
        c["model_id"] != MODEL_ID  # ADDED
        or c["noise_power"] != NOISE_POWER  # ADDED
        or c["native_theta"] != "alpha_bar"  # ADDED
    ):  # ADDED
        raise ValueError("Configuration conflicts with the completed Phase 1 design")  # ADDED
    if (  # ADDED
        c["noise_steps"] != 1000  # ADDED
        or c["beta_schedule"] != "scaled_linear"  # ADDED
        or c["beta_start"] != 0.00085  # ADDED
        or c["beta_end"] != 0.012  # ADDED
    ):  # ADDED
        raise ValueError("Unexpected native schedule")  # ADDED
    if c["train_epochs"] < 5 or len(c["checkpoints_epochs"]) != 5:  # ADDED
        raise ValueError("Five equally spaced checkpoint epochs are required")  # ADDED
    epochs = c["checkpoints_epochs"]  # ADDED
    if (  # ADDED
        epochs != sorted(set(epochs))  # ADDED
        or epochs[-1] != c["train_epochs"]  # ADDED
        or len(set(b - a for a, b in zip([0] + epochs, epochs))) != 1  # ADDED
    ):  # ADDED
        raise ValueError(  # ADDED
            "Checkpoint epochs must be equally spaced through the final epoch"  # ADDED
        )  # ADDED
    if c["micro_batch_size"] < 1 or c["effective_batch_size"] % c["micro_batch_size"]:  # ADDED
        raise ValueError("Effective batch must be divisible by micro batch")  # ADDED
    if c["resolution"] != [256, 256] or c["validation_frequency"] != "once per epoch":  # ADDED
        raise ValueError("Unsupported resolution or validation frequency")  # ADDED
    if (  # ADDED
        c["precision"] not in ("no", "fp16", "bf16")  # ADDED
        or c["inference_precision"] != "float32"  # ADDED
    ):  # ADDED
        raise ValueError(  # ADDED
            "Unsupported precision; final evaluation uses float32 for every run"  # ADDED
        )  # ADDED
    if (  # ADDED
        c["optimizer"] != "AdamW"  # ADDED
        or c["trajectory_steps"] != 5  # ADDED
        or c["evidence_samples"] < 3  # ADDED
    ):  # ADDED
        raise ValueError("Unsupported optimizer/evidence configuration")  # ADDED
    if c["inference_steps"] < 6 or c["validation_examples"] < 1:  # ADDED
        raise ValueError("Invalid validation/inference settings")  # ADDED
    if (  # ADDED
        c["inference_sampler"] != "EulerAncestralDiscreteScheduler"  # ADDED
        or c["validation_metric"] != "native_schedule_epsilon_mse"  # ADDED
    ):  # ADDED
        raise ValueError("Unsupported inference or validation policy")  # ADDED
    if (  # ADDED
        c["trainable"] != ["UNet"]  # ADDED
        or c["frozen"] != ["VAE", "text encoder"]  # ADDED
        or not c["no_augmentation"]  # ADDED
    ):  # ADDED
        raise ValueError("Unsupported trainable components or augmentation")  # ADDED
    if (  # ADDED
        c["metrics"] != ["LPIPS", "PSNR", "SSIM"]  # ADDED
        or not 0 <= c["conditioning_dropout"] <= 1 / 3  # ADDED
    ):  # ADDED
        raise ValueError("Unsupported metrics or dropout")  # ADDED
    if (  # ADDED
        any(  # ADDED
            c[k] < 1  # ADDED
            for k in (  # ADDED
                "train_patches",  # ADDED
                "val_patches",  # ADDED
                "test_patches",  # ADDED
                "effective_batch_size",  # ADDED
            )  # ADDED
        )  # ADDED
        or c["learning_rate"] <= 0  # ADDED
        or c["gradient_clip"] <= 0  # ADDED
        or c["weight_decay"] < 0  # ADDED
    ):  # ADDED
        raise ValueError("Invalid split sizes or optimizer settings")  # ADDED
    return c  # ADDED


def check_examples(path, split, config, official=True):  # ADDED
    rows = load_jsonl(path)  # ADDED
    if not rows or any(r.get("split") != split for r in rows):  # ADDED
        raise ValueError(f"Expected {split} examples")  # ADDED
    keys = [(r["id"], r["product"]) for r in rows]  # ADDED
    if len(keys) != len(set(keys)):  # ADDED
        raise ValueError("Duplicate example/product")  # ADDED
    if (  # ADDED
        len({r["stats_sha256"] for r in rows}) != 1  # ADDED
        or len({r["manifest_sha256"] for r in rows}) != 1  # ADDED
    ):  # ADDED
        raise ValueError("Mixed preprocessing or source manifests")  # ADDED
    if official:  # ADDED
        if any(r["stats_sampled_patches"] != config["train_patches"] for r in rows):  # ADDED
            raise ValueError("Statistics must cover the full training split")  # ADDED
        count = config[  # ADDED
            {"train": "train_patches", "val": "val_patches", "test": "test_patches"}[  # ADDED
                split  # ADDED
            ]  # ADDED
        ]  # ADDED
        if len({r["id"] for r in rows}) != count or len(rows) != count * (  # ADDED
            3 if split == "test" else 1  # ADDED
        ):  # ADDED
            raise ValueError(f"Incomplete {split} examples for the selected protocol")  # ADDED
    for r in rows:  # ADDED
        if not Path(r["image"]).is_file() or not Path(r["target"]).is_file():  # ADDED
            raise FileNotFoundError("Missing prepared image or target")  # ADDED
        if (  # ADDED
            file_hash(r["image"]) != r["image_sha256"]  # ADDED
            or file_hash(r["target"]) != r["target_sha256"]  # ADDED
        ):  # ADDED
            raise ValueError("Prepared image or target changed")  # ADDED
    return rows  # ADDED


def check_split_controls(train, val, test):  # ADDED
    groups = [{r["roi"] for r in rows} for rows in (train, val, test)]  # ADDED
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):  # ADDED
        raise ValueError("ROI leakage across splits")  # ADDED
    if len({r["stats_sha256"] for rows in (train, val, test) for r in rows}) != 1:  # ADDED
        raise ValueError("All splits must use the same training-only statistics")  # ADDED
