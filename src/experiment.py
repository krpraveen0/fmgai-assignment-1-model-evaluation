"""Controlled training, generation and evaluation for Phase 2."""  # ADDED

import argparse  # ADDED
import csv  # ADDED
import gc  # ADDED
import hashlib  # ADDED
import json  # ADDED
import platform  # ADDED
import random  # ADDED
from pathlib import Path  # ADDED

import numpy as np  # ADDED
from core import file_hash, load_jsonl  # ADDED
from config import load_config, check_examples, check_split_controls  # ADDED

NOISE_POWER = 2.00  # ADDED


def save_json(path, value):  # ADDED
    path = Path(path)  # ADDED
    path.parent.mkdir(parents=True, exist_ok=True)  # ADDED
    path.write_text(json.dumps(value, indent=2) + "\n")  # ADDED


def scheduler_for(run, c):  # ADDED
    if c["noise_power"] != NOISE_POWER:  # ADDED
        raise ValueError("Exponent conflicts with the declared constant")  # ADDED
    from diffusers import DDPMScheduler  # ADDED

    if run not in ("A", "B"):  # ADDED
        raise ValueError("Schedule must be A or B")  # ADDED
    beta = (  # ADDED
        np.linspace(  # ADDED
            np.sqrt(c["beta_start"]),  # ADDED
            np.sqrt(c["beta_end"]),  # ADDED
            c["noise_steps"],  # ADDED
            dtype=np.float64,  # ADDED
        )  # ADDED
        ** 2  # ADDED
    )  # ADDED
    ab = np.cumprod(1 - beta)  # ADDED
    trained = beta if run == "A" else 1 - (1 - beta) ** c["noise_power"]  # ADDED
    scheduler = DDPMScheduler(  # ADDED
        num_train_timesteps=len(beta), trained_betas=trained, prediction_type="epsilon"  # ADDED
    )  # ADDED
    expected = ab if run == "A" else ab ** c["noise_power"]  # ADDED
    np.testing.assert_allclose(  # ADDED
        scheduler.alphas_cumprod.numpy(), expected, rtol=1e-5, atol=1e-7  # ADDED
    )  # ADDED
    return scheduler  # ADDED


def state_hash(model):  # ADDED
    digest = hashlib.sha256()  # ADDED
    for name, value in sorted(model.state_dict().items()):  # ADDED
        digest.update(name.encode())  # ADDED
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())  # ADDED
    return digest.hexdigest()  # ADDED


def image_tensor(path):  # ADDED
    import torch  # ADDED
    from PIL import Image  # ADDED

    with Image.open(path) as image:  # ADDED
        a = np.asarray(image.convert("RGB"), dtype=np.float32)  # ADDED
    if a.shape != (256, 256, 3):  # ADDED
        raise ValueError("Prepared images must be 256x256 RGB")  # ADDED
    return torch.from_numpy(a.copy().transpose(2, 0, 1) / 127.5 - 1)  # ADDED


def choose_device(requested):  # ADDED
    import torch  # ADDED

    if requested != "auto":  # ADDED
        return requested  # ADDED
    return (  # ADDED
        "cuda"  # ADDED
        if torch.cuda.is_available()  # ADDED
        else ("mps" if torch.backends.mps.is_available() else "cpu")  # ADDED
    )  # ADDED


def make_pipeline(c, device, checkpoint=None, components=None):  # ADDED
    import torch  # ADDED
    from diffusers import (  # ADDED
        StableDiffusionInstructPix2PixPipeline,  # ADDED
        UNet2DConditionModel,  # ADDED
        EulerAncestralDiscreteScheduler,  # ADDED
    )  # ADDED

    options = dict(components or {})  # ADDED
    if checkpoint is not None:  # ADDED
        options["unet"] = UNet2DConditionModel.from_pretrained(checkpoint)  # ADDED
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(  # ADDED
        c["model_id"],  # ADDED
        revision=c["model_revision"],  # ADDED
        dtype=torch.float32,  # ADDED
        **options,  # ADDED
    )  # ADDED
    if pipe.unet.config.in_channels != 8:  # ADDED
        raise ValueError("Expected pretrained eight-channel UNet")  # ADDED
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)  # ADDED
    pipe.to(device)  # ADDED
    pipe.enable_attention_slicing()  # ADDED
    pipe.vae.enable_slicing()  # ADDED
    return pipe  # ADDED


def capture_indices(timesteps, count=5):  # ADDED
    values = np.asarray(timesteps, dtype=float)  # ADDED
    targets = np.linspace(values[0], values[-1], count)  # ADDED
    indices = [int(np.argmin(abs(values - target))) for target in targets]  # ADDED
    if len(set(indices)) != count:  # ADDED
        raise ValueError("Cannot select five distinct reverse timesteps")  # ADDED
    return indices  # ADDED


def sample_grid(records, path, trajectories=False):  # ADDED
    from PIL import Image, ImageDraw, ImageFont  # ADDED

    cells = []  # ADDED
    for index, row in enumerate(records[:3]):  # ADDED
        if trajectories:  # ADDED
            cells.extend(  # ADDED
                (  # ADDED
                    state["preview"],  # ADDED
                    f"{row['run']} sample {index} t={state['state_t']:.2f}",  # ADDED
                )  # ADDED
                for state in row["states"]  # ADDED
            )  # ADDED
        else:  # ADDED
            cells.append(  # ADDED
                (row["prediction"], f"{row['run']} {row['id']} {row['product']}"[:36])  # ADDED
            )  # ADDED
    columns = 5 if trajectories else min(3, len(cells))  # ADDED
    canvas = Image.new(  # ADDED
        "RGB", (256 * columns, 280 * ((len(cells) + columns - 1) // columns)), "white"  # ADDED
    )  # ADDED
    draw = ImageDraw.Draw(canvas)  # ADDED
    font = ImageFont.load_default(size=13)  # ADDED
    for index, (filename, label) in enumerate(cells):  # ADDED
        x, y = (index % columns) * 256, (index // columns) * 280  # ADDED
        draw.text((x + 3, y + 3), label, font=font, fill="black")  # ADDED
        with Image.open(filename) as image:  # ADDED
            canvas.paste(image.convert("RGB"), (x, y + 24))  # ADDED
    canvas.save(path)  # ADDED


def generate(pipe, rows, c, outdir, run, trajectories=False):  # ADDED
    import torch  # ADDED
    from PIL import Image  # ADDED

    out = Path(outdir)  # ADDED
    out.mkdir(parents=True, exist_ok=True)  # ADDED
    records = []  # ADDED
    pipe.scheduler.set_timesteps(c["inference_steps"])  # ADDED
    post_times = pipe.scheduler.timesteps.cpu().numpy()[1:]  # ADDED
    targets = capture_indices(post_times, c["trajectory_steps"]) if trajectories else []  # ADDED
    for index, row in enumerate(rows):  # ADDED
        sample_key = f"{row['id']}_{row['product']}"  # ADDED
        seed = c["seed"] + int(hashlib.sha256(sample_key.encode()).hexdigest()[:8], 16)  # ADDED
        states = []  # ADDED

        def callback(pipeline, step, timestep, kwargs):  # ADDED
            if trajectories and step in targets:  # ADDED
                latent = kwargs["latents"].detach().float().cpu()  # ADDED
                state_file = out / f"{run}_{index:05d}_step{step:03d}_latent.pt"  # ADDED
                torch.save(latent, state_file)  # ADDED
                with torch.no_grad():  # ADDED
                    decoded = pipeline.vae.decode(  # ADDED
                        kwargs["latents"].float() / pipeline.vae.config.scaling_factor,  # ADDED
                        return_dict=False,  # ADDED
                    )[0]  # ADDED
                preview = pipeline.image_processor.postprocess(  # ADDED
                    decoded, output_type="pil"  # ADDED
                )[0]  # ADDED
                preview_file = state_file.with_suffix(".png")  # ADDED
                preview.save(preview_file)  # ADDED
                next_t = (  # ADDED
                    float(pipeline.scheduler.timesteps[step + 1])  # ADDED
                    if step + 1 < len(pipeline.scheduler.timesteps)  # ADDED
                    else 0.0  # ADDED
                )  # ADDED
                states.append(  # ADDED
                    {  # ADDED
                        "step_index": step,  # ADDED
                        "transition_from_t": float(timestep),  # ADDED
                        "state_t": next_t,  # ADDED
                        "sigma": float(pipeline.scheduler.sigmas[step + 1]),  # ADDED
                        "latent": str(state_file.resolve()),  # ADDED
                        "preview": str(preview_file.resolve()),  # ADDED
                        "sha256": file_hash(state_file),  # ADDED
                    }  # ADDED
                )  # ADDED
            return kwargs  # ADDED

        with Image.open(row["image"]) as source, torch.inference_mode():  # ADDED
            result = pipe(  # ADDED
                prompt=row["prompt"],  # ADDED
                image=source.convert("RGB"),  # ADDED
                num_inference_steps=c["inference_steps"],  # ADDED
                guidance_scale=c["text_guidance_scale"],  # ADDED
                image_guidance_scale=c["image_guidance_scale"],  # ADDED
                generator=torch.Generator(device="cpu").manual_seed(seed),  # ADDED
                callback_on_step_end=callback,  # ADDED
                callback_on_step_end_tensor_inputs=["latents"],  # ADDED
            )  # ADDED
        path = out / f"{run}_{index:05d}_{sample_key}.png"  # ADDED
        result.images[0].save(path)  # ADDED
        records.append(  # ADDED
            {  # ADDED
                "id": row["id"],  # ADDED
                "product": row["product"],  # ADDED
                "prompt": row["prompt"],  # ADDED
                "seed": seed,  # ADDED
                "run": run,  # ADDED
                "target": row["target"],  # ADDED
                "image": row["image"],  # ADDED
                "image_sha256": file_hash(row["image"]),  # ADDED
                "prediction": str(path.resolve()),  # ADDED
                "prediction_sha256": file_hash(path),  # ADDED
                "target_sha256": file_hash(row["target"]),  # ADDED
                "stats_sha256": row["stats_sha256"],  # ADDED
                "states": states,  # ADDED
            }  # ADDED
        )  # ADDED
        save_json(out / "samples.json", records)  # ADDED
    sample_grid(  # ADDED
        records,  # ADDED
        out / ("trajectory_grid.png" if trajectories else "sample_grid.png"),  # ADDED
        trajectories,  # ADDED
    )  # ADDED
    return records  # ADDED


def load_inputs(args, c):  # ADDED
    paths = [  # ADDED
        Path(args.examples) / split / "examples.jsonl"  # ADDED
        for split in ("train", "val", "test")  # ADDED
    ]  # ADDED
    rows = [  # ADDED
        check_examples(path, split, c, official=not args.diagnostic)  # ADDED
        for path, split in zip(paths, ("train", "val", "test"))  # ADDED
    ]  # ADDED
    check_split_controls(*rows)  # ADDED
    return paths, rows  # ADDED


def train(args):  # ADDED
    import torch  # ADDED
    import torch.nn.functional as F  # ADDED
    from torch.utils.data import Dataset, DataLoader  # ADDED
    from accelerate import Accelerator  # ADDED
    from accelerate.utils import set_seed  # ADDED
    from diffusers import UNet2DConditionModel, AutoencoderKL  # ADDED
    from transformers import CLIPTextModel, CLIPTokenizer  # ADDED

    c = load_config(args.config)  # ADDED
    paths, (train_rows, val_rows, test_rows) = load_inputs(args, c)  # ADDED
    if args.max_steps is not None and args.max_steps < 1:  # ADDED
        raise ValueError("max steps must be positive")  # ADDED
    if args.max_steps and not args.diagnostic:  # ADDED
        raise ValueError("--max-steps requires --diagnostic")  # ADDED
    if args.diagnostic and args.resume:  # ADDED
        raise ValueError("Resume is for complete checkpointed runs")  # ADDED
    accumulation = (  # ADDED
        1 if args.diagnostic else c["effective_batch_size"] // c["micro_batch_size"]  # ADDED
    )  # ADDED
    device = choose_device(args.device or c["device"])  # ADDED
    accelerator = Accelerator(  # ADDED
        cpu=device == "cpu",  # ADDED
        mixed_precision=c["precision"],  # ADDED
        gradient_accumulation_steps=accumulation,  # ADDED
    )  # ADDED
    if (  # ADDED
        accelerator.num_processes != 1  # ADDED
        or str(accelerator.device).split(":")[0] != device  # ADDED
    ):  # ADDED
        raise ValueError("Use one process on the requested backend")  # ADDED
    set_seed(c["seed"])  # ADDED
    output = (  # ADDED
        Path(args.output) / ("diagnostic" if args.diagnostic else "runs") / args.run  # ADDED
    )  # ADDED
    output.mkdir(parents=True, exist_ok=True)  # ADDED
    if (output / "train_loss.csv").exists() and not args.resume:  # ADDED
        raise FileExistsError("Existing run: use --resume or a new output root")  # ADDED
    unet = UNet2DConditionModel.from_pretrained(  # ADDED
        c["model_id"], subfolder="unet", revision=c["model_revision"]  # ADDED
    )  # ADDED
    if unet.config.in_channels != 8:  # ADDED
        raise ValueError("Refusing to replace the pretrained input convolution")  # ADDED
    initial_hash = state_hash(unet)  # ADDED
    vae = (  # ADDED
        AutoencoderKL.from_pretrained(  # ADDED
            c["model_id"], subfolder="vae", revision=c["model_revision"]  # ADDED
        )  # ADDED
        .eval()  # ADDED
        .to(accelerator.device)  # ADDED
    )  # ADDED
    text = (  # ADDED
        CLIPTextModel.from_pretrained(  # ADDED
            c["model_id"], subfolder="text_encoder", revision=c["model_revision"]  # ADDED
        )  # ADDED
        .eval()  # ADDED
        .to(accelerator.device)  # ADDED
    )  # ADDED
    tokenizer = CLIPTokenizer.from_pretrained(  # ADDED
        c["model_id"], subfolder="tokenizer", revision=c["model_revision"]  # ADDED
    )  # ADDED
    vae.requires_grad_(False)  # ADDED
    text.requires_grad_(False)  # ADDED
    unet.enable_gradient_checkpointing()  # ADDED
    optimizer = torch.optim.AdamW(  # ADDED
        unet.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"]  # ADDED
    )  # ADDED

    class Pairs(Dataset):  # ADDED
        def __len__(self):  # ADDED
            return len(train_rows)  # ADDED

        def __getitem__(self, i):  # ADDED
            row = train_rows[i]  # ADDED
            return (  # ADDED
                image_tensor(row["image"]),  # ADDED
                image_tensor(row["target"]),  # ADDED
                row["prompt"],  # ADDED
            )  # ADDED

    loader = DataLoader(  # ADDED
        Pairs(), batch_size=c["micro_batch_size"], shuffle=False, num_workers=0  # ADDED
    )  # ADDED
    unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)  # ADDED
    schedule = scheduler_for(args.run, c)  # ADDED
    native = scheduler_for("A", c)  # ADDED
    resolved = {  # ADDED
        "config": c,  # ADDED
        "run": args.run,  # ADDED
        "diagnostic": args.diagnostic,  # ADDED
        "actual_effective_batch_size": accumulation * c["micro_batch_size"],  # ADDED
        "initial_model_sha256": initial_hash,  # ADDED
        "input_hashes": {  # ADDED
            s: file_hash(p) for s, p in zip(("train", "val", "test"), paths)  # ADDED
        },  # ADDED
        "device": str(accelerator.device),  # ADDED
        "python": platform.python_version(),  # ADDED
        "torch": torch.__version__,  # ADDED
    }  # ADDED
    start_epoch = 0  # ADDED
    if args.resume:  # ADDED
        prior = json.loads((Path(args.resume) / "progress.json").read_text())  # ADDED
        if prior["resolved"] != resolved:  # ADDED
            raise ValueError("Resume configuration/inputs/initial checkpoint differ")  # ADDED
        accelerator.load_state(str(Path(args.resume) / "state"))  # ADDED
        if device == "mps":  # ADDED
            torch.mps.set_rng_state(  # ADDED
                torch.load(  # ADDED
                    Path(args.resume) / "state/mps_rng.pt",  # ADDED
                    map_location="cpu",  # ADDED
                    weights_only=True,  # ADDED
                )  # ADDED
            )  # ADDED
        start_epoch = prior["epoch"]  # ADDED
    partner = (  # ADDED
        Path(args.output)  # ADDED
        / ("diagnostic" if args.diagnostic else "runs")  # ADDED
        / ("B" if args.run == "A" else "A")  # ADDED
        / "resolved.json"  # ADDED
    )  # ADDED
    if partner.exists():  # ADDED
        prior = json.loads(partner.read_text())  # ADDED
        for key in (  # ADDED
            "config",  # ADDED
            "input_hashes",  # ADDED
            "initial_model_sha256",  # ADDED
            "actual_effective_batch_size",  # ADDED
        ):  # ADDED
            if prior[key] != resolved[key]:  # ADDED
                raise ValueError(f"A/B control mismatch: {key}")  # ADDED
    save_json(output / "resolved.json", resolved)  # ADDED
    save_json(output / "in_progress.json", {"run":args.run,"resume_epoch":start_epoch})  # ADDED
    np.savez(  # ADDED
        output / "schedule.npz",  # ADDED
        betas=schedule.betas.cpu().numpy(),  # ADDED
        alphas_cumprod=schedule.alphas_cumprod.cpu().numpy(),  # ADDED
    )  # ADDED
    log_path = output / ("diagnostic_loss.csv" if args.diagnostic else "train_loss.csv")  # ADDED
    history = list(csv.DictReader(log_path.open())) if log_path.exists() else []  # ADDED
    history = [r for r in history if int(r["epoch"]) <= start_epoch]  # ADDED

    def loss_batch(src, tgt, prompts, sch, dropout=True):  # ADDED
        with torch.no_grad(), accelerator.autocast():  # ADDED
            lat = (  # ADDED
                vae.encode(tgt.to(accelerator.device)).latent_dist.sample()  # ADDED
                * vae.config.scaling_factor  # ADDED
            )  # ADDED
            cond = vae.encode(src.to(accelerator.device)).latent_dist.mode()  # ADDED
            tokens = tokenizer(  # ADDED
                list(prompts),  # ADDED
                padding="max_length",  # ADDED
                max_length=tokenizer.model_max_length,  # ADDED
                truncation=True,  # ADDED
                return_tensors="pt",  # ADDED
            ).input_ids.to(accelerator.device)  # ADDED
            enc = text(tokens)[0]  # ADDED
            if dropout:  # ADDED
                p = c["conditioning_dropout"]  # ADDED
                random = torch.rand(len(src), device=accelerator.device)  # ADDED
                empty = tokenizer(  # ADDED
                    [""] * len(src),  # ADDED
                    padding="max_length",  # ADDED
                    max_length=tokenizer.model_max_length,  # ADDED
                    return_tensors="pt",  # ADDED
                ).input_ids.to(accelerator.device)  # ADDED
                enc = torch.where((random < 2 * p)[:, None, None], text(empty)[0], enc)  # ADDED
                cond = cond * (~((random >= p) & (random < 3 * p)))[:, None, None, None]  # ADDED
            noise = torch.randn_like(lat)  # ADDED
            t = torch.randint(  # ADDED
                0, c["noise_steps"], (len(src),), device=accelerator.device  # ADDED
            )  # ADDED
            noisy = sch.add_noise(lat, noise, t)  # ADDED
        with accelerator.autocast():  # ADDED
            prediction = unet(  # ADDED
                torch.cat([noisy, cond], dim=1), t, enc, return_dict=False  # ADDED
            )[0]  # ADDED
            return F.mse_loss(prediction.float(), noise.float())  # ADDED

    steps = 0  # ADDED
    for epoch in range(start_epoch + 1, c["train_epochs"] + 1):  # ADDED
        unet.train()  # ADDED
        total, seen = 0.0, 0  # ADDED
        for batch_index, (src, tgt, prompts) in enumerate(loader):  # ADDED
            with accelerator.accumulate(unet):  # ADDED
                loss = loss_batch(src, tgt, prompts, schedule)  # ADDED
                if not torch.isfinite(loss):  # ADDED
                    raise FloatingPointError("Nonfinite training loss")  # ADDED
                group_start = (  # ADDED
                    (batch_index // accumulation) * accumulation * c["micro_batch_size"]  # ADDED
                )  # ADDED
                group_samples = min(  # ADDED
                    accumulation * c["micro_batch_size"], len(train_rows) - group_start  # ADDED
                )  # ADDED
                accelerator.backward(loss * accumulation * len(src) / group_samples)  # ADDED
                if accelerator.sync_gradients:  # ADDED
                    norm = accelerator.clip_grad_norm_(  # ADDED
                        unet.parameters(), c["gradient_clip"]  # ADDED
                    )  # ADDED
                    if not torch.isfinite(norm):  # ADDED
                        raise FloatingPointError("Nonfinite gradient norm")  # ADDED
                optimizer.step()  # ADDED
                if accelerator.optimizer_step_was_skipped:  # ADDED
                    raise FloatingPointError("Optimizer update skipped")  # ADDED
                optimizer.zero_grad()  # ADDED
            total += float(loss.detach()) * len(src)  # ADDED
            seen += len(src)  # ADDED
            steps += 1  # ADDED
            if args.diagnostic and steps >= (args.max_steps or 1):  # ADDED
                break  # ADDED
        history.append({"epoch": epoch, "train_loss": total / seen})  # ADDED
        with log_path.open("w", newline="") as stream:  # ADDED
            writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss"])  # ADDED
            writer.writeheader()  # ADDED
            writer.writerows(history)  # ADDED
        if args.diagnostic:  # ADDED
            save_json(  # ADDED
                output / "diagnostic.json",  # ADDED
                {  # ADDED
                    "micro_steps": steps,  # ADDED
                    "loss": total / seen,  # ADDED
                    "optimizer_update_verified": state_hash(  # ADDED
                        accelerator.unwrap_model(unet)  # ADDED
                    )  # ADDED
                    != initial_hash,  # ADDED
                    "submission_result": False,  # ADDED
                },  # ADDED
            )  # ADDED
            break  # ADDED
        unet.eval()  # ADDED
        saved_python_rng = random.getstate()  # ADDED
        saved_numpy_rng = np.random.get_state()  # ADDED
        saved_rng = torch.get_rng_state()  # ADDED
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None  # ADDED
        mps_rng = torch.mps.get_rng_state() if device == "mps" else None  # ADDED
        try:  # ADDED
            set_seed(c["seed"] + 999)  # ADDED
            values = []  # ADDED
            with torch.no_grad():  # ADDED
                for row in val_rows[: c["validation_examples"]]:  # ADDED
                    value = loss_batch(  # ADDED
                        image_tensor(row["image"])[None],  # ADDED
                        image_tensor(row["target"])[None],  # ADDED
                        [row["prompt"]],  # ADDED
                        native,  # ADDED
                        dropout=False,  # ADDED
                    )  # ADDED
                    values.append(float(value))  # ADDED
            if not np.all(np.isfinite(values)):  # ADDED
                raise FloatingPointError("Nonfinite validation loss")  # ADDED
            save_json(  # ADDED
                output / f"validation/epoch_{epoch:03d}.json",  # ADDED
                {  # ADDED
                    "epoch": epoch,  # ADDED
                    "metric": c["validation_metric"],  # ADDED
                    "value": float(np.mean(values)),  # ADDED
                    "count": len(values),  # ADDED
                    "native_schedule_for_both_runs": True,  # ADDED
                },  # ADDED
            )  # ADDED
            if epoch in c["checkpoints_epochs"]:  # ADDED
                checkpoint = output / f"checkpoints/epoch_{epoch:03d}"  # ADDED
                checkpoint.mkdir(parents=True, exist_ok=True)  # ADDED
                accelerator.unwrap_model(unet).save_pretrained(checkpoint / "unet")  # ADDED
                pipe = make_pipeline(  # ADDED
                    c,  # ADDED
                    str(accelerator.device),  # ADDED
                    components={  # ADDED
                        "unet": accelerator.unwrap_model(unet),  # ADDED
                        "vae": vae,  # ADDED
                        "text_encoder": text,  # ADDED
                        "tokenizer": tokenizer,  # ADDED
                    },  # ADDED
                )  # ADDED
                generate(  # ADDED
                    pipe,  # ADDED
                    val_rows[: c["evidence_samples"]],  # ADDED
                    c,  # ADDED
                    checkpoint / "samples",  # ADDED
                    args.run,  # ADDED
                )  # ADDED
                pipe.disable_attention_slicing()  # ADDED
                del pipe  # ADDED
        finally:  # ADDED
            random.setstate(saved_python_rng)  # ADDED
            np.random.set_state(saved_numpy_rng)  # ADDED
            torch.set_rng_state(saved_rng)  # ADDED
            if cuda_rng is not None:  # ADDED
                torch.cuda.set_rng_state_all(cuda_rng)  # ADDED
            if mps_rng is not None:  # ADDED
                torch.mps.set_rng_state(mps_rng)  # ADDED
        if epoch in c["checkpoints_epochs"]:  # ADDED
            accelerator.save_state(str(checkpoint / "state"))  # ADDED
            if mps_rng is not None:  # ADDED
                torch.save(mps_rng, checkpoint / "state/mps_rng.pt")  # ADDED
            save_json(  # ADDED
                checkpoint / "progress.json", {"epoch": epoch, "resolved": resolved}  # ADDED
            )  # ADDED
        print(  # ADDED
            f"Run {args.run}, epoch {epoch}, train loss {total / seen:.6f}", flush=True  # ADDED
        )  # ADDED
    if not args.diagnostic:  # ADDED
        save_json(  # ADDED
            output / "complete.json",  # ADDED
            {  # ADDED
                "completed_epochs": c["train_epochs"],  # ADDED
                "initial_model_sha256": initial_hash,  # ADDED
                "final_checkpoint": str(  # ADDED
                    (  # ADDED
                        output / f"checkpoints/epoch_{c['train_epochs']:03d}/unet"  # ADDED
                    ).resolve()  # ADDED
                ),  # ADDED
            },  # ADDED
        )  # ADDED

    (output / "in_progress.json").unlink(missing_ok=True)  # ADDED


def infer(args):  # ADDED
    c = load_config(args.config)  # ADDED
    rows = check_examples(Path(args.examples) / "test/examples.jsonl", "test", c)  # ADDED
    checkpoint = None  # ADDED
    if args.run != "0":  # ADDED
        if (Path(args.output)/f"runs/{args.run}/in_progress.json").exists():  # ADDED
            raise ValueError("Training is unfinished; resume it before inference")  # ADDED
        resolved = json.loads(  # ADDED
            (Path(args.output) / f"runs/{args.run}/resolved.json").read_text()  # ADDED
        )  # ADDED
        if resolved["config"] != c or resolved["diagnostic"]:  # ADDED
            raise ValueError("Inference config differs from the completed training run")  # ADDED
        complete = json.loads(  # ADDED
            (Path(args.output) / f"runs/{args.run}/complete.json").read_text()  # ADDED
        )  # ADDED
        if resolved["input_hashes"]["test"] != file_hash(  # ADDED
            Path(args.examples) / "test/examples.jsonl"  # ADDED
        ):  # ADDED
            raise ValueError("Test examples changed since training")  # ADDED
        checkpoint = complete["final_checkpoint"]  # ADDED
    pipe = make_pipeline(c, choose_device(args.device or c["device"]), checkpoint)  # ADDED
    generated = generate(  # ADDED
        pipe, rows, c, Path(args.output) / f"inference/{args.run}", "Run" + args.run  # ADDED
    )  # ADDED
    if args.run != "0":  # ADDED
        generate(  # ADDED
            pipe,  # ADDED
            rows[: c["evidence_samples"]],  # ADDED
            c,  # ADDED
            Path(args.output) / f"trajectories/{args.run}",  # ADDED
            "Run" + args.run,  # ADDED
            trajectories=True,  # ADDED
        )  # ADDED
    save_json(  # ADDED
        Path(args.output) / f"inference/{args.run}/protocol.json",  # ADDED
        {  # ADDED
            "config": c,  # ADDED
            "device": str(pipe.device),  # ADDED
            "checkpoint": checkpoint or c["model_revision"],  # ADDED
            "count": len(generated),  # ADDED
        },  # ADDED
    )  # ADDED


def evaluate(args):  # ADDED
    import torch  # ADDED
    import lpips  # ADDED
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # ADDED
    from PIL import Image  # ADDED

    c = load_config(args.config)  # ADDED
    root = Path(args.output)  # ADDED
    records = {  # ADDED
        r: json.loads((root / f"inference/{r}/samples.json").read_text())  # ADDED
        for r in ("0", "A", "B")  # ADDED
    }  # ADDED
    protocols = [  # ADDED
        json.loads((root / f"inference/{r}/protocol.json").read_text())  # ADDED
        for r in ("0", "A", "B")  # ADDED
    ]  # ADDED
    if any(  # ADDED
        p["config"] != c or p["device"] != protocols[0]["device"] for p in protocols  # ADDED
    ):  # ADDED
        raise ValueError("Inference protocol/backend differs across runs")  # ADDED
    keys = lambda rows: [  # ADDED
        (  # ADDED
            r["id"],  # ADDED
            r["product"],  # ADDED
            r["prompt"],  # ADDED
            r["image_sha256"],  # ADDED
            r["seed"],  # ADDED
            r["target_sha256"],  # ADDED
            r["stats_sha256"],  # ADDED
        )  # ADDED
        for r in rows  # ADDED
    ]  # ADDED
    if keys(records["0"]) != keys(records["A"]) or keys(records["A"]) != keys(  # ADDED
        records["B"]  # ADDED
    ):  # ADDED
        raise ValueError("Evaluation samples or preprocessing differ")  # ADDED
    if len(records["0"]) != c["test_patches"] * 3:  # ADDED
        raise ValueError("Incomplete full test evaluation")  # ADDED
    metric = lpips.LPIPS(net="alex").eval()  # ADDED
    summary = []  # ADDED
    for run, rows in records.items():  # ADDED
        scores = []  # ADDED
        for row in rows:  # ADDED
            if (  # ADDED
                file_hash(row["prediction"]) != row["prediction_sha256"]  # ADDED
                or file_hash(row["target"]) != row["target_sha256"]  # ADDED
            ):  # ADDED
                raise ValueError("Image changed after generation")  # ADDED
            with (  # ADDED
                Image.open(row["prediction"]) as pred,  # ADDED
                Image.open(row["target"]) as truth,  # ADDED
            ):  # ADDED
                a = np.asarray(pred.convert("RGB"), dtype=np.float32) / 255  # ADDED
                b = np.asarray(truth.convert("RGB"), dtype=np.float32) / 255  # ADDED
            with torch.no_grad():  # ADDED
                perceptual = float(  # ADDED
                    metric(  # ADDED
                        torch.from_numpy(a.transpose(2, 0, 1).copy())[None] * 2 - 1,  # ADDED
                        torch.from_numpy(b.transpose(2, 0, 1).copy())[None] * 2 - 1,  # ADDED
                    ).item()  # ADDED
                )  # ADDED
            scores.append(  # ADDED
                {  # ADDED
                    "id": row["id"],  # ADDED
                    "product": row["product"],  # ADDED
                    "lpips": perceptual,  # ADDED
                    "psnr": float(peak_signal_noise_ratio(b, a, data_range=1)),  # ADDED
                    "ssim": float(  # ADDED
                        structural_similarity(b, a, channel_axis=-1, data_range=1)  # ADDED
                    ),  # ADDED
                }  # ADDED
            )  # ADDED
        save_json(root / f"metrics/{run}_per_sample.json", scores)  # ADDED
        summary.append(  # ADDED
            {  # ADDED
                "run": run,  # ADDED
                "samples": len(scores),  # ADDED
                **{  # ADDED
                    k: float(np.mean([s[k] for s in scores]))  # ADDED
                    for k in ("lpips", "psnr", "ssim")  # ADDED
                },  # ADDED
            }  # ADDED
        )  # ADDED
    save_json(  # ADDED
        root / "metrics/provenance.json",  # ADDED
        {  # ADDED
            "config": c,  # ADDED
            "samples": {  # ADDED
                r: file_hash(root / f"inference/{r}/samples.json")  # ADDED
                for r in ("0", "A", "B")  # ADDED
            },  # ADDED
            "losses": {  # ADDED
                r: file_hash(root / f"runs/{r}/train_loss.csv") for r in ("A", "B")  # ADDED
            },  # ADDED
        },  # ADDED
    )  # ADDED
    save_json(root / "metrics/summary.json", summary)  # ADDED
    with (root / "metrics/summary.csv").open("w", newline="") as stream:  # ADDED
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))  # ADDED
        writer.writeheader()  # ADDED
        writer.writerows(summary)  # ADDED
    import matplotlib.pyplot as plt  # ADDED

    for run in ("A", "B"):  # ADDED
        rows = list(csv.DictReader((root / f"runs/{run}/train_loss.csv").open()))  # ADDED
        plt.plot(  # ADDED
            [int(r["epoch"]) for r in rows],  # ADDED
            [float(r["train_loss"]) for r in rows],  # ADDED
            label=run,  # ADDED
        )  # ADDED
    plt.xlabel("Epoch")  # ADDED
    plt.ylabel("Training epsilon MSE")  # ADDED
    plt.legend()  # ADDED
    plt.tight_layout()  # ADDED
    plt.savefig(root / "metrics/loss_curves.png", dpi=180)  # ADDED
    plt.close()  # ADDED


def main():  # ADDED
    parser = argparse.ArgumentParser(description="Phase 2 controlled experiments")  # ADDED
    parser.add_argument("command", choices=["train", "infer", "evaluate"])  # ADDED
    parser.add_argument("--config", default="experiment.json")  # ADDED
    parser.add_argument("--examples", default="outputs/prepared")  # ADDED
    parser.add_argument("--output", default="outputs/experiment")  # ADDED
    parser.add_argument("--run", choices=["0", "A", "B"])  # ADDED
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])  # ADDED
    parser.add_argument("--resume")  # ADDED
    parser.add_argument("--diagnostic", action="store_true")  # ADDED
    parser.add_argument("--max-steps", type=int)  # ADDED
    args = parser.parse_args()  # ADDED
    if args.command == "train" and args.run not in ("A", "B"):  # ADDED
        parser.error("train needs --run A or B")  # ADDED
    if args.command == "infer" and args.run is None:  # ADDED
        parser.error("infer needs --run 0, A or B")  # ADDED
    {"train": train, "infer": infer, "evaluate": evaluate}[args.command](args)  # ADDED


if __name__ == "__main__":  # ADDED
    main()  # ADDED
