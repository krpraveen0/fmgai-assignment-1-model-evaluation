"""Tiny synthetic models exercise orchestration, not model quality."""  # ADDED

import json  # ADDED
from types import SimpleNamespace  # ADDED
from pathlib import Path  # ADDED
import numpy as np  # ADDED
import pytest  # ADDED


def test_complete_training_checkpoint_resume_and_inference(tmp_path, monkeypatch):  # ADDED
    torch = pytest.importorskip("torch")  # ADDED
    from PIL import Image  # ADDED
    from diffusers import UNet2DConditionModel, AutoencoderKL  # ADDED
    from transformers import CLIPTextModel, CLIPTokenizer  # ADDED
    import experiment as experiment  # ADDED
    from config import load_config  # ADDED
    from core import file_hash  # ADDED

    class UNet(torch.nn.Module):  # ADDED
        def __init__(self):  # ADDED
            super().__init__()  # ADDED
            self.conv = torch.nn.Conv2d(8, 4, 1)  # ADDED
            self.config = SimpleNamespace(in_channels=8)  # ADDED

        def enable_gradient_checkpointing(self):  # ADDED
            pass  # ADDED

        def forward(self, x, t, enc, return_dict=False):  # ADDED
            return (self.conv(x.float()),)  # ADDED

        def save_pretrained(self, path):  # ADDED
            Path(path).mkdir(parents=True, exist_ok=True)  # ADDED
            torch.save(self.state_dict(), Path(path) / "weights.pt")  # ADDED

    class VAE(torch.nn.Module):  # ADDED
        config = SimpleNamespace(scaling_factor=0.18215)  # ADDED

        def encode(self, x):  # ADDED
            lat = torch.nn.functional.avg_pool2d(x, 8)  # ADDED
            lat = torch.cat([lat, lat[:, :1]], 1)  # ADDED
            return SimpleNamespace(  # ADDED
                latent_dist=SimpleNamespace(sample=lambda: lat, mode=lambda: lat)  # ADDED
            )  # ADDED

        def decode(self, x, return_dict=False):  # ADDED
            return (torch.nn.functional.interpolate(x[:, :3], scale_factor=8),)  # ADDED

    class Text(torch.nn.Module):  # ADDED
        def forward(self, t):  # ADDED
            return (torch.zeros(t.shape[0], t.shape[1], 8),)  # ADDED

    class Tokenizer:  # ADDED
        model_max_length = 4  # ADDED

        def __call__(self, strings, **kwargs):  # ADDED
            return SimpleNamespace(  # ADDED
                input_ids=torch.ones(len(strings), 4, dtype=torch.long)  # ADDED
            )  # ADDED

    for cls, factory in (  # ADDED
        (UNet2DConditionModel, UNet),  # ADDED
        (AutoencoderKL, VAE),  # ADDED
        (CLIPTextModel, Text),  # ADDED
        (CLIPTokenizer, Tokenizer),  # ADDED
    ):  # ADDED
        monkeypatch.setattr(  # ADDED
            cls, "from_pretrained", lambda *a, _factory=factory, **k: _factory()  # ADDED
        )  # ADDED

    class Schedule:  # ADDED
        def set_timesteps(self, n):  # ADDED
            self.timesteps = torch.linspace(999, 0, n)  # ADDED
            self.sigmas = torch.linspace(1, 0, n + 1)  # ADDED

    class Pipe:  # ADDED
        def __init__(self):  # ADDED
            self.scheduler = Schedule()  # ADDED
            self.vae = VAE()  # ADDED
            self.device = torch.device("cpu")  # ADDED
            self.image_processor = SimpleNamespace(  # ADDED
                postprocess=lambda a, **k: [Image.new("RGB", (256, 256))]  # ADDED
            )  # ADDED

        def disable_attention_slicing(self):  # ADDED
            pass  # ADDED

        def __call__(self, **kwargs):  # ADDED
            self.scheduler.set_timesteps(kwargs["num_inference_steps"])  # ADDED
            for i, t in enumerate(self.scheduler.timesteps):  # ADDED
                kwargs["callback_on_step_end"](  # ADDED
                    self, i, t, {"latents": torch.zeros(1, 4, 32, 32)}  # ADDED
                )  # ADDED
            return SimpleNamespace(images=[Image.new("RGB", (256, 256))])  # ADDED

    monkeypatch.setattr(experiment, "make_pipeline", lambda *a, **k: Pipe())  # ADDED
    c = load_config()  # ADDED
    c.update(  # ADDED
        train_epochs=5,  # ADDED
        checkpoints_epochs=[1, 2, 3, 4, 5],  # ADDED
        train_patches=2,  # ADDED
        val_patches=3,  # ADDED
        test_patches=3,  # ADDED
        precision="no",  # ADDED
        effective_batch_size=2,  # ADDED
        validation_examples=1,  # ADDED
        inference_steps=6,  # ADDED
    )  # ADDED
    config = tmp_path / "config.json"  # ADDED
    config.write_text(json.dumps(c))  # ADDED
    prepared = tmp_path / "prepared"  # ADDED
    for split, count in (("train", 2), ("val", 3), ("test", 3)):  # ADDED
        folder = prepared / split  # ADDED
        folder.mkdir(parents=True)  # ADDED
        rows = []  # ADDED
        for i in range(count):  # ADDED
            path = folder / f"{i}.png"  # ADDED
            Image.fromarray(np.full((256, 256, 3), 80 + i, dtype=np.uint8)).save(path)  # ADDED
            for product in (  # ADDED
                ("true_color", "false_color_ir", "shortwave_ir")  # ADDED
                if split == "test"  # ADDED
                else ("true_color",)  # ADDED
            ):  # ADDED
                rows.append(  # ADDED
                    {  # ADDED
                        "id": f"{split}_{i}",  # ADDED
                        "roi": split,  # ADDED
                        "split": split,  # ADDED
                        "product": product,  # ADDED
                        "prompt": "Create the composite.",  # ADDED
                        "image": str(path),  # ADDED
                        "target": str(path),  # ADDED
                        "stats_sha256": "fixture",  # ADDED
                        "manifest_sha256": split,  # ADDED
                        "stats_sampled_patches": 2,  # ADDED
                        "image_sha256": file_hash(path),  # ADDED
                        "target_sha256": file_hash(path),  # ADDED
                    }  # ADDED
                )  # ADDED
        (folder / "examples.jsonl").write_text(  # ADDED
            "".join(json.dumps(r) + "\n" for r in rows)  # ADDED
        )  # ADDED
    args = SimpleNamespace(  # ADDED
        config=config,  # ADDED
        examples=prepared,  # ADDED
        output=tmp_path / "results",  # ADDED
        run="A",  # ADDED
        device="cpu",  # ADDED
        diagnostic=False,  # ADDED
        max_steps=None,  # ADDED
        resume=None,  # ADDED
    )  # ADDED
    experiment.train(args)  # ADDED
    assert len((args.output / "runs/A/train_loss.csv").read_text().splitlines()) == 6  # ADDED
    assert (  # ADDED
        len(list((args.output / "runs/A/checkpoints").glob("*/samples/samples.json")))  # ADDED
        == 5  # ADDED
    )  # ADDED
    saved_final = torch.load(  # ADDED
        args.output / "runs/A/checkpoints/epoch_005/unet/weights.pt", weights_only=True  # ADDED
    )  # ADDED
    args.resume = args.output / "runs/A/checkpoints/epoch_004"  # ADDED
    experiment.train(args)  # ADDED
    assert len((args.output / "runs/A/train_loss.csv").read_text().splitlines()) == 6  # ADDED
    resumed_final = torch.load(  # ADDED
        args.output / "runs/A/checkpoints/epoch_005/unet/weights.pt", weights_only=True  # ADDED
    )  # ADDED
    assert all(torch.equal(saved_final[k], resumed_final[k]) for k in saved_final)  # ADDED
    experiment.infer(args)  # ADDED
    rows = json.loads((args.output / "inference/A/samples.json").read_text())  # ADDED
    assert len(rows) == 9  # ADDED
    states = json.loads((args.output / "trajectories/A/samples.json").read_text())  # ADDED
    assert len(states) == 3 and all(len(r["states"]) == 5 for r in states)  # ADDED
    args.run = "B"  # ADDED
    args.resume = None  # ADDED
    experiment.train(args)  # ADDED
    assert (  # ADDED
        json.loads((args.output / "runs/A/resolved.json").read_text())[  # ADDED
            "initial_model_sha256"  # ADDED
        ]  # ADDED
        == json.loads((args.output / "runs/B/resolved.json").read_text())[  # ADDED
            "initial_model_sha256"  # ADDED
        ]  # ADDED
    )  # ADDED

    pytest.importorskip("skimage")  # ADDED
    lpips = pytest.importorskip("lpips")  # ADDED

    class FixtureMetric(torch.nn.Module):  # ADDED
        def forward(self, a, b):  # ADDED
            return ((a - b) ** 2).mean()  # ADDED

    monkeypatch.setattr(lpips, "LPIPS", lambda **k: FixtureMetric())  # ADDED
    experiment.infer(args)  # ADDED
    args.run = "0"  # ADDED
    experiment.infer(args)  # ADDED
    experiment.evaluate(args)  # ADDED
    summary = json.loads((args.output / "metrics/summary.json").read_text())  # ADDED
    assert {r["run"] for r in summary} == {"0", "A", "B"}  # ADDED
    assert all(r["samples"] == 9 for r in summary)  # ADDED
