# ADDED
"""SEN12MS preparation and noise-schedule checks."""  # ADDED

import csv  # ADDED
import hashlib  # ADDED
import json  # ADDED
import re  # ADDED
from collections import defaultdict  # ADDED
from pathlib import Path  # ADDED

import numpy as np  # ADDED

SEED = 20261004  # ADDED
NOISE_POWER = 2.00  # ADDED
MODEL_ID = "timbrooks/instruct-pix2pix"  # MODIFIED
MODEL_REVISION = "31519b5cb02a7fd89b906d88731cd4d6a7bbf88d"  # MODIFIED
TRAIN_N, VAL_N, TEST_N = 10000, 1000, 2000  # ADDED
PRODUCTS = {  # ADDED
    "true_color": ("B04", "B03", "B02"),  # MODIFIED
    "false_color_ir": ("B08", "B04", "B03"),  # MODIFIED
    "shortwave_ir": ("B12", "B8A", "B04"),  # MODIFIED
}  # ADDED
# ADDED: Prompts name only the requested composite.
PROMPTS = {  # ADDED
    "true_color": "Create the true-colour Sentinel-2 composite.",  # MODIFIED
    "false_color_ir": "Create the false-colour infrared Sentinel-2 composite.",  # MODIFIED
    "shortwave_ir": "Create the shortwave-infrared Sentinel-2 composite.",  # MODIFIED
}  # ADDED
# ADDED: Eight train/validation templates and four test templates.
TEMPLATES = {  # ADDED
    "true_color": [  # MODIFIED
        "Create the true-colour Sentinel-2 composite.",  # MODIFIED
        "Produce the true-colour optical view.",  # MODIFIED
        "Render this SAR patch in true colour.",  # MODIFIED
        "Show the matching true-colour Sentinel-2 image.",  # MODIFIED
        "Generate the RGB natural-colour composite.",  # MODIFIED
        "Make a true-colour satellite rendering.",  # MODIFIED
        "Create the visible-band colour product.",  # MODIFIED
        "Generate a true-colour view of this SAR image.",  # MODIFIED
        "Render the corresponding natural RGB product.",  # MODIFIED
        "Produce the visible-light colour composite.",  # MODIFIED
        "Show this area using true-colour optical bands.",  # MODIFIED
        "Convert this SAR input into true-colour Sentinel-2.",  # MODIFIED
    ],  # ADDED
    "false_color_ir": [  # MODIFIED
        "Create the false-colour infrared composite.",  # MODIFIED
        "Produce a false-colour NIR view.",  # MODIFIED
        "Render this SAR patch as false-colour infrared.",  # MODIFIED
        "Show the matching false-colour optical view.",  # MODIFIED
        "Generate the B8-B4-B3 composite.",  # MODIFIED
        "Create the false-colour infrared satellite rendering.",  # MODIFIED
        "Generate the NIR-R-G band composite.",  # MODIFIED
        "Produce the infrared false-colour product.",  # MODIFIED
        "Render the corresponding false-colour infrared image.",  # MODIFIED
        "Show the false-colour NIR composite.",  # MODIFIED
        "Create the optical composite using NIR, red and green.",  # MODIFIED
        "Convert this SAR input to false-colour infrared.",  # MODIFIED
    ],  # ADDED
    "shortwave_ir": [  # MODIFIED
        "Create the shortwave-infrared composite.",  # MODIFIED
        "Produce a SWIR composite view.",  # MODIFIED
        "Render the SAR patch using the SWIR product.",  # MODIFIED
        "Show the matching shortwave-infrared view.",  # MODIFIED
        "Generate the B12-B8A-B4 composite.",  # MODIFIED
        "Create the shortwave-infrared satellite rendering.",  # MODIFIED
        "Generate the SWIR-NIR-red composite.",  # MODIFIED
        "Produce the shortwave-infrared optical product.",  # MODIFIED
        "Render the corresponding shortwave-infrared image.",  # MODIFIED
        "Show this patch as a SWIR composite.",  # MODIFIED
        "Create the optical product using B12, B8A and B4.",  # MODIFIED
        "Convert this SAR input to shortwave infrared.",  # MODIFIED
    ],  # MODIFIED
}  # ADDED
assert all(len(v) == 12 for v in TEMPLATES.values())  # ADDED

S2_BANDS = (  # MODIFIED
    "B01",  # MODIFIED
    "B02",  # MODIFIED
    "B03",  # MODIFIED
    "B04",  # MODIFIED
    "B05",  # MODIFIED
    "B06",  # MODIFIED
    "B07",  # MODIFIED
    "B08",  # MODIFIED
    "B8A",  # MODIFIED
    "B09",  # MODIFIED
    "B10",  # MODIFIED
    "B11",  # MODIFIED
    "B12",  # MODIFIED
)  # MODIFIED
STAT_CHANNELS = (  # MODIFIED
    "SAR_VV",  # MODIFIED
    "SAR_VH",  # MODIFIED
    "SAR_DIFF",  # MODIFIED
    "B02",  # MODIFIED
    "B03",  # MODIFIED
    "B04",  # MODIFIED
    "B08",  # MODIFIED
    "B8A",  # MODIFIED
    "B12",  # MODIFIED
)  # MODIFIED


def native_schedule():  # ADDED
    """Exact scaled_linear beta schedule in the declared 1000-step checkpoint."""  # ADDED
    beta = np.linspace(np.sqrt(0.00085), np.sqrt(0.012), 1000, dtype=np.float64) ** 2  # MODIFIED
    alpha = 1.0 - beta  # ADDED
    ab = np.cumprod(alpha)  # ADDED
    return beta, alpha, ab  # ADDED


def power_schedule():  # ADDED
    """Reconstruct beta' from declared theta'=alpha_bar**2.00."""  # ADDED
    beta, alpha, ab = native_schedule()  # ADDED
    ap = alpha**NOISE_POWER  # MODIFIED
    bp = 1.0 - ap  # ADDED
    abp = np.cumprod(ap)  # ADDED
    np.testing.assert_allclose(abp, ab**NOISE_POWER, rtol=1e-11, atol=1e-15)  # MODIFIED
    assert np.all((ab > 0) & (ab < 1))  # ADDED
    assert np.all((abp > 0) & (abp < 1))  # ADDED
    assert np.all((bp > 0) & (bp < 1))  # ADDED
    assert 1.5 < NOISE_POWER < 3  # ADDED
    return bp, abp  # ADDED


def save_schedule(outdir):  # ADDED
    import matplotlib.pyplot as plt  # ADDED

    outdir = Path(outdir)  # MODIFIED
    outdir.mkdir(parents=True, exist_ok=True)  # MODIFIED
    beta, alpha, ab = native_schedule()  # MODIFIED
    bp, abp = power_schedule()  # MODIFIED
    rows = []  # ADDED
    for i in range(1000):  # ADDED
        sa = ab[i] / (1 - ab[i])  # MODIFIED
        sb = abp[i] / (1 - abp[i])  # MODIFIED
        rows.append(  # MODIFIED
            [i + 1, beta[i], bp[i], alpha[i], ab[i], abp[i], np.log(sa), np.log(sb)]  # MODIFIED
        )  # MODIFIED
    header = [  # MODIFIED
        "t",  # MODIFIED
        "beta_native",  # MODIFIED
        "beta_powered",  # MODIFIED
        "alpha_native",  # MODIFIED
        "alpha_bar_native",  # MODIFIED
        "alpha_bar_powered",  # MODIFIED
        "log_snr_native",  # MODIFIED
        "log_snr_powered",  # MODIFIED
    ]  # MODIFIED
    with (outdir / "schedule_all_1000.csv").open("w", newline="") as f:  # MODIFIED
        w = csv.writer(f)  # MODIFIED
        w.writerow(header)  # MODIFIED
        w.writerows(rows)  # MODIFIED
    with (outdir / "schedule_five_steps.csv").open("w", newline="") as f:  # MODIFIED
        w = csv.writer(f)  # MODIFIED
        w.writerow(header)  # MODIFIED
        w.writerows(rows[i] for i in [0, 249, 499, 749, 999])  # MODIFIED
    t = np.arange(1, 1001)  # ADDED
    fig, ax = plt.subplots(figsize=(9, 5))  # ADDED
    ax.semilogy(t, ab, label="Run A: native alpha-bar", linewidth=2)  # MODIFIED
    ax.semilogy(t, abp, "--", label="Run B: alpha-bar raised to 2.00", linewidth=2)  # MODIFIED
    ax.set(  # MODIFIED
        xlabel="Diffusion timestep (1-based)",  # MODIFIED
        ylabel="Cumulative signal retention (log)",  # MODIFIED
        title="Original vs powered training schedule",  # MODIFIED
    )  # MODIFIED
    ax.grid(True, alpha=0.25)  # MODIFIED
    ax.legend()  # MODIFIED
    fig.tight_layout()  # MODIFIED
    fig.savefig(outdir / "schedule_comparison.png", dpi=180)  # MODIFIED
    plt.close(fig)  # MODIFIED
    fig, ax = plt.subplots(figsize=(9, 5))  # ADDED
    ax.plot(t, np.log(ab / (1 - ab)), label="Run A")  # MODIFIED
    ax.plot(t, np.log(abp / (1 - abp)), label="Run B")  # MODIFIED
    ax.set(  # MODIFIED
        xlabel="Diffusion timestep (1-based)",  # MODIFIED
        ylabel="Natural log SNR",  # MODIFIED
        title="SNR change from schedule power",  # MODIFIED
    )  # MODIFIED
    ax.grid(True, alpha=0.25)  # MODIFIED
    ax.legend()  # MODIFIED
    fig.tight_layout()  # MODIFIED
    fig.savefig(outdir / "log_snr_comparison.png", dpi=180)  # MODIFIED
    plt.close(fig)  # MODIFIED
    print(f"Schedule audited: T=1000, alpha_bar_T={ab[-1]:.8f}, powered={abp[-1]:.8f}")  # MODIFIED


def pair_inventory(root):  # ADDED
    """Find matching summer SAR and optical patches."""  # ADDED
    root = Path(root)  # ADDED
    s1_files = sorted(root.rglob("ROIs1868_summer_s1_*_p*.tif"))  # MODIFIED
    if not s1_files:  # ADDED
        raise FileNotFoundError(  # MODIFIED
            f"No extracted summer S1 TIFFs under {root}. Extract both SEN12MS archives first."  # MODIFIED
        )  # MODIFIED
    records, missing = [], []  # ADDED
    for p in s1_files:  # ADDED
        m = re.fullmatch(r"ROIs1868_summer_s1_(\d+)_p(\d+)\.tif", p.name)  # MODIFIED
        if not m:  # ADDED
            continue  # ADDED
        roi, patch = m.groups()  # ADDED
        s2name = p.name.replace("_s1_", "_s2_")  # MODIFIED
        # ADDED: Match s1_<ROI> to s2_<ROI>.
        s2dir = p.parent.name.replace("s1_", "s2_")  # MODIFIED
        q = p.parent.parent / s2dir / s2name  # ADDED
        if not q.exists():  # ADDED
            # ADDED: Also support flattened folders.
            q = p.parent / s2name  # ADDED
        if not q.exists():  # ADDED
            missing.append(p.name)  # MODIFIED
            continue  # MODIFIED
        records.append(  # MODIFIED
            {  # MODIFIED
                "id": f"1868_summer_{roi}_p{patch}",  # MODIFIED
                "roi": roi,  # MODIFIED
                "s1": str(p.resolve()),  # MODIFIED
                "s2": str(q.resolve()),  # MODIFIED
            }  # MODIFIED
        )  # MODIFIED
    if missing:  # ADDED
        raise RuntimeError(  # MODIFIED
            f"{len(missing)} missing S2 matches; examples: {missing[:5]}"  # MODIFIED
        )  # MODIFIED
    if not records:  # ADDED
        raise RuntimeError("No valid SEN12MS S1/S2 pairs found")  # MODIFIED
    if len({r["id"] for r in records}) != len(records):  # MODIFIED
        raise RuntimeError("Duplicate patch identifiers found")  # MODIFIED
    return records  # ADDED


def validate_pair(row):  # ADDED
    import rasterio  # ADDED

    with rasterio.open(row["s1"]) as s1, rasterio.open(row["s2"]) as s2:  # MODIFIED
        if s1.count != 2 or s2.count != 13:  # MODIFIED
            raise ValueError(f"Wrong band counts: {row['id']}")  # MODIFIED
        if (s1.width, s1.height, s2.width, s2.height) != (256, 256, 256, 256):  # MODIFIED
            raise ValueError(f"Wrong dimensions: {row['id']}")  # MODIFIED
        if s1.crs is None or s2.crs is None or s1.crs != s2.crs:  # MODIFIED
            raise ValueError(f"Missing or different CRS: {row['id']}")  # MODIFIED
        if not np.allclose(tuple(s1.transform), tuple(s2.transform), rtol=0, atol=1e-6):  # MODIFIED
            raise ValueError(f"Pixel-grid mismatch: {row['id']}")  # MODIFIED
        if not np.all(s1.read_masks() == 255) or not np.all(s2.read_masks() == 255):  # MODIFIED
            raise ValueError(f"Invalid/nodata pixels: {row['id']}")  # MODIFIED
        a = s1.read()  # MODIFIED
        b = s2.read()  # MODIFIED
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):  # ADDED
            raise ValueError(f"NaN/Inf in {row['id']}")  # MODIFIED
        # ADDED: Check CRS when both files provide it.
        if s1.crs and s2.crs and s1.crs != s2.crs:  # ADDED
            raise ValueError(f"CRS mismatch: {row['id']}")  # MODIFIED
        return a, b  # ADDED


def group_splits(records, train_n=TRAIN_N, val_n=VAL_N, test_n=TEST_N, seed=SEED):  # ADDED
    """Allocate entire ROI groups to disjoint splits, then select exact patch counts."""  # ADDED
    groups = defaultdict(list)  # ADDED
    for r in records:  # ADDED
        groups[r["roi"]].append(r)  # MODIFIED
    keys = sorted(groups)  # ADDED
    rng = np.random.default_rng(seed)  # ADDED
    rng.shuffle(keys)  # ADDED
    targets = [("test", test_n), ("val", val_n), ("train", train_n)]  # MODIFIED
    cursor, result = 0, {}  # ADDED
    for split, target in targets:  # ADDED
        available = []  # ADDED
        while len(available) < target and cursor < len(keys):  # ADDED
            roi = keys[cursor]  # MODIFIED
            cursor += 1  # MODIFIED
            available += groups[roi]  # ADDED
        if len(available) < target:  # ADDED
            raise ValueError(  # MODIFIED
                f"Not enough ROI-separated pairs for {split}: need {target}; have {len(available)}"  # MODIFIED
            )  # MODIFIED
        rng.shuffle(available)  # ADDED
        result[split] = available[:target]  # ADDED
    assert (  # MODIFIED
        len(  # MODIFIED
            set(r["roi"] for r in result["train"])  # MODIFIED
            & set(r["roi"] for r in result["test"])  # MODIFIED
        )  # MODIFIED
        == 0  # MODIFIED
    )  # MODIFIED
    assert (  # MODIFIED
        len(  # MODIFIED
            set(r["roi"] for r in result["train"])  # MODIFIED
            & set(r["roi"] for r in result["val"])  # MODIFIED
        )  # MODIFIED
        == 0  # MODIFIED
    )  # MODIFIED
    assert (  # MODIFIED
        len(  # MODIFIED
            set(r["roi"] for r in result["val"]) & set(r["roi"] for r in result["test"])  # MODIFIED
        )  # MODIFIED
        == 0  # MODIFIED
    )  # MODIFIED
    return result  # ADDED


def write_manifest(splits, outdir):  # ADDED
    outdir = Path(outdir)  # MODIFIED
    outdir.mkdir(parents=True, exist_ok=True)  # MODIFIED
    for split, records in splits.items():  # ADDED
        with (outdir / f"{split}.jsonl").open("w") as f:  # MODIFIED
            for row in sorted(records, key=lambda r: r["id"]):  # MODIFIED
                f.write(json.dumps(dict(row, split=split)) + "\n")  # MODIFIED
    with (outdir / "split_summary.json").open("w") as f:  # MODIFIED
        json.dump(  # MODIFIED
            {  # MODIFIED
                s: {  # MODIFIED
                    "patches": len(v),  # MODIFIED
                    "rois": len(set(r["roi"] for r in v)),  # MODIFIED
                    "sha256": file_hash(outdir / f"{s}.jsonl"),  # MODIFIED
                }  # MODIFIED
                for s, v in splits.items()  # MODIFIED
            },  # MODIFIED
            f,  # MODIFIED
            indent=2,  # MODIFIED
        )  # MODIFIED


def file_hash(path):  # MODIFIED
    digest = hashlib.sha256()  # MODIFIED
    with Path(path).open("rb") as stream:  # MODIFIED
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):  # MODIFIED
            digest.update(chunk)  # MODIFIED
    return digest.hexdigest()  # MODIFIED


def checked_manifest(path, split):  # MODIFIED
    path = Path(path)  # MODIFIED
    rows = load_jsonl(path)  # MODIFIED
    summary = json.loads((path.parent / "split_summary.json").read_text())  # MODIFIED
    if path.stem != split or not rows or any(r.get("split") != split for r in rows):  # MODIFIED
        raise ValueError(f"Expected nonempty {split} manifest")  # MODIFIED
    if summary[split]["sha256"] != file_hash(path) or summary[split]["patches"] != len(  # MODIFIED
        rows  # MODIFIED
    ):  # MODIFIED
        raise ValueError("Manifest differs from the recorded split")  # MODIFIED
    return rows  # MODIFIED


def load_jsonl(path):  # ADDED
    with open(path) as f:  # MODIFIED
        return [json.loads(line) for line in f if line.strip()]  # MODIFIED


def array_channels(s1, s2):  # ADDED
    # ADDED: Use the SEN12MS band order.
    d = {"SAR_VV": s1[0].astype(np.float32), "SAR_VH": s1[1].astype(np.float32)}  # MODIFIED
    d["SAR_DIFF"] = d["SAR_VV"] - d["SAR_VH"]  # MODIFIED
    for b in STAT_CHANNELS[3:]:  # ADDED
        d[b] = s2[S2_BANDS.index(b)].astype(np.float32)  # ADDED
    return d  # ADDED


def fit_stats(  # MODIFIED
    train_manifest, outpath, max_patches=None, pixels_per_patch=128, seed=SEED  # MODIFIED
):  # MODIFIED
    """Estimate clipping quantiles from sampled training pixels."""  # ADDED
    rows = checked_manifest(train_manifest, "train")  # MODIFIED
    if pixels_per_patch < 1 or (max_patches is not None and max_patches < 1):  # MODIFIED
        raise ValueError("Pixel and patch counts must be positive")  # MODIFIED
    if max_patches is not None:  # MODIFIED
        rows = rows[:max_patches]  # MODIFIED
    rng = np.random.default_rng(seed)  # MODIFIED
    acc = {k: [] for k in STAT_CHANNELS}  # MODIFIED
    for i, r in enumerate(rows):  # MODIFIED
        s1, s2 = validate_pair(r)  # MODIFIED
        bands = array_channels(s1, s2)  # MODIFIED
        idx = rng.choice(256 * 256, min(pixels_per_patch, 256 * 256), replace=False)  # MODIFIED
        for k in STAT_CHANNELS:  # ADDED
            acc[k].append(bands[k].reshape(-1)[idx])  # ADDED
        if (i + 1) % 1000 == 0:  # MODIFIED
            print("Stats sampled patches:", i + 1)  # MODIFIED
    if not rows:  # MODIFIED
        raise ValueError("Empty train manifest")  # MODIFIED
    stats = {  # MODIFIED
        k: [  # MODIFIED
            float(np.percentile(np.concatenate(acc[k]), 2)),  # MODIFIED
            float(np.percentile(np.concatenate(acc[k]), 98)),  # MODIFIED
        ]  # MODIFIED
        for k in STAT_CHANNELS  # MODIFIED
    }  # MODIFIED
    if any(not hi > lo for lo, hi in stats.values()):  # MODIFIED
        raise ValueError("Degenerate clipping quantiles")  # MODIFIED
    path = Path(outpath)  # MODIFIED
    path.parent.mkdir(parents=True, exist_ok=True)  # MODIFIED
    path.write_text(  # MODIFIED
        json.dumps(  # MODIFIED
            {  # MODIFIED
                "source_manifest": str(Path(train_manifest).resolve()),  # MODIFIED
                "source_sha256": file_hash(train_manifest),  # MODIFIED
                "split": "train",  # MODIFIED
                "sampled_patches": len(rows),  # MODIFIED
                "pixels_per_patch": pixels_per_patch,  # MODIFIED
                "seed": seed,  # MODIFIED
                "stats": stats,  # MODIFIED
                "warning": "Sampled pixel quantiles (reproducible estimate); not exact all-pixel quantiles",  # MODIFIED
            },  # MODIFIED
            indent=2,  # MODIFIED
        )  # MODIFIED
    )  # MODIFIED
    return stats  # ADDED


def render_rgb(channels, names, stats):  # ADDED
    from PIL import Image  # ADDED

    rgb = []  # MODIFIED
    for n in names:  # ADDED
        lo, hi = stats[n]  # MODIFIED
        rgb.append(np.clip((channels[n] - lo) / (hi - lo), 0, 1))  # MODIFIED
    a = (np.stack(rgb, axis=-1) * 255).round().astype(np.uint8)  # MODIFIED
    return Image.fromarray(a, mode="RGB")  # MODIFIED


def render_examples(manifest, stats_file, outdir, max_patches=3, seed=SEED):  # MODIFIED
    from PIL import Image  # ADDED

    stats_record = json.loads(Path(stats_file).read_text())  # MODIFIED
    source = Path(stats_record["source_manifest"])  # MODIFIED
    checked_manifest(source, "train")  # MODIFIED
    if (  # MODIFIED
        stats_record.get("split") != "train"  # MODIFIED
        or file_hash(source) != stats_record["source_sha256"]  # MODIFIED
    ):  # MODIFIED
        raise ValueError("Statistics do not match their training manifest")  # MODIFIED
    stats = stats_record["stats"]  # MODIFIED
    split = Path(manifest).stem  # MODIFIED
    if split not in ("train", "val", "test"):  # MODIFIED
        raise ValueError(  # MODIFIED
            "Manifest filename must be train.jsonl, val.jsonl or test.jsonl"  # MODIFIED
        )  # MODIFIED
    rows = checked_manifest(manifest, split)  # MODIFIED
    outdir = Path(outdir)  # MODIFIED
    outdir.mkdir(parents=True, exist_ok=True)  # MODIFIED
    output = []  # MODIFIED
    rng = np.random.default_rng(seed)  # MODIFIED
    product_order = list(PRODUCTS)  # MODIFIED
    allocation = rng.permutation(len(rows)) % len(  # MODIFIED
        product_order  # MODIFIED
    )  # MODIFIED: Balance the three products.
    template_indices = rng.integers(0, 8, size=len(rows))  # MODIFIED
    for index, r in enumerate(rows[:max_patches]):  # MODIFIED
        s1, s2 = validate_pair(r)  # MODIFIED
        channels = array_channels(s1, s2)  # MODIFIED
        img = render_rgb(channels, ["SAR_VV", "SAR_VH", "SAR_DIFF"], stats)  # MODIFIED
        inp = outdir / f"{r['id']}_SAR.png"  # MODIFIED
        img.save(inp)  # MODIFIED
        names = (  # MODIFIED
            product_order  # MODIFIED
            if split == "test"  # MODIFIED
            else [product_order[int(allocation[index])]]  # MODIFIED
        )  # MODIFIED
        for j, name in enumerate(names):  # MODIFIED
            bands = PRODUCTS[name]  # MODIFIED
            target = render_rgb(channels, bands, stats)  # MODIFIED
            targetpath = outdir / f"{r['id']}_{name}_GT.png"  # MODIFIED
            target.save(targetpath)  # MODIFIED
            if split == "test":  # MODIFIED
                # MODIFIED: Reserve templates 8–11 for testing.
                t_idx = 8 + int(rng.integers(0, 4))  # MODIFIED
            else:  # MODIFIED
                t_idx = int(template_indices[index])  # MODIFIED
            output.append(  # MODIFIED
                {  # MODIFIED
                    "id": r["id"],  # MODIFIED
                    "roi": r["roi"],  # MODIFIED
                    "split": split,  # MODIFIED
                    "product": name,  # MODIFIED
                    "stats_sha256": file_hash(stats_file),  # MODIFIED
                    "manifest_sha256": file_hash(manifest),  # MODIFIED
                    "stats_sampled_patches": stats_record["sampled_patches"],  # MODIFIED
                    "image_sha256": file_hash(inp),  # MODIFIED
                    "target_sha256": file_hash(targetpath),  # MODIFIED
                    "template_index": t_idx,  # MODIFIED
                    "prompt": TEMPLATES[name][t_idx],  # MODIFIED
                    "image": str(inp.resolve()),  # MODIFIED
                    "target": str(targetpath.resolve()),  # MODIFIED
                }  # MODIFIED
            )  # MODIFIED
    target_path = outdir / "examples.jsonl"  # MODIFIED
    with target_path.open("w") as f:  # MODIFIED
        for r in output:  # MODIFIED
            f.write(json.dumps(r) + "\n")  # MODIFIED
    print(  # MODIFIED
        "Rendered",  # MODIFIED
        len(output),  # MODIFIED
        "paired image/prompt examples from",  # MODIFIED
        len(rows[:max_patches]),  # MODIFIED
        "patches:",  # MODIFIED
        target_path,  # MODIFIED
    )  # MODIFIED
    return target_path  # ADDED
