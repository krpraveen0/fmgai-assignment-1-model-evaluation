# ADDED: Tests use synthetic GeoTIFFs.
import json  # ADDED
import numpy as np  # ADDED
import pytest  # ADDED
from core import (  # ADDED
    native_schedule,  # ADDED
    power_schedule,  # ADDED
    group_splits,  # ADDED
    pair_inventory,  # ADDED
    write_manifest,  # ADDED
    render_examples,  # ADDED
    fit_stats,  # ADDED
    load_jsonl,  # ADDED
)  # ADDED


def test_schedule_range_power_and_terminal():  # ADDED
    beta, alpha, ab = native_schedule()  # ADDED
    bp, abp = power_schedule()  # ADDED
    assert len(beta) == len(bp) == 1000  # ADDED
    assert np.all(beta > 0) and np.all(beta < 1)  # ADDED
    assert np.all(ab > 0) and np.all(ab < 1)  # ADDED
    np.testing.assert_allclose(abp, ab**2, rtol=1e-11, atol=1e-15)  # ADDED
    np.testing.assert_allclose(bp, 1 - (1 - beta) ** 2, rtol=1e-13)  # ADDED
    assert 0.004 < ab[-1] < 0.005  # ADDED
    assert abp[-1] < ab[-1]  # ADDED


def test_exact_group_separation():  # ADDED
    data = []  # ADDED
    for roi in range(16):  # ADDED
        for p in range(6):  # ADDED
            data.append({"roi": str(roi), "id": f"{roi}_p{p}"})  # ADDED
    d = group_splits(data, train_n=24, val_n=12, test_n=12)  # ADDED
    assert {s: len(r) for s, r in d.items()} == {"test": 12, "val": 12, "train": 24}  # ADDED
    for a, b in [("train", "test"), ("train", "val"), ("val", "test")]:  # ADDED
        assert not {r["roi"] for r in d[a]} & {r["roi"] for r in d[b]}  # ADDED


def fixture_geotiffs(tmp_path):  # ADDED
    import rasterio  # ADDED
    from rasterio.transform import from_origin  # ADDED

    base = tmp_path / "ROIs1868_summer"  # ADDED
    for roi in range(6):  # ADDED
        for patch in range(2):  # ADDED
            arr1 = np.stack(  # ADDED
                [  # ADDED
                    np.full((256, 256), -10 - roi, dtype=np.float32),  # ADDED
                    np.full((256, 256), -18 - patch, dtype=np.float32),  # ADDED
                ]  # ADDED
            )  # ADDED
            # ADDED: Avoid constant-value quantiles.
            yy, xx = np.indices((256, 256))  # ADDED
            grad = (xx + yy).astype(np.float32)  # ADDED
            arr1 += grad / 1000  # ADDED
            arr2 = np.stack([grad + (i + 1) * 10 for i in range(13)]).astype(np.float32)  # ADDED
            for sensor, a in [("s1", arr1), ("s2", arr2)]:  # ADDED
                directory = base / f"{sensor}_{roi}"  # ADDED
                directory.mkdir(parents=True, exist_ok=True)  # ADDED
                filename = directory / f"ROIs1868_summer_{sensor}_{roi}_p{patch}.tif"  # ADDED
                with rasterio.open(  # ADDED
                    filename,  # ADDED
                    "w",  # ADDED
                    driver="GTiff",  # ADDED
                    width=256,  # ADDED
                    height=256,  # ADDED
                    count=a.shape[0],  # ADDED
                    dtype="float32",  # ADDED
                    crs="EPSG:4326",  # ADDED
                    transform=from_origin(0, 1, 1, 1),  # ADDED
                ) as ds:  # ADDED
                    ds.write(a)  # ADDED
    return tmp_path  # ADDED


def test_pair_stats_render_and_instructions(tmp_path):  # ADDED
    root = fixture_geotiffs(tmp_path)  # ADDED
    pairs = pair_inventory(root)  # ADDED
    assert len(pairs) == 12  # ADDED
    assert len(set(r["roi"] for r in pairs)) == 6  # ADDED
    d = group_splits(pairs, train_n=4, val_n=2, test_n=2)  # ADDED
    manifest = tmp_path / "manifests"  # ADDED
    write_manifest(d, manifest)  # ADDED
    stats_path = tmp_path / "stats.json"  # ADDED
    fit_stats(manifest / "train.jsonl", stats_path, max_patches=2, pixels_per_patch=256)  # ADDED
    assert json.loads(stats_path.read_text())["sampled_patches"] == 2  # ADDED
    p = render_examples(  # ADDED
        manifest / "train.jsonl", stats_path, tmp_path / "renders_train", max_patches=2  # ADDED
    )  # ADDED
    train = load_jsonl(p)  # ADDED
    assert len(train) == 2 and all(  # ADDED
        r["template_index"] < 8 and r["split"] == "train" for r in train  # ADDED
    )  # ADDED
    t = render_examples(  # ADDED
        manifest / "test.jsonl", stats_path, tmp_path / "renders_test", max_patches=1  # ADDED
    )  # ADDED
    test = load_jsonl(t)  # ADDED
    assert len(test) == 3 and all(8 <= r["template_index"] < 12 for r in test)  # ADDED
    assert {x["product"] for x in test} == {  # ADDED
        "true_color",  # ADDED
        "false_color_ir",  # ADDED
        "shortwave_ir",  # ADDED
    }  # ADDED
