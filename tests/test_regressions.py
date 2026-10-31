import json  # ADDED
from types import SimpleNamespace  # ADDED
import numpy as np  # ADDED
import pytest  # ADDED
import rasterio  # ADDED
from rasterio.transform import from_origin  # ADDED
from core import (  # ADDED
    pair_inventory,  # ADDED
    validate_pair,  # ADDED
    fit_stats,  # ADDED
    group_splits,  # ADDED
    write_manifest,  # ADDED
    file_hash,  # ADDED
    render_examples,  # ADDED
)  # ADDED
from config import load_config  # ADDED
from experiment import capture_indices, scheduler_for  # ADDED
from test_core import fixture_geotiffs  # ADDED


def test_reject_shifted_grid_and_nodata(tmp_path):  # ADDED
    fixture_geotiffs(tmp_path)  # ADDED
    rows = pair_inventory(tmp_path)  # ADDED
    with rasterio.open(rows[0]["s2"], "r+") as ds:  # ADDED
        ds.transform = from_origin(100, 100, 1, 1)  # ADDED
    with pytest.raises(ValueError, match="Pixel-grid"):  # ADDED
        validate_pair(rows[0])  # ADDED
    with rasterio.open(rows[1]["s2"], "r+") as ds:  # ADDED
        data = ds.read()  # ADDED
        data[:, 0, 0] = -9999  # ADDED
        ds.nodata = -9999  # ADDED
        ds.write(data)  # ADDED
    with pytest.raises(ValueError, match="nodata"):  # ADDED
        validate_pair(rows[1])  # ADDED


def test_stats_reject_test_and_tampered_manifest(tmp_path):  # ADDED
    fixture_geotiffs(tmp_path)  # ADDED
    rows = pair_inventory(tmp_path)  # ADDED
    write_manifest(group_splits(rows, 4, 2, 2), tmp_path / "manifests")  # ADDED
    with pytest.raises(ValueError):  # ADDED
        fit_stats(tmp_path / "manifests/test.jsonl", tmp_path / "stats.json")  # ADDED
    path = tmp_path / "manifests/train.jsonl"  # ADDED
    path.write_text(path.read_text().replace("1868_summer", "changed", 1))  # ADDED
    with pytest.raises(ValueError, match="differs"):  # ADDED
        fit_stats(path, tmp_path / "stats.json")  # ADDED


def test_invalid_initial_pair_is_logged_during_splitting(tmp_path, monkeypatch):  # ADDED
    import prepare as runner  # ADDED

    c = load_config()  # ADDED
    c.update(train_patches=2, val_patches=1, test_patches=1)  # ADDED
    monkeypatch.setattr(runner, "load_config", lambda _: c)  # ADDED
    rows = [{"id": str(i), "roi": str(i)} for i in range(5)]  # ADDED
    monkeypatch.setattr(runner, "pair_inventory", lambda _: rows)  # ADDED

    def validate(row):  # ADDED
        if row["id"] == "0":  # ADDED
            raise ValueError("invalid")  # ADDED

    monkeypatch.setattr(runner, "validate_pair", validate)  # ADDED
    runner.validate_dataset(  # ADDED
        SimpleNamespace(  # ADDED
            config="unused",  # ADDED
            data_root="unused",  # ADDED
            make_splits=True,  # ADDED
            check_pairs=12,  # ADDED
            outdir=tmp_path,  # ADDED
        )  # ADDED
    )  # ADDED
    assert json.loads((tmp_path / "excluded_invalid.json").read_text())[0]["id"] == "0"  # ADDED
    assert (  # ADDED
        sum(  # ADDED
            json.loads(line)["id"] == "0"  # ADDED
            for p in tmp_path.glob("*.jsonl")  # ADDED
            for line in p.read_text().splitlines()  # ADDED
        )  # ADDED
        == 0  # ADDED
    )  # ADDED


def test_configuration_changes_actual_scheduler(tmp_path):  # ADDED
    pytest.importorskip("torch")  # ADDED
    c = load_config()  # ADDED
    a, b = scheduler_for("A", c), scheduler_for("B", c)  # ADDED
    np.testing.assert_allclose(  # ADDED
        b.alphas_cumprod.numpy(), a.alphas_cumprod.numpy() ** 2, rtol=2e-5, atol=1e-7  # ADDED
    )  # ADDED
    with pytest.raises(ValueError):  # ADDED
        scheduler_for("typo", c)  # ADDED
    c["checkpoints_epochs"] = [1, 2, 3, 4, 20]  # ADDED
    path = tmp_path / "bad.json"  # ADDED
    path.write_text(json.dumps(c))  # ADDED
    with pytest.raises(ValueError, match="equally spaced"):  # ADDED
        load_config(path)  # ADDED


def test_trajectory_selection_uses_timestep_values():  # ADDED
    times = np.array([999, 850, 700, 550, 400, 250, 100, 0])  # ADDED
    selected = capture_indices(times)  # ADDED
    assert selected == [0, 2, 3, 5, 7]  # ADDED
