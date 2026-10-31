# Noise schedule ablation in diffusion models

Local experiment using InstructPix2Pix and SEN12MS summer. Run 0, both 20-epoch training runs, all 18,000 final predictions, checkpoint samples, trajectories and test metrics are complete. Fill the responsibility and signature fields on the final page of `report/report.docx` before handing in.

## Contents

- `src/`: the executed implementation, with shorter module names and matching imports.
- `data/manifests/`, `data/statistics.json`: the recorded ROI splits and training-only normalization statistics.
- `results/runs/A`, `results/runs/B`: training loss CSVs, validation, schedules, configurations and five sets of checkpoint samples.
- `results/inference/0`, `A`, `B`: 6,000 labelled test predictions per run and their metadata.
- `results/trajectories/A`, `B`: actual noisy latent tensors, decoded previews and timestep labels.
- `results/metrics/`: LPIPS, PSNR, SSIM, per-image scores and loss curves.
- `report/report.docx`: the report in the professor's nine-section order, with responsibility and signature spaces.
- `provenance/`: environment, dataset checksums, schedule verification, execution metadata and original-versus-modified code.
- `tests/`: the existing synthetic workflow and regression checks. These tests are not experiment results.

Colab notebooks, Docker files, legacy starters, packaging utilities and unrelated intermediate tests were excluded. The original working project is unchanged. Raw archives, extracted TIFFs, cached pretrained weights and large trained model/optimizer weights are excluded. The commands below regenerate them.

## Local environment

The experiment ran with Python 3.12.11 and Apple MPS. Exact direct dependencies are in `requirements.txt`; the complete historical environment is recorded in `provenance/environment.txt`.

```bash
cd Submission
uv venv --python 3.12.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
bash run.sh -m pytest -q tests
```

Run commands from this directory. `run.sh` uses this folder's `.venv` and caches. On the original Mac, the existing results remain in the original working folder; this is the cleaned submission copy.

## Download and preparation

```bash
bash run.sh src/download.py --root data --extract
bash run.sh src/prepare.py checkpoint --outdir provenance/checkpoint
bash run.sh src/prepare.py schedule --outdir provenance/schedule
bash run.sh src/prepare.py dataset --data-root data --make-splits --outdir data/manifests
bash run.sh src/prepare.py stats --train-manifest data/manifests/train.jsonl --out data/statistics.json
bash run.sh src/prepare.py render --manifest data/manifests/train.jsonl --stats data/statistics.json --outdir data/prepared/train --max-patches 10000
bash run.sh src/prepare.py render --manifest data/manifests/val.jsonl --stats data/statistics.json --outdir data/prepared/val --max-patches 1000
bash run.sh src/prepare.py render --manifest data/manifests/test.jsonl --stats data/statistics.json --outdir data/prepared/test --max-patches 2000
```

These commands recreate preparation artifacts. Recorded JSON files preserve the original execution paths and hashes; do not manually rewrite those files and treat them as a new execution. The original `outputs/experiment/` tree corresponds to the submitted `results/` tree. Re-running preparation generates paths valid on the current machine.

## Reproduce the three runs

Use a fresh output directory for reproduction so the submitted results remain unchanged. Seed 20261004, exponent 2.00, BF16 training, 20 epochs, AdamW 5e-5, effective batch 64 and all other settings are fixed in `experiment.json`. A and B differ only in training noise schedule. Run sequentially with the Mac powered and lid open.

```bash
caffeinate -i bash run.sh src/experiment.py infer --run 0 --device mps --examples data/prepared --output reproduced
caffeinate -i bash run.sh src/experiment.py train --run A --device mps --examples data/prepared --output reproduced
caffeinate -i bash run.sh src/experiment.py train --run B --device mps --examples data/prepared --output reproduced
caffeinate -i bash run.sh src/experiment.py infer --run A --device mps --examples data/prepared --output reproduced
caffeinate -i bash run.sh src/experiment.py infer --run B --device mps --examples data/prepared --output reproduced
caffeinate -i bash run.sh src/experiment.py evaluate --examples data/prepared --output reproduced
```

Training checkpoints can be resumed with `--resume reproduced/runs/A/checkpoints/epoch_004`. Full-model training needs sufficient memory and disk space. On another supported native backend replace `--device mps` consistently for all runs. Docker was not used for the reported experiment.

## Results

| Run | LPIPS lower is better | PSNR higher is better | SSIM higher is better |
|---|---:|---:|---:|
| Pretrained 0 | 0.808778 | 8.449295 | 0.059701 |
| Original schedule A | 0.570881 | 12.436583 | 0.238348 |
| Powered schedule B | 0.560575 | 13.640791 | 0.291870 |

All scores include black outputs as saved. The report explains this limitation, the first visible checkpoint differences and the mathematical interpretation. No results have been replaced during cleanup.
