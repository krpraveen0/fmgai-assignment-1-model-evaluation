# ADDED
"""SEN12MS preparation and noise-schedule checks."""
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 20261004  # ADDED
NOISE_POWER = 2.00  # ADDED
MODEL_ID = 'timbrooks/instruct-pix2pix'  # ADDED
MODEL_REVISION = '31519b5cb02a7fd89b906d88731cd4d6a7bbf88d'  # ADDED
TRAIN_N, VAL_N, TEST_N = 10000, 1000, 2000  # ADDED
PRODUCTS = {  # ADDED
    'true_color': ('B04', 'B03', 'B02'),
    'false_color_ir': ('B08', 'B04', 'B03'),
    'shortwave_ir': ('B12', 'B8A', 'B04'),
}
# ADDED: Prompts name only the requested composite.
PROMPTS = {
    'true_color': 'Create the true-colour Sentinel-2 composite.',
    'false_color_ir': 'Create the false-colour infrared Sentinel-2 composite.',
    'shortwave_ir': 'Create the shortwave-infrared Sentinel-2 composite.',
}
# ADDED: Eight train/validation templates and four test templates.
TEMPLATES = {
 'true_color': [
  'Create the true-colour Sentinel-2 composite.', 'Produce the true-colour optical view.',
  'Render this SAR patch in true colour.', 'Show the matching true-colour Sentinel-2 image.',
  'Generate the RGB natural-colour composite.', 'Make a true-colour satellite rendering.',
  'Create the visible-band colour product.', 'Generate a true-colour view of this SAR image.',
  'Render the corresponding natural RGB product.', 'Produce the visible-light colour composite.',
  'Show this area using true-colour optical bands.', 'Convert this SAR input into true-colour Sentinel-2.'
 ],
 'false_color_ir': [
  'Create the false-colour infrared composite.', 'Produce a false-colour NIR view.',
  'Render this SAR patch as false-colour infrared.', 'Show the matching false-colour optical view.',
  'Generate the B8-B4-B3 composite.', 'Create the false-colour infrared satellite rendering.',
  'Generate the NIR-R-G band composite.', 'Produce the infrared false-colour product.',
  'Render the corresponding false-colour infrared image.', 'Show the false-colour NIR composite.',
  'Create the optical composite using NIR, red and green.', 'Convert this SAR input to false-colour infrared.'
 ],
 'shortwave_ir': [
  'Create the shortwave-infrared composite.', 'Produce a SWIR composite view.',
  'Render the SAR patch using the SWIR product.', 'Show the matching shortwave-infrared view.',
  'Generate the B12-B8A-B4 composite.', 'Create the shortwave-infrared satellite rendering.',
  'Generate the SWIR-NIR-red composite.', 'Produce the shortwave-infrared optical product.',
  'Render the corresponding shortwave-infrared image.', 'Show this patch as a SWIR composite.',
  'Create the optical product using B12, B8A and B4.', 'Convert this SAR input to shortwave infrared.'
 ]
}
assert all(len(v) == 12 for v in TEMPLATES.values())

S2_BANDS = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12')
STAT_CHANNELS = ('SAR_VV', 'SAR_VH', 'SAR_DIFF', 'B02', 'B03', 'B04', 'B08', 'B8A', 'B12')


def native_schedule():
    """Exact scaled_linear beta schedule in the declared 1000-step checkpoint."""
    beta = np.linspace(np.sqrt(.00085), np.sqrt(.012), 1000, dtype=np.float64) ** 2
    alpha = 1.0 - beta
    ab = np.cumprod(alpha)
    return beta, alpha, ab


def power_schedule():
    """Reconstruct beta' from declared theta'=alpha_bar**2.00."""
    beta, alpha, ab = native_schedule()
    ap = alpha ** NOISE_POWER
    bp = 1.0 - ap
    abp = np.cumprod(ap)
    np.testing.assert_allclose(abp, ab ** NOISE_POWER, rtol=1e-11, atol=1e-15)
    assert np.all((ab > 0) & (ab < 1))
    assert np.all((abp > 0) & (abp < 1))
    assert np.all((bp > 0) & (bp < 1))
    assert 1.5 < NOISE_POWER < 3
    return bp, abp


def save_schedule(outdir):
    import matplotlib.pyplot as plt
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    beta, alpha, ab = native_schedule(); bp, abp = power_schedule()
    rows = []
    for i in range(1000):
        sa = ab[i] / (1 - ab[i]); sb = abp[i] / (1 - abp[i])
        rows.append([i + 1, beta[i], bp[i], alpha[i], ab[i], abp[i], np.log(sa), np.log(sb)])
    header = ['t', 'beta_native', 'beta_powered', 'alpha_native', 'alpha_bar_native', 'alpha_bar_powered', 'log_snr_native', 'log_snr_powered']
    with (outdir / 'schedule_all_1000.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    with (outdir / 'schedule_five_steps.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows[i] for i in [0, 249, 499, 749, 999])
    t = np.arange(1, 1001)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(t, ab, label='Run A: native alpha-bar', linewidth=2)
    ax.semilogy(t, abp, '--', label='Run B: alpha-bar raised to 2.00', linewidth=2)
    ax.set(xlabel='Diffusion timestep (1-based)', ylabel='Cumulative signal retention (log)', title='Original vs powered training schedule')
    ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(outdir / 'schedule_comparison.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, np.log(ab / (1-ab)), label='Run A')
    ax.plot(t, np.log(abp / (1-abp)), label='Run B')
    ax.set(xlabel='Diffusion timestep (1-based)', ylabel='Natural log SNR', title='SNR change from schedule power')
    ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(outdir / 'log_snr_comparison.png', dpi=180); plt.close(fig)
    print(f'Schedule audited: T=1000, alpha_bar_T={ab[-1]:.8f}, powered={abp[-1]:.8f}')


def pair_inventory(root):
    """Find matching summer SAR and optical patches."""
    root = Path(root)
    s1_files = sorted(root.rglob('ROIs1868_summer_s1_*_p*.tif'))
    if not s1_files:
        raise FileNotFoundError(f'No extracted summer S1 TIFFs under {root}. Extract both SEN12MS archives first.')
    records, missing = [], []
    for p in s1_files:
        m = re.fullmatch(r'ROIs1868_summer_s1_(\d+)_p(\d+)\.tif', p.name)
        if not m:
            continue
        roi, patch = m.groups()
        s2name = p.name.replace('_s1_', '_s2_')
        # ADDED: Match s1_<ROI> to s2_<ROI>.
        s2dir = p.parent.name.replace('s1_', 's2_')
        q = p.parent.parent / s2dir / s2name
        if not q.exists():
            # ADDED: Also support flattened folders.
            q = p.parent / s2name
        if not q.exists():
            missing.append(p.name); continue
        records.append({'id': f'1868_summer_{roi}_p{patch}', 'roi': roi, 's1': str(p.resolve()), 's2': str(q.resolve())})
    if missing:
        raise RuntimeError(f'{len(missing)} missing S2 matches; examples: {missing[:5]}')
    if not records:
        raise RuntimeError('No valid SEN12MS S1/S2 pairs found')
    if len({r['id'] for r in records}) != len(records):
        raise RuntimeError('Duplicate patch identifiers found')
    return records


def validate_pair(row):
    import rasterio
    with rasterio.open(row['s1']) as s1, rasterio.open(row['s2']) as s2:
        assert s1.count == 2, f'Expected VV/VH two bands: {row["s1"]}'
        assert s2.count == 13, f'Expected Sentinel-2 13 bands: {row["s2"]}'
        assert (s1.width, s1.height, s2.width, s2.height) == (256, 256, 256, 256)
        a = s1.read(); b = s2.read()
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError(f'NaN/Inf in {row["id"]}')
        # ADDED: Check CRS when both files provide it.
        if s1.crs and s2.crs and s1.crs != s2.crs:
            raise ValueError(f'CRS mismatch: {row["id"]}')
        return a, b


def group_splits(records, train_n=TRAIN_N, val_n=VAL_N, test_n=TEST_N, seed=SEED):
    """Allocate entire ROI groups to disjoint splits, then select exact patch counts."""
    groups = defaultdict(list)
    for r in records:
        groups[r['roi']].append(r)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    targets = [('test', test_n), ('val', val_n), ('train', train_n)]
    cursor, result = 0, {}
    for split, target in targets:
        available = []
        while len(available) < target and cursor < len(keys):
            roi = keys[cursor]; cursor += 1
            available += groups[roi]
        if len(available) < target:
            raise ValueError(f'Not enough ROI-separated pairs for {split}: need {target}; have {len(available)}')
        rng.shuffle(available)
        result[split] = available[:target]
    assert len(set(r['roi'] for r in result['train']) & set(r['roi'] for r in result['test'])) == 0
    assert len(set(r['roi'] for r in result['train']) & set(r['roi'] for r in result['val'])) == 0
    assert len(set(r['roi'] for r in result['val']) & set(r['roi'] for r in result['test'])) == 0
    return result


def write_manifest(splits, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        with (outdir / f'{split}.jsonl').open('w') as f:
            for row in sorted(records, key=lambda r:r['id']):
                f.write(json.dumps(row) + '\n')
    with (outdir / 'split_summary.json').open('w') as f:
        json.dump({s: {'patches':len(v), 'rois':len(set(r['roi'] for r in v))} for s,v in splits.items()},f,indent=2)


def load_jsonl(path):
    with open(path) as f: return [json.loads(line) for line in f if line.strip()]


def array_channels(s1, s2):
    # ADDED: Use the SEN12MS band order.
    d = {'SAR_VV': s1[0].astype(np.float32), 'SAR_VH': s1[1].astype(np.float32)}
    d['SAR_DIFF'] = d['SAR_VV'] - d['SAR_VH']
    for b in STAT_CHANNELS[3:]:
        d[b] = s2[S2_BANDS.index(b)].astype(np.float32)
    return d


def fit_stats(train_manifest, outpath, max_patches=None, pixels_per_patch=128):
    """Estimate clipping quantiles from sampled training pixels."""
    rows = load_jsonl(train_manifest)
    if max_patches is not None: rows = rows[:max_patches]
    rng = np.random.default_rng(SEED)
    acc = {k:[] for k in STAT_CHANNELS}
    for i,r in enumerate(rows):
        s1,s2 = validate_pair(r)
        bands = array_channels(s1,s2)
        idx = rng.choice(256*256, min(pixels_per_patch, 256*256), replace=False)
        for k in STAT_CHANNELS:
            acc[k].append(bands[k].reshape(-1)[idx])
        if (i+1) % 1000 == 0: print('Stats sampled patches:',i+1)
    if not rows: raise ValueError('Empty train manifest')
    stats = {k: [float(np.percentile(np.concatenate(acc[k]),2)),float(np.percentile(np.concatenate(acc[k]),98))] for k in STAT_CHANNELS}
    if any(not hi > lo for lo,hi in stats.values()): raise ValueError('Degenerate clipping quantiles')
    path = Path(outpath); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'source_manifest':str(train_manifest), 'sampled_patches':len(rows),
                                'pixels_per_patch':pixels_per_patch,'seed':SEED,'stats':stats,
                                'warning':'Sampled pixel quantiles (reproducible estimate); not exact all-pixel quantiles'}, indent=2))
    return stats


def render_rgb(channels, names, stats):
    from PIL import Image
    rgb=[]
    for n in names:
        lo,hi = stats[n]
        rgb.append(np.clip((channels[n]-lo)/(hi-lo),0,1))
    a=(np.stack(rgb,axis=-1)*255).round().astype(np.uint8)
    return Image.fromarray(a,mode='RGB')


def render_examples(manifest, stats_file, outdir, max_patches=3):
    from PIL import Image
    stats = json.loads(Path(stats_file).read_text())['stats']
    rows = load_jsonl(manifest)
    split=Path(manifest).stem
    if split not in ('train','val','test'):
        raise ValueError('Manifest filename must be train.jsonl, val.jsonl or test.jsonl')
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    output=[]
    rng=np.random.default_rng(SEED)
    product_order=list(PRODUCTS)
    allocation=rng.permutation(len(rows)) % len(product_order)  # ADDED: Balance the three products.
    template_indices=rng.integers(0,8,size=len(rows))
    for index,r in enumerate(rows[:max_patches]):
        s1,s2 = validate_pair(r)
        channels = array_channels(s1,s2)
        img = render_rgb(channels, ['SAR_VV','SAR_VH','SAR_DIFF'], stats)
        inp = outdir / f'{r["id"]}_SAR.png'; img.save(inp)
        names = product_order if split=='test' else [product_order[int(allocation[index])]]
        for j,name in enumerate(names):
            bands=PRODUCTS[name]
            target = render_rgb(channels,bands,stats)
            targetpath=outdir/f'{r["id"]}_{name}_GT.png';target.save(targetpath)
            if split=='test':
                # ADDED: Reserve templates 8–11 for testing.
                t_idx=8+int(rng.integers(0,4))
            else: t_idx=int(template_indices[index])
            output.append({'id':r['id'],'roi':r['roi'],'split':split,'product':name,
                          'template_index':t_idx,'prompt':TEMPLATES[name][t_idx],
                          'image':str(inp.resolve()),'target':str(targetpath.resolve())})
    target_path = outdir/'examples.jsonl'
    with target_path.open('w') as f:
        for r in output: f.write(json.dumps(r)+'\n')
    print('Rendered',len(output),'paired image/prompt examples from',len(rows[:max_patches]),'patches:',target_path)
    return target_path
