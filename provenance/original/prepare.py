# ADDED
"""Hardware checks, Run 0 inference, and A/B training diagnostics."""
import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from phase2_core import (SEED, NOISE_POWER as FROZEN_POWER, MODEL_ID, MODEL_REVISION, native_schedule,
                         power_schedule, pair_inventory, validate_pair, group_splits,
                         write_manifest, load_jsonl, save_schedule, fit_stats, render_examples)


NOISE_POWER = 2.00  # ADDED
assert NOISE_POWER == FROZEN_POWER  # ADDED

def gpu_report(outdir):
    import torch
    d=Path(outdir);d.mkdir(parents=True, exist_ok=True)
    gpu=torch.cuda.is_available()
    mps = torch.backends.mps.is_available()  # ADDED
    info={'gpu_available':gpu,'pytorch':torch.__version__, 'torch_cuda':torch.version.cuda,
          'mps_available':mps, 'selected_device':'cuda' if gpu else ('mps' if mps else 'cpu'),  # ADDED
          'cuda_bf16_supported':bool(torch.cuda.is_bf16_supported()) if gpu else False,
          'colab_disk_free_gib':round(shutil.disk_usage('/content' if Path('/content').exists() else '/').free/2**30,2)}
    if gpu:
        p=torch.cuda.get_device_properties(0)
        info.update({'gpu_name':p.name,'vram_gib':round(p.total_memory/2**30,2)})
        print('GPU:',p.name,'VRAM GiB:',info['vram_gib'])
        if not info['cuda_bf16_supported']:
            print('STOP for official Phase1-frozen bf16 protocol: GPU cannot run bf16. fp16 diagnostic smoke is separately labeled.')
    elif mps: print('Apple MPS available: local inference supported; CUDA training smoke remains separate.')  # MODIFIED
    else: print('CPU environment: schedule and dataset tools available; accelerated inference unavailable.')  # MODIFIED
    (d/'gpu_preflight.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))
    return info


def validate_dataset(args):
    records = pair_inventory(args.data_root)
    print('Paired summer patches:',len(records),'ROI groups:',len(set(r['roi'] for r in records)))
    for r in records[:min(args.check_pairs,len(records))]:
        validate_pair(r)
    print('GeoTIFF validity sampled:',min(args.check_pairs,len(records)),'pairs')
    if args.make_splits:
        # ADDED: Filter invalid pairs before splitting by ROI.
        valid, invalid = [], []
        for i,row in enumerate(records):
            try:
                validate_pair(row)
                valid.append(row)
            except Exception as exc:
                invalid.append({'id':row['id'],'reason':str(exc)})
            if (i+1)%1000==0: print('Validated actual S1/S2 patches:',i+1,'of',len(records))
        Path(args.outdir).mkdir(parents=True,exist_ok=True)
        (Path(args.outdir)/'excluded_invalid.json').write_text(json.dumps(invalid,indent=2))
        print('Usable paired patches after invalid exclusion:',len(valid),'excluded:',len(invalid))
        splits = group_splits(valid)
        write_manifest(splits,args.outdir)
        print('FROZEN SPLITS:',json.dumps({k:len(v) for k,v in splits.items()}))
    else:
        print('Pair inspection only. Run with --make-splits after complete summer archives are available.')


def import_model_components():
    import torch
    from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
    from transformers import CLIPTokenizer, CLIPTextModel
    return torch, DDPMScheduler, UNet2DConditionModel, AutoencoderKL, CLIPTokenizer, CLIPTextModel


def training_scheduler(which):
    torch, DDPMScheduler, *rest = import_model_components()
    beta,alpha,ab = native_schedule()
    powered_betas, powered_ab = power_schedule()
    if which == 'A':
        scheduler = DDPMScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,
                                  beta_schedule='scaled_linear',prediction_type='epsilon')
        expected=ab
    else:
        scheduler = DDPMScheduler(num_train_timesteps=1000,trained_betas=powered_betas,
                                  prediction_type='epsilon')  # ADDED
        expected=powered_ab
    np.testing.assert_allclose(scheduler.alphas_cumprod.cpu().numpy(),expected,rtol=1e-5,atol=1e-7)
    return scheduler


def checkpoint_inspect(outdir):
    from huggingface_hub import hf_hub_download
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    names=['scheduler/scheduler_config.json','unet/config.json','tokenizer/tokenizer_config.json']
    loaded={}
    for name in names:
        path=hf_hub_download(repo_id=MODEL_ID,filename=name,revision=MODEL_REVISION)
        loaded[name]=json.loads(Path(path).read_text())
    sch=loaded[names[0]]; unet=loaded[names[1]]
    assert sch['beta_schedule']=='scaled_linear' and sch['prediction_type']=='epsilon'
    assert sch['num_train_timesteps']==1000 and sch['beta_start']==.00085 and sch['beta_end']==.012
    assert unet['in_channels']==8, 'STOP: Expected pretrained 8-channel conditioning UNet'
    (out/'checkpoint_config_audit.json').write_text(json.dumps({'repo':MODEL_ID,'revision':MODEL_REVISION,'loaded':loaded},indent=2))
    print('Small configuration files verified; NOTE: large model weights not downloaded by this command.')


def run_zero(examples_path, outdir, count=3, device='auto'):  # MODIFIED
    """Generate baseline outputs without fine-tuning."""
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    from PIL import Image
    if count < 1: raise ValueError('count must be positive')  # ADDED
    if device == 'auto':  # ADDED
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')  # ADDED
    if device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')  # ADDED
    if device == 'mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')  # ADDED
    output=Path(outdir);output.mkdir(parents=True,exist_ok=True)
    examples=load_jsonl(examples_path)
    if not examples: raise ValueError('No prepared examples supplied')
    # ADDED
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device == 'cuda' else torch.float32  # MODIFIED: Use float32 on MPS/CPU.
    print('Loading actual pretrained checkpoint. First download is multi-GB.')
    pipe=StableDiffusionInstructPix2PixPipeline.from_pretrained(
        MODEL_ID,revision=MODEL_REVISION,torch_dtype=dtype)
    assert pipe.unet.config.in_channels==8
    pipe.scheduler=EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    if device == 'cuda': pipe.enable_model_cpu_offload()  # MODIFIED
    else:  # ADDED: MPS uses direct device placement.
        pipe.to(device)  # ADDED
        pipe.enable_attention_slicing()  # ADDED
        pipe.enable_vae_slicing()  # ADDED
    outputs=[]
    for idx,row in enumerate(examples[:count]):
        image=Image.open(row['image']).convert('RGB')
        g=torch.Generator(device='cuda' if device == 'cuda' else 'cpu').manual_seed(SEED+idx)  # MODIFIED: MPS uses CPU-generated noise.
        with torch.inference_mode():
            edited=pipe(prompt=row['prompt'],image=image,num_inference_steps=50,
                        guidance_scale=7.5,image_guidance_scale=1.5,generator=g).images[0]
        path=output/f'Run0_{idx:03d}_{row["id"]}_{row["product"]}.png'
        edited.save(path)
        outputs.append({'device':device,'dtype':str(dtype),  # MODIFIED
                        'id':row['id'],'product':row['product'],'seed':SEED+idx,
                        'prediction':str(path),'source':row['image'],'target':row['target'],
                        'checkpoint':MODEL_ID,'revision':MODEL_REVISION,'fine_tuned':False})
        print('Saved real pretrained Run 0 image:',path)
    (output/'run0_manifest.json').write_text(json.dumps(outputs,indent=2))
    del pipe
    if device == 'cuda': torch.cuda.empty_cache()  # MODIFIED
    elif device == 'mps': torch.mps.empty_cache()  # ADDED


def tensor_from_png(filename,device='cuda'):
    from PIL import Image
    import torch
    img=np.asarray(Image.open(filename).convert('RGB'),dtype=np.float32)
    return torch.from_numpy((img/127.5-1.0).transpose(2,0,1)).unsqueeze(0).to(device)


def one_run_smoke(which,rows,outdir,steps=1):
    import gc
    import torch
    import torch.nn.functional as F
    from contextlib import nullcontext
    _, _, UNet2DConditionModel, AutoencoderKL, CLIPTokenizer, CLIPTextModel=import_model_components()
    assert torch.cuda.is_available()
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if dtype==torch.float16:
        print('DIAGNOSTIC ONLY: fp16 used because this GPU cannot run declared bf16. Not official Run A/B.')
    vae=AutoencoderKL.from_pretrained(MODEL_ID,subfolder='vae',revision=MODEL_REVISION,torch_dtype=dtype).eval().to('cuda')
    text_encoder=CLIPTextModel.from_pretrained(MODEL_ID,subfolder='text_encoder',revision=MODEL_REVISION,torch_dtype=dtype).eval().to('cuda')
    tokenizer=CLIPTokenizer.from_pretrained(MODEL_ID,subfolder='tokenizer',revision=MODEL_REVISION)
    # MODIFIED: Preserve the pretrained eight-channel convolution.
    unet=UNet2DConditionModel.from_pretrained(MODEL_ID,subfolder='unet',revision=MODEL_REVISION,
                                               torch_dtype=torch.float32).to('cuda')
    assert unet.config.in_channels==8 and unet.conv_in.weight.shape[1]==8
    conv_sha=hashlib.sha256(unet.conv_in.weight.detach().float().cpu().numpy().tobytes()).hexdigest()
    print(f'Run {which}, verified 8-channel pretrained conv_in SHA256:',conv_sha)
    unet.enable_gradient_checkpointing();unet.train()
    vae.requires_grad_(False);text_encoder.requires_grad_(False)
    sch=training_scheduler(which)
    # ADDED: Eight-bit optimizer is for diagnostics only.
    import bitsandbytes as bnb
    optimizer=bnb.optim.AdamW8bit(unet.parameters(),lr=5e-5,weight_decay=.01)
    scaler=torch.cuda.amp.GradScaler(enabled=(dtype==torch.float16))
    os.makedirs(outdir,exist_ok=True)
    losses=[]
    # ADDED: Match A/B noise and timesteps after model loading.
    torch.manual_seed(SEED+123);torch.cuda.manual_seed_all(SEED+123)
    for i in range(steps):
        row=rows[i % len(rows)]
        src=tensor_from_png(row['image']); tgt=tensor_from_png(row['target'])
        with torch.no_grad(),torch.autocast('cuda',dtype=dtype):
            lat=vae.encode(tgt.to(dtype)).latent_dist.sample()*vae.config.scaling_factor
            cond=vae.encode(src.to(dtype)).latent_dist.mode()
            tok=tokenizer([row['prompt']],max_length=tokenizer.model_max_length,
                          padding='max_length',truncation=True,return_tensors='pt').input_ids.to('cuda')
            enc=text_encoder(tok)[0]
            # ADDED: Conditioning dropout, p=0.05.
            r=torch.rand(1,device='cuda'); p=.05
            if r.item() < 2*p:
                empty=tokenizer([''],max_length=tokenizer.model_max_length,padding='max_length',
                                truncation=True,return_tensors='pt').input_ids.to('cuda')
                enc=text_encoder(empty)[0]
            if (r.item()<p) or (2*p<=r.item()<3*p): cond=torch.zeros_like(cond)
            noise=torch.randn_like(lat)
            time=torch.randint(0,1000,(1,),device='cuda',dtype=torch.long)
            noisy=sch.add_noise(lat,noise,time)
            x=torch.cat([noisy,cond],dim=1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=dtype):
            pred=unet(x,time,enc,return_dict=False)[0]
            loss=F.mse_loss(pred.float(),noise.float())
        assert torch.isfinite(loss).item(),f'Run {which} step {i} nonfinite loss'
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unet.parameters(),1.0)
        scaler.step(optimizer);scaler.update()
        losses.append({'diagnostic_step':i+1,'loss':float(loss.item()),'noise_schedule':which})
        print(f'DIAGNOSTIC Run {which} step {i+1}/{steps} loss={loss.item():.6f}')
    outdir=Path(outdir)
    with (outdir/f'smoke_{which}_diagnostic.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['diagnostic_step','loss','noise_schedule']);w.writeheader();w.writerows(losses)
    (outdir/f'smoke_{which}_audit.json').write_text(json.dumps({
        'initial_conv_in_sha256':conv_sha,'pretrained_checkpoint':MODEL_ID,
        'revision':MODEL_REVISION,'power':NOISE_POWER if which=='B' else 1.0,
        'steps':steps,'dtype':str(dtype),'optimizer':'AdamW8bit (diagnostic only)',
        'declared_precision_bf16_met':bool(dtype==torch.bfloat16),
        'note':'NO official fine-tuning; diagnostic steps / tiny train-only sample; not Phase 2 epoch/train_loss CSV'},indent=2))
    del optimizer,unet,vae,text_encoder,sch,scaler
    gc.collect();torch.cuda.empty_cache()
    return conv_sha


def two_run_smoke(examples_path,outdir,steps):
    import torch
    gpu_report(outdir)
    if not torch.cuda.is_available():  # MODIFIED
        raise RuntimeError('This diagnostic uses CUDA autocast and bitsandbytes. On Mac use local checks and Run 0; the A/B gradient smoke needs a CUDA environment.')  # ADDED
    if steps < 1: raise ValueError('steps must be positive')  # ADDED
    rows=load_jsonl(examples_path)
    if not rows: raise ValueError('Provide rendered TRAIN examples, never test examples!')
    if any(row.get('split') != 'train' for row in rows):  # ADDED
        raise ValueError('Smoke test accepts only examples explicitly labeled split=train')  # ADDED
    a=one_run_smoke('A',rows,outdir,steps)
    b=one_run_smoke('B',rows,outdir,steps)
    if a != b: raise RuntimeError('STOP: A/B pretrained conv weight hashes differ!')
    print('PASS: A/B loaded identical initial 8-channel conv weights. Diagnostic only; official runs not started.')


def main():
    p=argparse.ArgumentParser(description='Phase 2 Colab Free starter, no fabricated training evidence')
    sub=p.add_subparsers(dest='command',required=True)
    def add(cmd,*opts):
        z=sub.add_parser(cmd)
        for a,kw in opts: z.add_argument(a,**kw)
        return z
    add('preflight', ('--outdir',dict(default='outputs/preflight')))
    add('schedule', ('--outdir',dict(default='outputs/schedule')))
    add('checkpoint',('--outdir',dict(default='outputs/checkpoint')))
    add('dataset',('--data-root',dict(required=True)),('--outdir',dict(default='outputs/manifests')),
        ('--check-pairs',dict(type=int,default=10)),('--make-splits',dict(action='store_true')))
    add('stats',('--train-manifest',dict(required=True)),('--out',dict(default='outputs/train_stats.json')),
        ('--max-patches',dict(type=int,default=None)),('--pixels-per-patch',dict(type=int,default=128)))
    add('render',('--manifest',dict(required=True)),('--stats',dict(required=True)),
        ('--outdir',dict(required=True)),('--max-patches',dict(type=int,default=3)))
    add('run0',('--examples',dict(required=True)),('--outdir',dict(default='outputs/run0')),
        ('--count',dict(type=int,default=3)),('--device',dict(choices=['auto','mps','cuda','cpu'],default='auto')))  # MODIFIED
    add('smoke',('--train-examples',dict(required=True)),('--outdir',dict(default='outputs/smoke')),
        ('--steps',dict(type=int,default=1)))
    args=p.parse_args()
    if args.command=='preflight':gpu_report(args.outdir)
    elif args.command=='schedule':save_schedule(args.outdir)
    elif args.command=='checkpoint':checkpoint_inspect(args.outdir)
    elif args.command=='dataset':validate_dataset(args)
    elif args.command=='stats':fit_stats(args.train_manifest,args.out,args.max_patches,args.pixels_per_patch)
    elif args.command=='render':render_examples(args.manifest,args.stats,args.outdir,args.max_patches)
    elif args.command=='run0':run_zero(args.examples,args.outdir,args.count,args.device)  # MODIFIED
    elif args.command=='smoke':two_run_smoke(args.train_examples,args.outdir,args.steps)

if __name__=='__main__':main()
