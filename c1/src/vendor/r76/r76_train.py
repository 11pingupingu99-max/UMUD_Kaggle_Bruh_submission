"""R76T: A76 boundary-supervised ConvNeXt and F76 ResNet34 fascicle model.
Training starts from ImageNet, never from a possibly overlapping old UMUD fold.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from pathlib import Path
from contextlib import nullcontext
import copy
import gc
import json
import math
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34, ResNet34_Weights, convnext_tiny, ConvNeXt_Tiny_Weights
from r45_core import (atomic_json, sha256, read_gray, read_binary, letterbox_image,
                      restore_map, axial_diff)
from r45_models import MuscleNet, Block, atomic_torch
from r45_decode import apo_pair_subpixel
from r45_apo import precision_mode
from r76_data import Dataset, digest
import r71_train

VERSION='R76T_v1.0.0'
COMMON=dict(side=1024, batch=1, accumulation=4, seed=20261027, n_folds=5,
            encoder_lr=4e-5, decoder_lr=4e-4, weight_decay=1e-4, warmup_epochs=2,
            rotation_degrees=7., scale_range=(.88,1.12), shift_frac=.04, workers=2,
            ema_decay=.995, save_every_steps=160, save_every_epochs=1, max_grad_norm=1.)
APO_CONFIG=dict(COMMON,epochs=24,edge_weight=2.,dice_weight=.5,
                edge_sigma=2.,edge_ce_weight=.04,coordinate_weight=.04,slope_weight=.04,
                geometry_ramp_epochs=5)
FASC_CONFIG=dict(COMMON,epochs=20,encoder_lr=6e-5,decoder_lr=6e-4,
                 recall_weight=.5,orientation_weight=.15,pos_weight=3.,
                 tube_threshold=.2,band_px=3)


class ResNetFasc(nn.Module):
    """A different encoder from R22/F71; full-resolution skip paths preserve thin traces."""
    def __init__(self):
        super().__init__();self.encoder=resnet34(weights=None)
        self.encoder.fc.requires_grad_(False)
        self.stem_full=Block(1,16)
        self.up16=Block(512+256,160);self.up8=Block(160+128,96)
        self.up4=Block(96+64,64);self.up2=Block(64+64,32);self.up1=Block(32+16,24)
        self.head=nn.Conv2d(24,3,1)
        nn.init.normal_(self.head.weight,std=.01)
        with torch.no_grad():self.head.bias.copy_(torch.tensor([-2.,1.,0.]))
        self.register_buffer('rgb_mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('rgb_std',torch.tensor([.229,.224,.225])[None,:,None,None])
    def train(self,mode=True):
        super().train(mode)
        # Batch 1: keep pretrained running statistics; affine BN parameters still learn.
        for m in self.encoder.modules():
            if isinstance(m,nn.BatchNorm2d):m.eval()
        return self
    def forward(self,x):
        full=self.stem_full(x);e=self.encoder
        a=e.relu(e.bn1(e.conv1((x.expand(-1,3,-1,-1)-self.rgb_mean)/self.rgb_std)))
        b=e.layer1(e.maxpool(a));c=e.layer2(b);d=e.layer3(c);z=e.layer4(d)
        def up(z,s,block):return block(torch.cat([F.interpolate(z,size=s.shape[-2:],mode='bilinear',align_corners=False),s],1))
        z=up(z,d,self.up16);z=up(z,c,self.up8);z=up(z,b,self.up4)
        z=up(z,a,self.up2);z=up(z,full,self.up1)
        return self.head(z)


def prepare_pretrained(project,run,task,allow_download=True):
    run=Path(run);pre=run/'pretrained';pre.mkdir(parents=True,exist_ok=True)
    if task=='apo':
        name='convnext_tiny_imagenet1k_v1_encoder.pt'
        candidates=[Path(project)/'outputs'/tag/'pretrained'/name for tag in ('R22_v1_0_0','R45T_v1_0_0')]
        candidates.append(pre/name)
        found=next((p for p in candidates if p.is_file()),None)
        if found:return found
        if not allow_download:raise FileNotFoundError('ImageNet ConvNeXt encoder not available')
        m=convnext_tiny(weights=None)
        m.load_state_dict(ConvNeXt_Tiny_Weights.IMAGENET1K_V1.get_state_dict(progress=True,check_hash=True))
        atomic_torch(pre/name,{'encoder':m.features.state_dict()});return pre/name
    dest=pre/'resnet34_imagenet1k_v1.pt'
    if not dest.is_file():
        if not allow_download:raise FileNotFoundError('ImageNet ResNet34 weights not available')
        sd=ResNet34_Weights.IMAGENET1K_V1.get_state_dict(progress=True,check_hash=True)
        atomic_torch(dest,{'encoder':sd})
    return dest


def new_model(task,pretrained=None):
    if task=='apo':
        m=MuscleNet(3)
        with torch.no_grad():m.head.bias.copy_(torch.tensor([-2.,0.,0.]))
    elif task=='fasc':m=ResNetFasc()
    else:raise ValueError(task)
    if pretrained is not None:
        sd=torch.load(pretrained,map_location='cpu',weights_only=True)
        m.encoder.load_state_dict(sd['encoder'],strict=True)
    return m


def load_model(path,device='cpu'):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('version')!=VERSION:raise ValueError(f'Not an R76 checkpoint: {path}')
    m=new_model(state['task']);m.load_state_dict({k:v.float() if v.is_floating_point() else v for k,v in state['model'].items()},strict=True)
    return m.to(device).eval(),state


def autocast(device):
    dtype,_=precision_mode(device)
    return torch.amp.autocast('cuda',dtype=dtype) if device=='cuda' else nullcontext()


def apo_loss(out,batch,cfg,epoch):
    out=out.float();t=batch['target'];v=batch['valid'];edge=batch['edge']
    logit=out[:,:1]
    # pos_weight=1 preserves the meaning of a 0.5 mask crossing.
    bce=(F.binary_cross_entropy_with_logits(logit,t,reduction='none')*edge).sum()/edge.sum().clamp_min(1)
    p=torch.sigmoid(logit)*v;dims=(1,2,3)
    dice=1-((2*(p*t).sum(dims)+1)/(p.sum(dims)+(t*v).sum(dims)+1)).mean()
    base=(1-cfg['dice_weight'])*bce+cfg['dice_weight']*dice
    y=batch['edge_y'];has=batch['edge_visible'];den=has.sum().clamp_min(1)
    h=out.shape[-2];rows=torch.arange(h,device=out.device,dtype=torch.float32)[None,None,:,None]
    aux=out[:,1:3].masked_fill(v.expand(-1,2,-1,-1)<.5,-1e4)
    lp=F.log_softmax(aux,dim=2)
    q=torch.exp(-.5*((rows-y[:,:,None,:])/cfg['edge_sigma'])**2)*v
    q=q/q.sum(2,keepdim=True).clamp_min(1e-7)
    ce=(-q*lp).sum(2);ce=(ce*has).sum()/den
    yp=(lp.exp()*rows).sum(2)
    coord=(F.smooth_l1_loss((yp-y)/4,torch.zeros_like(y),reduction='none')*has).sum()/den
    # A line slope is dimensionless; 1/atan converts to angle in the pixel frame.
    xx=torch.arange(y.shape[-1],device=y.device,dtype=torch.float32)[None,None,:]
    n=has.sum(2).clamp_min(1);xm=(xx*has).sum(2)/n
    xc=xx-xm[:,:,None];d=(xc.square()*has).sum(2).clamp_min(1)
    ap=(xc*yp*has).sum(2)/d;ag=(xc*y*has).sum(2)/d
    usable=has.sum(2)>=16
    slope=(torch.abs(torch.atan(ap)-torch.atan(ag))*180/math.pi*usable).sum()/usable.sum().clamp_min(1)
    ramp=min(1.,(epoch+1)/max(1,cfg['geometry_ramp_epochs']))
    total=base+ramp*(cfg['edge_ce_weight']*ce+cfg['coordinate_weight']*coord+cfg['slope_weight']*slope)
    vals=dict(bce=bce,dice=dice,edge_ce=ce,coordinate=coord,slope_deg=slope)
    return total,{k:float(a.detach()) for k,a in vals.items()}


def loss_fn(task,out,batch,cfg,epoch):
    if task=='apo':return apo_loss(out,batch,cfg,epoch)
    return r71_train.fasc_loss(out,batch['target'],batch['valid'],batch['cs'],batch['band'],cfg,cfg['pos_weight'])


@torch.inference_mode()
def predict(models,square,task,device='cpu',tta=True):
    if not models:raise ValueError('Empty ensemble')
    x=torch.from_numpy(np.ascontiguousarray(square[None,None],dtype=np.float32)).to(device)
    total_p=None;total_c=None;total_s=None;n=0
    for model in models:
        for flip in ([False,True] if tta else [False]):
            with autocast(device):o=model(torch.flip(x,(-1,)) if flip else x)
            o=o.float()
            if flip:o=torch.flip(o,(-1,))
            p=o[:,:1].sigmoid()
            total_p=p if total_p is None else total_p+p
            if task=='fasc':
                cs=F.normalize(o[:,1:3],dim=1,eps=1e-6)
                if flip:cs=cs*cs.new_tensor([1.,-1.])[None,:,None,None]
                total_c=cs[:,:1] if total_c is None else total_c+cs[:,:1]
                total_s=cs[:,1:2] if total_s is None else total_s+cs[:,1:2]
            n+=1
    arr=lambda z:(z/n)[0,0].cpu().numpy().astype(np.float32)
    return arr(total_p) if task=='apo' else tuple(arr(z) for z in (total_p,total_c,total_s))


def complete_folds(out_dir):
    return sorted(int(p.parent.name.split('_')[-1]) for p in Path(out_dir).glob('fold_*/DONE.json'))


class SessionBudgetReached(Exception):pass


def train_fold(frame,cache,out_dir,pretrained,cfg,fold,task,device='cuda',deadline=None,log=print):
    out=Path(out_dir)/f'fold_{fold}';out.mkdir(parents=True,exist_ok=True)
    tr=frame[frame.fold!=fold].reset_index(drop=True);va=frame[frame.fold==fold].reset_index(drop=True)
    if len(tr)==0 or len(va)==0:raise ValueError('Empty train/held split')
    if set(tr.group_id)&set(va.group_id):raise ValueError('Duplicate/sequence leak')
    science={k:v for k,v in cfg.items() if k not in ('workers','batch','accumulation','save_every_steps','save_every_epochs')}
    code_hash=digest({name:sha256(Path(__file__).with_name(name)) for name in
        ('r76_train.py','r76_data.py','r45_models.py','r45_core.py','r45_apo.py','r45_decode.py','r71_train.py')})
    identity=digest({'version':VERSION,'training_code_sha256':code_hash,'task':task,'cfg':science,'fold':fold,
                     'pretrained_sha256':sha256(pretrained) if pretrained else 'RANDOM_TEST_ONLY',
                     'tr':tr[['name','image_sha256','mask_sha256','group_id']].to_dict('records'),
                     'va':va[['name','image_sha256','mask_sha256','group_id']].to_dict('records')})
    done=out/'DONE.json'
    if done.is_file():
        d=json.loads(done.read_text())
        if d['identity']!=identity:raise RuntimeError('Finished fold identity changed; use a new RUN_NAME')
        if sha256(out/'final.pt')!=d['final_sha256']:raise RuntimeError('Final checkpoint hash mismatch')
        return d
    torch.manual_seed(cfg['seed']+fold);np.random.seed(cfg['seed']+fold)
    if device=='cuda':torch.cuda.manual_seed_all(cfg['seed']+fold)
    model=new_model(task,pretrained).to(device)
    enc=list(p for p in model.encoder.parameters() if p.requires_grad);ids={id(p) for p in enc}
    dec=[p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    opt=torch.optim.AdamW([{'params':enc,'lr':cfg['encoder_lr']},{'params':dec,'lr':cfg['decoder_lr']}],weight_decay=cfg['weight_decay'])
    schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda e:min(1.,(e+1)/max(cfg['warmup_epochs'],1))*(.05+.95*.5*(1+math.cos(math.pi*min(e,cfg['epochs'])/cfg['epochs']))))
    dtype,use_scaler=precision_mode(device);scaler=torch.amp.GradScaler('cuda',enabled=use_scaler)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    epoch=0;cursor=0;updates=0;history=[];stats={};seen=0;last=out/'last.pt'
    effective=cfg['batch']*cfg['accumulation']
    if last.is_file():
        st=torch.load(last,map_location='cpu',weights_only=True)
        if st['identity']!=identity:raise RuntimeError('Resume identity changed; use a new RUN_NAME')
        if st['effective_batch']!=effective:raise RuntimeError('Resume must preserve batch*accumulation')
        model.load_state_dict(st['model']);ema.load_state_dict(st['ema']);opt.load_state_dict(st['optimizer']);schedule.load_state_dict(st['scheduler'])
        if use_scaler and st['scaler']:scaler.load_state_dict(st['scaler'])
        for a in opt.state.values():
            for k,v in a.items():
                if torch.is_tensor(v) and k!='step':a[k]=v.to(device)
        epoch=st['epoch'];cursor=st['cursor'];updates=st['updates'];history=st['history'];stats=st['stats'];seen=st['seen']
        torch.set_rng_state(st['torch_rng'])
        if device=='cuda' and st.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(st['cuda_rng'])
        del st;log(f'{task} fold {fold}: resumed epoch {epoch+1}, sample {cursor}/{len(tr)}')
    ds=Dataset(tr,cache,task,cfg,True,fold)
    def cpu_state(m):return {k:v.detach().cpu() for k,v in m.state_dict().items()}
    def save():
        atomic_torch(last,{'version':VERSION,'identity':identity,'task':task,'model':cpu_state(model),'ema':cpu_state(ema),
                          'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),'scaler':scaler.state_dict() if use_scaler else None,
                          'epoch':epoch,'cursor':cursor,'updates':updates,'history':history,'stats':stats,'seen':seen,
                          'effective_batch':effective,'torch_rng':torch.get_rng_state(),
                          'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else None})
    for ep in range(epoch,cfg['epochs']):
        epoch=ep;ds.epoch=ep
        if deadline and time.time()>=deadline:save();raise SessionBudgetReached(f'{task} fold {fold}: saved before epoch {ep+1}')
        order=torch.randperm(len(ds),generator=torch.Generator().manual_seed(cfg['seed']+fold*1000+ep)).tolist()
        original_cursor=cursor
        loader=torch.utils.data.DataLoader(torch.utils.data.Subset(ds,order[cursor:]),batch_size=cfg['batch'],shuffle=False,
                  num_workers=cfg['workers'] if device=='cuda' else 0,pin_memory=device=='cuda',persistent_workers=False,
                  generator=torch.Generator().manual_seed(cfg['seed']+ep+991))
        model.train();opt.zero_grad(set_to_none=True);start=time.time();pending=0;window_size=min(effective,len(ds)-cursor)
        for step,b in enumerate(loader):
            b={k:v.to(device,non_blocking=True) for k,v in b.items()};count=len(b['x'])
            with autocast(device):prediction=model(b['x'])
            loss,metrics=loss_fn(task,prediction,b,cfg,ep)
            if not torch.isfinite(loss):raise FloatingPointError(f'{task}: nonfinite loss, epoch {ep}, cursor {cursor}')
            scaler.scale(loss*count/window_size).backward()
            pending+=count;cursor+=count;seen+=count
            for k,v in dict(loss=float(loss.detach()),**metrics).items():stats[k]=stats.get(k,0.)+v*count
            boundary=pending>=window_size
            if boundary:
                scaler.unscale_(opt)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['max_grad_norm'])
                if not torch.isfinite(norm):
                    # fp16 overflow: discard the entire window, let the scaler reduce its scale.
                    if not use_scaler:raise FloatingPointError('Nonfinite gradients in fp32/bf16')
                oldscale=scaler.get_scale();scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
                did_step=(not use_scaler) or scaler.get_scale()>=oldscale
                if did_step:
                    updates+=1;decay=min(cfg['ema_decay'],1-1/(updates+1))
                    with torch.no_grad():
                        for a,b0 in zip(ema.state_dict().values(),model.state_dict().values()):
                            if a.is_floating_point():a.lerp_(b0.detach(),1-decay)
                            else:a.copy_(b0)
                pending=0;window_size=min(effective,len(ds)-cursor)
                if deadline and time.time()>=deadline:
                    save();raise SessionBudgetReached(f'{task} fold {fold}: saved epoch {ep+1}, sample {cursor}/{len(ds)}')
                if updates and updates%cfg['save_every_steps']==0:save()
            if (step+1)%100==0:
                seconds=time.time()-start
                log(f'{task} fold {fold} epoch {ep+1}/{cfg["epochs"]}: {cursor}/{len(ds)} loss {stats["loss"]/max(seen,1):.4f}; '
                    f'{seconds/max(cursor-original_cursor,1):.3f}s/image')
        row=dict(epoch=ep+1,seconds=time.time()-start,**{k:v/max(seen,1) for k,v in stats.items()})
        history.append(row);pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        log(f'{task} fold {fold} epoch {ep+1}/{cfg["epochs"]}: loss={row["loss"]:.4f}, {row["seconds"]:.0f}s')
        schedule.step();epoch=ep+1;cursor=0;seen=0;stats={}
        if epoch%cfg['save_every_epochs']==0:save()
    # Quantize BEFORE OOF so evaluation uses exactly the exported weights.
    state={k:(v.detach().half().cpu() if v.is_floating_point() else v.detach().cpu()) for k,v in ema.state_dict().items()}
    final=out/'final.pt';atomic_torch(final,{'version':VERSION,'identity':identity,'task':task,'fold':fold,'epoch':cfg['epochs'],'cfg':cfg,'model':state})
    del model,ema,opt,ds;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    model,_=load_model(final,device)
    oof=evaluate(va,cache,cfg,[model],task,device)
    oof.to_csv(out/'oof.csv',index=False)
    report=summarize(oof,task)
    result=dict(version=VERSION,identity=identity,task=task,fold=fold,epochs=cfg['epochs'],side=cfg['side'],n_train=len(tr),n_held=len(va),
                final_sha256=sha256(final),oof_summary=report)
    atomic_json(done,result)
    if last.exists():last.unlink()
    del model;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return result


def evaluate(held,cache,cfg,models,task,device):
    if task=='fasc':
        ds=Dataset(held,cache,task,cfg,False,0);rows=[]
        for i in range(len(ds)):
            b=ds[i];p,c,s=predict(models,b['x'][0].numpy(),task,device,True)
            t=b['target'][0].numpy();band=b['band'][0].numpy();cs=b['cs'].numpy();v=b['valid'][0].numpy()
            pr=(p>=.5)*v;tp=float((pr*t).sum());pred_ang=.5*np.degrees(np.arctan2(s,c));gt_ang=.5*np.degrees(np.arctan2(cs[1],cs[0]))
            rows.append(dict(name=held.iloc[i]['name'],fold=int(held.iloc[i]['fold']),
                             dice=(2*tp+1)/(pr.sum()+t.sum()+1),recall=(tp+1)/(t.sum()+1),
                             angle_mae=float((axial_diff(pred_ang,gt_ang)*band).sum()/max(1.,band.sum()))))
        return pd.DataFrame(rows)
    rows=[]
    for r in held.itertuples(index=False):
        gray=read_gray(r.image);gt=read_binary(r.mask)
        if gt.shape!=gray.shape:
            import cv2
            gt=(cv2.resize(gt.astype(np.float32),(gray.shape[1],gray.shape[0]))>=.5).astype(np.uint8)
        p=restore_map(predict(models,letterbox_image(gray,cfg['side'])[0],task,device,True),gray.shape)
        pr=p>=.5;dice=float((2*(pr&(gt>0)).sum()+1)/(pr.sum()+(gt>0).sum()+1))
        a,b=apo_pair_subpixel(gt.astype(np.float32)),apo_pair_subpixel(p)
        row=dict(name=r.name,fold=int(r.fold),dice=dice,gt_pair=a is not None,pred_pair=b is not None)
        if a is not None and b is not None:
            same=max(abs(a['m'][k]-b['m'][k]) for k in ('upper_mid_y','lower_mid_y'))<.05*gray.shape[0]
            row.update(same_pair=bool(same),mt_error_px=b['m']['mt_inner_mean']-a['m']['mt_inner_mean'],
                       mt_relative_error=(b['m']['mt_inner_mean']-a['m']['mt_inner_mean'])/a['m']['mt_inner_mean'],
                       deep_error_deg=float(axial_diff(a['m']['deep_top_deg'],b['m']['deep_top_deg'])))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(oof,task):
    s=dict(images=len(oof),dice_mean=float(oof.dice.mean()),dice_median=float(oof.dice.median()))
    if task=='fasc':
        s.update(recall_mean=float(oof.recall.mean()),angle_mae_mean=float(oof.angle_mae.mean()));return s
    s.update(gt_pair_missing=int((~oof.gt_pair.astype(bool)).sum()),pred_pair_missing=int((~oof.pred_pair.astype(bool)).sum()))
    if 'same_pair' in oof:
        s['wrong_pair']=int((oof.same_pair==False).sum())
        for tag,d in [('all_pairs',oof),('same_pair',oof[oof.same_pair==True])]:
            for col in ('mt_error_px','mt_relative_error','deep_error_deg'):
                a=pd.to_numeric(d[col],errors='coerce').dropna()
                if len(a):
                    s[f'{tag}_{col}_n']=len(a);s[f'{tag}_{col}_mae']=float(a.abs().mean())
                    s[f'{tag}_{col}_p90']=float(a.abs().quantile(.9));s[f'{tag}_{col}_bias']=float(a.median())
    return s
