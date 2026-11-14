"""Full-resolution residual U-Net. SPDX-License-Identifier: GPL-3.0-or-later"""
from pathlib import Path
import os, json, random, math, time, gc
import numpy as np
import pandas as pd
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from umud_core import *

class Block(nn.Module):
    def __init__(self,ci,co):
        super().__init__();g=min(8,co)
        self.net=nn.Sequential(nn.Conv2d(ci,co,3,padding=1,bias=False),nn.GroupNorm(g,co),nn.SiLU(),
                               nn.Conv2d(co,co,3,padding=1,bias=False),nn.GroupNorm(g,co))
        self.skip=nn.Conv2d(ci,co,1,bias=False) if ci!=co else nn.Identity()
    def forward(self,x):return F.silu(self.net(x)+self.skip(x))

class FullResUNet(nn.Module):
    def __init__(self,base=24):
        super().__init__();ch=[base,base*2,base*4,base*8,base*16]
        self.enc=nn.ModuleList([Block(1 if i==0 else ch[i-1],c) for i,c in enumerate(ch)])
        self.dec=nn.ModuleList([Block(ch[i]+ch[i-1],ch[i-1]) for i in range(4,0,-1)])
        self.head=nn.Conv2d(base,1,1)
    def forward(self,x):
        fs=[]
        for i,b in enumerate(self.enc):x=b(x if i==0 else F.avg_pool2d(x,2));fs.append(x)
        for i,b in enumerate(self.dec):
            s=fs[-2-i];x=F.interpolate(x,size=s.shape[-2:],mode='bilinear',align_corners=False);x=b(torch.cat((x,s),1))
        return self.head(x)

def soft_erode(x):
    return torch.minimum(-F.max_pool2d(-x,(3,1),1,(1,0)),-F.max_pool2d(-x,(1,3),1,(0,1)))

def soft_skeleton(x,steps=3):
    def op(t):return F.max_pool2d(soft_erode(t),3,1,1)
    sk=F.relu(x-op(x))
    for _ in range(steps):
        x=soft_erode(x);d=F.relu(x-op(x));sk=sk+F.relu(d-sk*d)
    return sk

def segmentation_loss(logit,y,valid,line,pos_weight):
    # Reductions are FP32 even under autocast. Padding never enters any loss term.
    logit=logit.float();y=y.float();valid=valid.float();line=line.float();p=torch.sigmoid(logit)*valid
    bce=F.binary_cross_entropy_with_logits(logit,y,reduction='none',pos_weight=logit.new_tensor(pos_weight))
    bce=(bce*valid).sum()/valid.sum().clamp_min(1)
    dims=(1,2,3);dice=1-((2*(p*y).sum(dims)+1)/(p.sum(dims)+(y*valid).sum(dims)+1)).mean()
    sk=soft_skeleton(p);tp=((sk*y).sum(dims)+1)/(sk.sum(dims)+1)
    ts=((line*p).sum(dims)+1)/(line.sum(dims)+1)
    cl=1-(2*tp*ts/(tp+ts).clamp_min(1e-6)).mean()
    return .5*bce+.5*dice+.15*cl

class CachedDataset(torch.utils.data.Dataset):
    def __init__(self,frame,cache,augment=False):self.frame=frame.reset_index(drop=True);self.cache=Path(cache);self.augment=augment
    def __len__(self):return len(self.frame)
    def __getitem__(self,i):
        with np.load(self.cache/(self.frame.iloc[i].sample_id+'.npz'),allow_pickle=False) as z:
            x=z['image'].astype(np.float32)/255.;y=z['target'].astype(np.float32);line=z['line'].astype(np.float32);v=z['valid'].astype(np.float32)
        if self.augment:
            if random.random()<.5:x=x[:,::-1];y=y[:,::-1];line=line[:,::-1];v=v[:,::-1]
            # Photometric changes before normalization; no geometric downsampling of labels.
            x=np.clip(x,0,1)**random.uniform(.8,1.2)
            x=np.clip(x*random.uniform(.85,1.15)+random.uniform(-.03,.03),0,1)
            if random.random()<.3:x=np.clip(x+np.random.normal(0,.015,x.shape),0,1)
        return tuple(torch.from_numpy(np.ascontiguousarray(t[None],dtype=np.float32)) for t in (x,y,v,line))

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def rng_state():
    n=np.random.get_state()
    return dict(py=random.getstate(),np=[n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4])],
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(s):
    random.setstate(s['py']);n=s['np'];np.random.set_state((n[0],np.array(n[1],np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(s['torch'].cpu())
    if torch.cuda.is_available() and s['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])

def atomic_torch(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.pt');torch.save(state,tmp);os.replace(tmp,path)

def load_model(path,device='cpu'):
    d=torch.load(path,map_location='cpu',weights_only=True)
    if d.get('version')!=VERSION:raise ValueError('Wrong checkpoint version')
    model=FullResUNet(d['config']['base']);model.load_state_dict(d['model'],strict=True);model.to(device).eval();return model,d

@torch.inference_mode()
def predict_full(model,image,side=768,device='cpu',tta=False):
    x,_=letterbox_image(image,side);t=torch.from_numpy(x[None,None]).to(device)
    with torch.amp.autocast(device_type=device,dtype=torch.float16,enabled=(device=='cuda')):
        p=torch.sigmoid(model(t).float())
        if tta:p=(p+torch.flip(torch.sigmoid(model(torch.flip(t,(-1,))).float()),(-1,)))/2
    return restore_probability(p[0,0].cpu().numpy(),image.shape)

@torch.inference_mode()
def evaluate(model,frame,cache,device,thresholds=(.5,),prob_dir=None):
    ds=CachedDataset(frame,cache);model.eval();rows=[]
    if prob_dir:Path(prob_dir).mkdir(parents=True,exist_ok=True)
    for i in range(len(ds)):
        x,y,v,line=ds[i]
        with torch.amp.autocast(device_type=device,dtype=torch.float16,enabled=(device=='cuda')):p=torch.sigmoid(model(x[None].to(device)).float())[0,0].cpu().numpy()
        r=frame.iloc[i]
        if prob_dir:np.save(Path(prob_dir)/(r.sample_id+'.npy'),p.astype(np.float16),allow_pickle=False)
        for threshold in thresholds:
            metrics=segmentation_metrics(p,y[0].numpy(),line[0].numpy(),v[0].numpy(),threshold)
            rows.append(dict(sample_id=r.sample_id,group_id=int(r.group_id),fold=int(r.fold),threshold=threshold,**metrics))
    return pd.DataFrame(rows)

def summarize_evaluation(frame):
    rows=[]
    for thr,g in frame.groupby('threshold'):
        by=g.groupby('group_id')[['dice','center_f1','angle_error','angle_valid']].mean()
        rows.append(dict(threshold=float(thr),n=len(g),groups=len(by),dice=float(g.dice.mean()),center_f1=float(g.center_f1.mean()),
                         group_dice=float(by.dice.mean()),group_center_f1=float(by.center_f1.mean()),
                         angle_coverage=float(g.angle_valid.mean()),angle_MAE=float(g.angle_error.mean()) if g.angle_error.notna().any() else None))
    return rows

def train_canary(frame,cache,device,out,steps=100):
    """Tests learnability on fixed nonempty training patches, with fresh throwaway weights."""
    seed_all(731);ds=CachedDataset(frame,cache);items=[]
    for i in range(min(len(ds),30)):
        x,y,v,l=ds[i];yy,xx=torch.where(l[0]>0)
        if len(xx)==0:continue
        cy=int(yy.median());cx=int(xx.median());h,w=y.shape[-2:];y0=max(0,min(h-256,cy-128));x0=max(0,min(w-256,cx-128))
        items.append(tuple(z[:,y0:y0+256,x0:x0+256] for z in (x,y,v,l)))
        if len(items)==2:break
    if len(items)<2:raise RuntimeError('Canary needs two nonempty training masks')
    batch=[torch.stack([it[j] for it in items]).to(device) for j in range(4)];x,y,v,l=batch
    model=FullResUNet(base=16).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001)
    initial=None;final=None
    for step in range(steps):
        model.train();opt.zero_grad(set_to_none=True);z=model(x);loss=segmentation_loss(z,y,v,l,10.)
        if not torch.isfinite(loss):raise RuntimeError('Canary nonfinite loss')
        loss.backward();opt.step()
        if initial is None:initial=float(loss.detach())
        final=float(loss.detach())
    model.eval()
    with torch.no_grad():pr=torch.sigmoid(model(x))>.5;dice=float(((2*(pr*y).sum((1,2,3))+1)/(pr.sum((1,2,3))+y.sum((1,2,3))+1)).mean())
    report=dict(steps=steps,initial_loss=initial,final_loss=final,dice=dice,passed=bool(final<.7*initial and dice>.5))
    atomic_json(out,report);del model,opt,batch;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    if not report['passed']:raise RuntimeError(f'LEARNABILITY GATE FAILED: {report}. Inspect data before a long run.')
    return report

def train(frame,cache,run_dir,config,mode='develop'):
    """Develop: folds 1/2/3 train, 4 validation, 0 held out.
    Refit: all data, fixed selected epoch count; never reselect on fold 0/test.
    Resume completes the last epoch boundary; an interrupted partial epoch repeats.
    """
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    if mode not in ('develop','refit'):raise ValueError(mode)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cpu' and not config.get('allow_cpu',False):raise RuntimeError('Select a GPU runtime for training')
    torch.set_num_threads(min(4,os.cpu_count() or 1));seed_all(config['seed'])
    tr=frame[frame.fold.isin([1,2,3])].copy() if mode=='develop' else frame.copy()
    va=frame[frame.fold==4].copy();out=run_dir/mode;out.mkdir(exist_ok=True)
    identity=hashlib.sha256(json.dumps(dict(version=VERSION,config=config,mode=mode,rows=frame[['sample_id','fold','group_id']].to_dict('records')),sort_keys=True).encode()).hexdigest()
    old=out/'config.json'
    if old.exists() and json.loads(old.read_text())['identity']!=identity:raise RuntimeError('Configuration changed: choose a new RUN_NAME')
    atomic_json(old,dict(identity=identity,config=config,mode=mode))
    dataset=CachedDataset(tr,cache,augment=True)
    # Foreground weight uses only this run's training partition.
    pos=tot=0
    for i in np.linspace(0,len(dataset)-1,min(128,len(dataset))).astype(int):
        _,y,v,_=dataset[i];pos+=float(y.sum());tot+=float(v.sum())
    pw=float(np.clip(math.sqrt((tot-pos)/max(pos,1)),1,30))
    epochs=config['epochs'];selected=None
    if mode=='refit':
        selection=run_dir/'develop'/'selection.json'
        if not selection.exists():raise RuntimeError('Run develop before refit')
        selected=json.loads(selection.read_text());epochs=int(selected['best_epoch'])
    model=FullResUNet(config['base']).to(device);opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=1e-4)
    scaler=torch.amp.GradScaler(device,enabled=device=='cuda')
    # Keep the development learning-rate trajectory in refit; do not shorten cosine T_max.
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=config['epochs'],eta_min=config['lr']*.05)
    start=0;best=-1.;best_epoch=0;stale=0;history=[]
    last=out/'last.pt'
    if last.exists():
        st=torch.load(last,map_location='cpu',weights_only=True)
        if st['identity']!=identity:raise RuntimeError('Checkpoint identity mismatch')
        model.load_state_dict(st['model'],strict=True);opt.load_state_dict(st['optimizer']);scaler.load_state_dict(st['scaler']);sched.load_state_dict(st['scheduler'])
        for vals in opt.state.values():
            for key,v in vals.items():
                if torch.is_tensor(v):vals[key]=v.to(device)
        start=st['epoch'];best=st['best'];best_epoch=st['best_epoch'];stale=st['stale'];history=st['history'];pw=st['pos_weight'];restore_rng(st['rng'])
    begun=time.time()
    for epoch in range(start,epochs):
        if mode=='develop' and stale>=config['patience']:break
        model.train();loader=torch.utils.data.DataLoader(dataset,batch_size=config['batch'],shuffle=True,num_workers=0,pin_memory=device=='cuda')
        running=0.;count=0
        for step,batch in enumerate(loader,1):
            x,y,v,l=[t.to(device,non_blocking=True) for t in batch];opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):logit=model(x);loss=segmentation_loss(logit,y,v,l,pw)
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss in epoch {epoch+1}')
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(opt);scaler.update();running+=float(loss.detach())*len(x);count+=len(x)
            if step%50==0:
                print(f'epoch={epoch+1}/{epochs} batch={step}/{len(loader)} loss={running/count:.4f}',flush=True)
                atomic_json(out/'heartbeat.json',dict(epoch=epoch+1,batch=step,batches=len(loader),loss=running/count,time=time.time()))
        sched.step();valscore=None
        if mode=='develop':
            metrics=evaluate(model,va,cache,device);summary=summarize_evaluation(metrics)[0]
            valscore=summary['group_center_f1'];metrics.to_csv(out/'last_validation.csv',index=False)
            if valscore>best+1e-4:
                best=valscore;best_epoch=epoch+1;stale=0
                atomic_torch(out/'best.pt',dict(version=VERSION,model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config,identity=identity,epoch=best_epoch))
            else:stale+=1
        else:best_epoch=epoch+1
        history.append(dict(epoch=epoch+1,train_loss=running/count,val_group_center_f1=valscore,lr=opt.param_groups[0]['lr']))
        state=dict(version=VERSION,config=config,identity=identity,model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),scheduler=sched.state_dict(),
                   epoch=epoch+1,best=best,best_epoch=best_epoch,stale=stale,pos_weight=pw,rng=rng_state(),history=history)
        atomic_torch(last,state);pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print(history[-1],flush=True)
        if (time.time()-begun)/3600>config['budget_hours'] and epoch+1<epochs and stale<config['patience']:
            raise RuntimeError('SESSION BUDGET reached; checkpoint saved. Run all again to resume.')
    if mode=='develop':
        model,meta=load_model(out/'best.pt',device)
        # Three segmentation thresholds, selected ONLY on inner fold 4.
        metrics=evaluate(model,va,cache,device,thresholds=(.35,.5,.65));summaries=summarize_evaluation(metrics)
        picked=max(summaries,key=lambda d:d['group_center_f1'])
        atomic_json(out/'selection.json',dict(best_epoch=best_epoch,threshold=picked['threshold'],selection='fold4_group_center_f1',summaries=summaries,
                    identity=identity,checkpoint_sha256=sha256(out/'best.pt'),not_competition_score=True))
        metrics.to_csv(out/'threshold_validation.csv',index=False)
    else:
        atomic_torch(out/'best.pt',dict(version=VERSION,model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config,identity=identity,epoch=epochs))
        atomic_json(out/'selection.json',dict(best_epoch=epochs,threshold=selected['threshold'],selection='fixed_from_develop',identity=identity,checkpoint_sha256=sha256(out/'best.pt')))
    return out

class LegacyDinoApo(nn.Module):
    """Exact R12B architecture. Never downloads weights during inference."""
    def __init__(self,size=518):
        super().__init__();import timm
        self.enc=timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=False,num_classes=0,img_size=size)
        self.grid=size//self.enc.patch_embed.patch_size[0];self.size=size
        self.dec=nn.Sequential(nn.Conv2d(self.enc.embed_dim,256,3,padding=1),nn.BatchNorm2d(256),nn.GELU(),
                              nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(256,128,3,padding=1),nn.BatchNorm2d(128),nn.GELU(),
                              nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(128,64,3,padding=1),nn.BatchNorm2d(64),nn.GELU(),nn.Conv2d(64,1,1))
    def forward(self,x):
        t=self.enc.forward_features(x)[:,-self.grid*self.grid:]
        z=t.transpose(1,2).reshape(x.shape[0],-1,self.grid,self.grid)
        return F.interpolate(self.dec(z),size=(self.size,self.size),mode='bilinear',align_corners=False)

def load_legacy_apo(path,device):
    d=torch.load(path,map_location='cpu',weights_only=True);sd=d.get('model',d.get('state_dict',d))
    m=LegacyDinoApo();m.load_state_dict(sd,strict=True);m.to(device).eval();return m

@torch.inference_mode()
def predict_legacy_apo(model,path,device):
    # Preserve exact R12B uint8 BGR->RGB and anisotropic square input contract.
    image=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if image is None:raise ValueError(path)
    h,w=image.shape[:2];x=cv2.resize(image,(518,518),interpolation=cv2.INTER_LINEAR)[:,:,::-1].astype(np.float32)/255.
    x=(x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    t=torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1)[None])).to(device)
    p=torch.sigmoid(model(t).float())[0,0].cpu().numpy()
    return cv2.resize(p,(w,h),interpolation=cv2.INTER_LINEAR)
