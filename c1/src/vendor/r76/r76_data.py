"""R76: persistent, content-addressed caches and duplicate-aware grouped folds.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from pathlib import Path
import hashlib
import json
import os
import tarfile
import time
import cv2
import numpy as np
import pandas as pd
import torch
from r45_core import (read_gray, read_binary, normalize, letterbox_image, letterbox_soft,
                      aponeurosis_lines, atomic_json, sha256)
from r45_apo import augment as apo_augment, _edge_weight
import r71_train

DATA_VERSION = 'R76_DATA_v1.0.0'


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def pairs(image_dir, mask_dir):
    ext = {'.png', '.tif', '.tiff', '.bmp', '.jpg', '.jpeg'}
    def index(folder):
        out = {}
        for p in sorted(Path(folder).iterdir()):
            if p.suffix.lower() in ext:
                if p.stem in out:
                    raise ValueError(f'Ambiguous image stem: {p.stem}')
                out[p.stem] = p
        return out
    a, b = index(image_dir), index(mask_dir)
    common = sorted(a.keys() & b.keys())
    if not common:
        raise FileNotFoundError(f'No paired training data in {image_dir} / {mask_dir}')
    return pd.DataFrame([dict(name=n, file_name=a[n].name, image=str(a[n]), mask=str(b[n])) for n in common])


class UnionFind:
    def __init__(self, n): self.parent = list(range(n))
    def root(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]; i = self.parent[i]
        return i
    def join(self, i, j):
        a, b = self.root(i), self.root(j)
        if a != b: self.parent[max(a, b)] = min(a, b)


def audit_and_folds(frame, out_dir, n_folds=5, seed=20261027, legacy_folds=None,
                    block_size=10, near_corr=.997, log=print):
    """Join legacy sequence blocks, identical decoded images (including mirrored copies),
    and conservative near-duplicates. A group never crosses a new fold.
    Known subject/video identifiers should replace approximate filename blocks when available.
    Similarity is only a leakage check; no test labels or predictions enter the split.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    f = frame.sort_values('name').reset_index(drop=True).copy()
    old = pd.read_csv(legacy_folds) if legacy_folds and Path(legacy_folds).is_file() else None
    if old is not None:
        key = 'name' if 'name' in old else 'file_name'
        cols = [key] + [k for k in ('group_id', 'fold') if k in old]
        old = old[cols].drop_duplicates(key).rename(columns={'group_id':'legacy_group', 'fold':'legacy_fold'})
        f = f.merge(old, on=key, how='left', validate='one_to_one')
    if 'legacy_group' not in f:
        f['legacy_group'] = np.arange(len(f)) // block_size
    if f.legacy_group.isna().any():
        raise ValueError('Some training images are missing legacy sequence groups; inspect data before training')
    uf = UnionFind(len(f)); hashes = {}; mirrors = {}; thumbs = []; shapes = []; rows = []
    for _, g in f.groupby('legacy_group', sort=False):
        idx = g.index.tolist()
        for j in idx[1:]: uf.join(idx[0], j)
    same_image = UnionFind(len(f))
    for i, r in enumerate(f.itertuples(index=False)):
        a = read_gray(r.image); b = read_binary(r.mask)
        decoded = np.ascontiguousarray(a)
        prefix = f'{a.shape}|{a.dtype}|'.encode()
        image_hash = hashlib.sha256(prefix + decoded.tobytes()).hexdigest()
        mirror_hash = hashlib.sha256(prefix + np.ascontiguousarray(a[:, ::-1]).tobytes()).hexdigest()
        canonical = min(image_hash, mirror_hash)
        if canonical in hashes:
            j = hashes[canonical]; uf.join(i, j); same_image.join(i, j)
        else: hashes[canonical] = i
        # A crop reduces dominance of unchanged device UI in a near-duplicate test.
        h, w = a.shape; crop = normalize(a)[int(.12*h):int(.88*h), int(.08*w):int(.92*w)]
        thumb = cv2.resize(crop, (48, 40), interpolation=cv2.INTER_AREA).astype(np.float32)
        thumbs.append(thumb.ravel()); shapes.append(a.shape)
        rows.append(dict(name=r.name, image_sha256=sha256(r.image), mask_sha256=sha256(r.mask),
                         decoded_sha256=image_hash, mirror_canonical=canonical, h=h, w=w,
                         mask_h=b.shape[0], mask_w=b.shape[1], mask_fg=float(b.mean())))
        if (i+1) % 200 == 0: log(f'audit: {i+1}/{len(f)}')
    t = np.stack(thumbs); z = t - t.mean(1, keepdims=True)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.maximum(norms, 1e-6)
    links = []
    # Chunking bounds RAM for the larger fascicle dataset.
    for start in range(0, len(f), 256):
        corr = z[start:start+256] @ z.T
        aa, bb = np.nonzero(corr >= near_corr)
        for ai, j in zip(aa, bb):
            i = start + int(ai); j = int(j)
            if j >= i or shapes[i] != shapes[j] or min(norms[i,0], norms[j,0]) < .5:
                continue
            rmse = float(np.sqrt(np.mean((t[i]-t[j])**2)))
            if rmse <= .015:
                uf.join(i,j); links.append(dict(a=f.iloc[i]['name'], b=f.iloc[j]['name'], corr=float(corr[ai,j]), rmse=rmse))
    roots = [uf.root(i) for i in range(len(f))]
    roots = {r:j for j,r in enumerate(sorted(set(roots)))}
    f['group_id'] = [roots[uf.root(i)] for i in range(len(f))]
    f['duplicate_id'] = [same_image.root(i) for i in range(len(f))]
    sizes = f.groupby('group_id').size()
    if len(sizes) < n_folds or sizes.max() > .45 * len(f):
        raise RuntimeError(f'Split has {len(sizes)} independent groups, largest {sizes.max()}/{len(f)}. '
                           'Provide actual video/subject groups; do not silently split connected duplicates.')
    rng = np.random.default_rng(seed)
    # Largest-first balanced assignment; shuffle breaks ties reproducibly.
    order = list(sizes.index); rng.shuffle(order); order.sort(key=lambda g:-int(sizes[g]))
    counts = np.zeros(n_folds, int); allocation = {}
    for g in order:
        candidates = np.flatnonzero(counts == counts.min())
        fold = int(rng.choice(candidates)); allocation[g] = fold; counts[fold] += int(sizes[g])
    f['fold'] = f.group_id.map(allocation).astype(int)
    for fold in range(n_folds):
        a = set(f.loc[f.fold==fold,'group_id']); b = set(f.loc[f.fold!=fold,'group_id'])
        assert not a & b
    audit = pd.DataFrame(rows)
    f = f.merge(audit, on='name', validate='one_to_one')
    identity = digest({'version':DATA_VERSION, 'seed':seed, 'near_corr':near_corr,
                       'rows':f[['name','image_sha256','mask_sha256','fold','group_id']].to_dict('records')})
    meta = dict(identity=identity, samples=len(f), groups=len(sizes), largest_group=int(sizes.max()),
                exact_or_mirrored_duplicates=len(f)-f.duplicate_id.nunique(), near_duplicate_links=len(links),
                folds={str(k):int(v) for k,v in f.fold.value_counts().sort_index().items()},
                mask_shape_mismatch=int(((f.h!=f.mask_h)|(f.w!=f.mask_w)).sum()),
                caution='Blocks and duplicate similarity are proxies, not verified subject identifiers.')
    fp = out_dir/'FOLDS.csv'; mp=out_dir/'DATA.json'
    if mp.exists() and json.loads(mp.read_text())['identity'] != identity:
        raise RuntimeError('Training data or split changed. Use a new RUN_NAME; previous checkpoints are preserved.')
    f.to_csv(fp,index=False); pd.DataFrame(links,columns=['a','b','corr','rmse']).to_csv(out_dir/'NEAR_DUPLICATES.csv',index=False)
    atomic_json(mp,meta); log('split:',meta)
    return f, meta


def build_cache(frame, directory, drive_tar, task, cfg, log=print):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True); drive_tar=Path(drive_tar)
    sig=digest({'version':DATA_VERSION,'task':task,'side':cfg['side'],
                'targets':{k:cfg.get(k) for k in ('tube_threshold','band_px')},
                'data':frame[['name','image_sha256','mask_sha256']].to_dict('records')})
    manifest=directory/'CACHE.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity'] != sig:
        raise RuntimeError('Local cache differs from current source data')
    if not manifest.exists() and drive_tar.is_file():
        with tarfile.open(drive_tar) as tf: tf.extractall(directory,filter='data')
        if not manifest.is_file() or json.loads(manifest.read_text())['identity'] != sig:
            raise RuntimeError('Persistent cache differs from current source data')
        log('cache restored:',drive_tar.name)
    for i,r in enumerate(frame.itertuples(index=False)):
        dest=directory/f'{r.name}.npz'
        if dest.exists(): continue
        gray=read_gray(r.image); mask=read_binary(r.mask)
        image=letterbox_image(gray,cfg['side'])[0]
        if task=='apo':
            if mask.shape!=gray.shape:
                mask=cv2.resize(mask.astype(np.float32),(gray.shape[1],gray.shape[0]),interpolation=cv2.INTER_LINEAR)
            target,valid=letterbox_soft(mask,cfg['side']); extra={}
        else:
            target,angles,valid,_=r71_train.make_targets(gray.shape,mask,cfg['side'],cfg['tube_threshold'],cfg['band_px'])
            extra={'ang':angles}
        tmp=dest.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,image=np.rint(255*image).astype(np.uint8),
                            target=np.rint(255*target).astype(np.uint8),valid=valid,**extra)
        os.replace(tmp,dest)
        if (i+1)%100==0:log(f'cache {task}: {i+1}/{len(frame)}')
    atomic_json(manifest,{'identity':sig,'task':task,'count':len(frame),'side':cfg['side']})
    if not drive_tar.exists():
        drive_tar.parent.mkdir(parents=True,exist_ok=True);tmp=drive_tar.with_suffix('.tmp.tar')
        with tarfile.open(tmp,'w') as tf:
            tf.add(manifest,arcname=manifest.name)
            for n in frame.name:tf.add(directory/f'{n}.npz',arcname=f'{n}.npz')
        os.replace(tmp,drive_tar); log('cache saved:',drive_tar.name,round(drive_tar.stat().st_size/1e6),'MB')
    return sig


def edge_coordinates(target, valid):
    """GT inner contours, calculated AFTER image/label augmentation in the same coordinate frame."""
    h,w=target.shape; coords=np.zeros((2,w),np.float32); visible=np.zeros((2,w),np.float32)
    pair,labels=aponeurosis_lines((target>=.5)&(valid>0),return_labels=True)
    if pair is None:return coords,visible
    for k,name in enumerate(('upper','lower')):
        m=labels==pair[name]['label'];has=m.any(0)
        if k==0: y=h-1-m[::-1].argmax(0)
        else:y=m.argmax(0)
        x=np.flatnonzero(has);yy=y[has]
        if k==0:
            yn=np.clip(yy+1,0,h-1);a=target[yy,x];b=target[yn,x]
            sub=yy+np.clip((a-.5)/np.maximum(a-b,1e-6),0,1)
        else:
            yp=np.clip(yy-1,0,h-1);a=target[yp,x];b=target[yy,x]
            sub=yp+np.clip((.5-a)/np.maximum(b-a,1e-6),0,1)
        coords[k,x]=sub;visible[k,x]=1
    visible[:,:2]=0;visible[:,-2:]=0
    return coords,visible


class Dataset(torch.utils.data.Dataset):
    def __init__(self,frame,cache,task,cfg,train,fold):
        self.frame=frame.reset_index(drop=True);self.cache=Path(cache);self.task=task;self.cfg=cfg
        self.train=train;self.fold=fold;self.epoch=0
    def __len__(self):return len(self.frame)
    def __getitem__(self,i):
        with np.load(self.cache/f'{self.frame.iloc[i]["name"]}.npz',allow_pickle=False) as z:
            x=z['image'].astype(np.float32)/255;t=z['target'].astype(np.float32)/255;v=z['valid'].astype(np.uint8)
            ang=z['ang'] if self.task=='fasc' else None
        rng=np.random.default_rng([self.cfg['seed'],self.fold,self.epoch,i])
        if self.task=='apo':
            if self.train:x,t,v=apo_augment(x,t,v,rng,self.cfg)
            y,has=edge_coordinates(t,v);edge=_edge_weight(t,v.astype(np.float32),self.cfg['edge_weight'])
            extras=dict(edge=edge[None],edge_y=y,edge_visible=has)
        else:
            if self.train:x,t,theta,v=r71_train.augment(x,t,ang,v,rng,self.cfg)
            else:theta=np.where(ang!=r71_train.NO_ANGLE,ang.astype(np.float32)/100,np.nan)
            t,c,s,band=r71_train.targets(t,theta,v,self.cfg);extras=dict(cs=np.stack([c,s]),band=band[None])
        out=dict(x=x[None],target=t[None],valid=v[None],**extras)
        return {k:torch.from_numpy(np.ascontiguousarray(a,dtype=np.float32)) for k,a in out.items()}
