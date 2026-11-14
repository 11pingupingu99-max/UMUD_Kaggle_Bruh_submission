"""R76: resumable measurements from images, frozen baseline controls, fixed lex5 FL.
The exports are candidate features, not an automatic replacement of calibrated B6.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from pathlib import Path
import gc
import json
import time
import numpy as np
import pandas as pd
from r45_core import (sha256, atomic_json, clean_json, read_gray, letterbox_image,
                      restore_map, aponeurosis_components, fasc_lines)
import r45_models
import r45_decode
import r45_infer
import r71_train
import r71_geom
import r76_train
from r76_data import digest

PA_KEYS=('pa_r17_median','pa_local_support_baseline','pa_local_median')


def verify(path,expected=None):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(path)
    got=sha256(path)
    if expected is not None and got!=expected:raise ValueError(f'Checksum mismatch: {path}')
    return {'path':str(path),'sha256':got}


def baseline_registry(project,drive_root):
    project,drive_root=Path(project),Path(drive_root)
    r22=project/'outputs'/'R22_v1_0_0';tc=json.loads((r22/'TRAINING_COMPLETE.json').read_text())
    reg={'r22':[],'v2':[],'f71':[]}
    for d in sorted(tc['checkpoints'],key=lambda x:x['fold']):
        reg['r22'].append(verify(r22/f'fold_{d["fold"]}'/'final.pt',d['checkpoint_sha256']))
    for tag,folder in [('v2',project/'outputs'/'R45T_v1_0_0'/'apo2'),('f71',project/'outputs'/'R71T_v1_0_0'/'T')]:
        for p in sorted(folder.glob('fold_*/DONE.json')):
            d=json.loads(p.read_text());reg[tag].append(verify(p.parent/'final.pt',d['final_sha256']))
    if any(len(reg[k])!=5 for k in ('r22','v2','f71')):
        raise RuntimeError(f'Required baseline: 5 R22, 5 v2, 5 F71 folds; found { {k:len(reg[k]) for k in reg} }')
    reg['r12b']=verify(drive_root/'UMUD_R12B'/'best_dinov2_frozen.pt',
                       '21ec34d0f5dd7664ecab0ca608f976fdf9d2a161e2af2873c21d513cd5ae19ef')
    return reg


def baseline_models(registry,device):
    m={key:[loader(r['path'],device)[0] for r in registry[key]] for key,loader in
       [('r22',r45_models.load_r22),('v2',r45_models.load_apo2),('f71',r71_train.load_f71)]}
    m['r12b']=r45_models.load_r12b(registry['r12b']['path'],device)
    return m


def native_baseline(row,gray,models,cache,identity,device):
    """Float32 local map cache: repeated A1/A3/A5 exports reuse identical baseline maps."""
    key=digest({'baseline':identity,'image':row.image_sha256})
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);p=cache/f'{key}.npz'
    if p.is_file():
        with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    sq=letterbox_image(gray,768)[0];sqf=letterbox_image(gray,1024)[0]
    maps={'r12b':r45_models.predict_r12b(models['r12b'],row.path,device),
          'v2':restore_map(r45_models.predict_apo2(models['v2'],sq,device,True),gray.shape)}
    for key,source,square in [('r22',models['r22'],sq),('f71',models['f71'],sqf)]:
        p0,c,s=r45_models.predict_r22(source,square,device,True)
        for k,a in zip(('p','c','s'),(p0,c,s)):maps[f'{key}_{k}']=restore_map(a,gray.shape)
    tmp=p.with_suffix('.tmp.npz');np.savez_compressed(tmp,**maps);tmp.replace(p)
    return maps


def same_pair(a,b,tolerance=.05):
    if a is None or b is None:return False
    h=a['shape'][0]
    return max(abs(a['m'][k]-b['m'][k]) for k in ('upper_mid_y','lower_mid_y'))<tolerance*h


def guarded_probability(candidate,reference):
    """Keep the baseline anatomical pair. Select matching candidate components when possible;
    fall back to the baseline probability if either member cannot be matched reliably.
    No image IDs, per-image constants, or leaderboard labels enter this rule.
    """
    base=r45_decode.apo_pair_subpixel(reference);raw=r45_decode.apo_pair_subpixel(candidate)
    if base is None:return candidate,{'guard':'baseline_pair_missing','raw_same_pair':False}
    if same_pair(base,raw):return candidate,{'guard':'same_pair','raw_same_pair':True}
    lab,components=aponeurosis_components(candidate>=.5)
    selected=[];h,w=candidate.shape;xm=(base['xa']+base['xb'])/2
    for name in ('uc','lc'):
        a,b=base['fits'][name];target=a*xm+b
        usable=[]
        for c in components:
            ln=c['center'];dist=abs(ln['a']*xm+ln['b']-target)
            if dist<.05*h and c['width']>=.2*w:usable.append((dist,c))
        if not usable:return reference,{'guard':'fallback_missing_component','raw_same_pair':False}
        selected.append(min(usable,key=lambda z:z[0])[1]['label'])
    if selected[0]==selected[1]:return reference,{'guard':'fallback_same_component','raw_same_pair':False}
    prob=np.where(np.isin(lab,selected),candidate,0).astype(np.float32)
    sub=r45_decode.apo_pair_subpixel(prob)
    if not same_pair(base,sub):return reference,{'guard':'fallback_pair_mismatch','raw_same_pair':False}
    return prob,{'guard':'matched_components','raw_same_pair':False}


def lex5_features(p,c,s,apo,ppm,frags=None):
    """The hard lex5 rule supported by the journal; no soft rank, no angular smoothing."""
    if frags is None:frags=r71_geom.frag_lines(p,c,s,.6)
    pair=r71_geom.apo_pair_record(apo)
    pf=r71_geom.protocol_fragments(frags,pair,p.shape[1],p.shape[0])
    chosen=r71_geom.lex_select(pf,5)
    return dict(lex5_fl=float(np.mean([f['fl_px'] for f in chosen])/ppm) if chosen else None,
                lex5_n=len(chosen),fragments=len(frags),protocol_fragments=len(pf))


def _cache(path,signature):
    got={}
    if Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            try:r=json.loads(line)
            except (ValueError,TypeError):continue
            if r.get('_sig')==signature and r.get('status')=='OK':got[r['image_id']]=r
    return got


def derived_features(rec):
    def median(keys):
        a=[rec.get(k) for k in keys];a=[v for v in a if v is not None and np.isfinite(v)]
        return float(np.median(a)) if a else None
    for apo in ('r12b','v2','a76_raw','a76_safe','a76_v2mix'):
        rec[f'pa_med6_{apo}']=median([f'{m}_{apo}_L_{p}' for m in ('r22','f71') for p in PA_KEYS])
    # The calibrated B6 uses a geometric mean of R22/F71 lex5 at fixed threshold.
    keys=[rec.get(f'{m}_r12b_lex5_fl') for m in ('r22','f71')]
    if all(v is not None and np.isfinite(v) and v>0 for v in keys):
        rec['cmb_r22_f71_lex5_fl']=float(np.sqrt(keys[0]*keys[1]))
    v=rec.get('f76_r12b_lex5_fl')
    if v is not None and v>0 and 'cmb_r22_f71_lex5_fl' in rec:
        rec['cmb_r22_f71_f76_lex5_fl']=float((keys[0]*keys[1]*v)**(1/3))
    return rec


def export_stage(run,project,registry,manifests,assets,model_dirs,local_cache,package_sha,
                 device='cuda',deadline=None,log=print):
    run=Path(run);model_dirs={k:Path(v) for k,v in model_dirs.items()}
    paths={task:[p.parent/'final.pt' for p in sorted(folder.glob('fold_*/DONE.json'))] for task,folder in model_dirs.items()}
    if not paths.get('apo'):raise ValueError('Train at least one A76 fold before exporting')
    # Only a fold with an observed and verified DONE marker can enter an ensemble.
    hashes={};sides={}
    for task,ps in paths.items():
        hashes[task]=[];task_sides=[]
        for p in ps:
            d=json.loads((p.parent/'DONE.json').read_text());verify(p,d['final_sha256']);hashes[task].append(d['final_sha256']);task_sides.append(int(d['side']))
        if task_sides:
            if len(set(task_sides))!=1:raise ValueError('Mixed resolutions within an ensemble')
            sides[task]=task_sides[0]
    signature=dict(package=package_sha,baseline=registry,new=hashes,sides=sides,
                   threshold=.6,lex_k=5,tta=True,guard_frac=.05,
                   manifests={k:[(r.image_id,r.image_sha256,float(r.px_per_mm)) for r in v.itertuples()] for k,v in manifests.items()})
    sig=digest(signature);out=run/'exports'/f'A{len(paths["apo"])}_F{len(paths.get("fasc",[]))}_{sig[:10]}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'SUMMARY.json').is_file():log('export already complete:',out.name);return out
    atomic_json(out/'signature.json',signature)
    base=baseline_models(registry,device)
    new={k:[r76_train.load_model(p,device)[0] for p in ps] for k,ps in paths.items()}
    base_id=digest(registry);all_frames={};errors=0
    try:
        for dataset,man in manifests.items():
            cache=out/f'{dataset}.jsonl';got=_cache(cache,sig)
            if cache.is_file() and cache.stat().st_size and not cache.read_bytes().endswith(b'\n'):
                with cache.open('a') as fh:fh.write('\n')
            log(f'{out.name} {dataset}: {len(got)}/{len(man)} cached')
            start=time.time()
            with cache.open('a') as fh:
                for j,r in enumerate(man.itertuples(index=False)):
                    if r.image_id in got:continue
                    if deadline and time.time()>=deadline:raise r76_train.SessionBudgetReached('Export saved per-image; rerun to continue')
                    rec=dict(image_id=r.image_id,_sig=sig)
                    try:
                        gray=read_gray(r.path);maps=native_baseline(r,gray,base,local_cache,base_id,device)
                        raw=restore_map(r76_train.predict(new['apo'],letterbox_image(gray,sides['apo'])[0],'apo',device,True),gray.shape)
                        safe,info=guarded_probability(raw,maps['r12b']);rec.update(info)
                        vv=r45_decode.apo_pair_subpixel(maps['v2']);bb=r45_decode.apo_pair_subpixel(maps['r12b'])
                        mix=.5*(safe+maps['v2']) if same_pair(vv,bb) else safe
                        mix,mi=guarded_probability(mix,maps['r12b'])
                        ap={'r12b':maps['r12b'],'v2':maps['v2'],'a76_raw':raw,'a76_safe':safe,'a76_v2mix':mix}
                        sources={k:tuple(maps[f'{k}_{v}'] for v in ('p','c','s')) for k in ('r22','f71')}
                        if new.get('fasc'):
                            sources['f76']=tuple(restore_map(z,gray.shape) for z in r76_train.predict(new['fasc'],letterbox_image(gray,sides['fasc'])[0],'fasc',device,True))
                        rec['_fragments']={};rec['_apo_pairs']={name:r71_geom.apo_pair_record(prob) for name,prob in ap.items()}
                        for tag,(p,c,s) in sources.items():
                            lines=fasc_lines(p>=.5,min_length=max(12,.025*p.shape[1]))
                            frags=r71_geom.frag_lines(p,c,s,.6);rec['_fragments'][tag]=frags
                            features=r45_decode.decode_image(p,c,s,ap,float(r.px_per_mm),with_streamlines=False,fascicle_lines=lines)
                            rec.update({f'{tag}_{k}':v for k,v in features.items()})
                            for name,prob in ap.items():
                                rec.update({f'{tag}_{name}_{k}':v for k,v in lex5_features(p,c,s,prob,float(r.px_per_mm),frags).items()})
                        rec.update(h=gray.shape[0],w=gray.shape[1],px_per_mm=float(r.px_per_mm),status='OK')
                        derived_features(rec)
                    except Exception as e:
                        rec['status']=f'ERROR:{type(e).__name__}:{str(e)[:300]}'
                    rec=clean_json(rec);fh.write(json.dumps(rec)+'\n');fh.flush();got[r.image_id]=rec
                    if (j+1)%25==0 or j+1==len(man):log(f'{dataset}: {j+1}/{len(man)}, {(time.time()-start)/60:.1f} min')
            frame=pd.DataFrame([got[i] for i in man.image_id])
            if frame.image_id.duplicated().any() or len(frame)!=len(man):raise RuntimeError('Export image coverage mismatch')
            errors+=int((frame.status!='OK').sum())
            for model in ('r22','f71','f76'):
                fs=[{'image_id':r['image_id'],'frag_id':i,'thr':.6,**a} for r in got.values()
                    for i,a in enumerate(r.get('_fragments',{}).get(model,[]))]
                if fs:pd.DataFrame(fs).to_csv(out/f'{dataset}_frags_{model}_t60.csv.gz',index=False,compression='gzip')
            aps=[{'image_id':r['image_id'],'cfg':k,**v} for r in got.values() for k,v in r.get('_apo_pairs',{}).items() if v is not None]
            if aps:pd.DataFrame(aps).to_csv(out/f'{dataset}_apo.csv.gz',index=False,compression='gzip')
            frame=frame.drop(columns=['_sig','_fragments','_apo_pairs'],errors='ignore')
            frame.to_csv(out/f'{dataset}_features.csv',index=False)
            all_frames[dataset]=frame
        if errors:raise RuntimeError(f'{errors} image failures. Inspect JSONL; rerun retries failed images.')
        if 'osf' in all_frames:
            osf_report(all_frames['osf'],Path(assets)/'osf35_full.csv').to_csv(out/'OSF_COMPARISON.csv',index=False)
        # Exact baseline control against the current D5 export, when present.
        for dataset,frame in all_frames.items():
            control=Path(project)/'outputs'/'R71T_v1_0_0'/'D5'/f'{dataset}_decoders.csv'
            if control.is_file():
                repro=control_comparison(frame,pd.read_csv(control));repro.to_csv(out/f'{dataset}_BASELINE_REPRO.csv',index=False)
        atomic_json(out/'SUMMARY.json',dict(status='FEATURE_EXPORT_COMPLETE',a76_folds=len(paths['apo']),f76_folds=len(paths.get('fasc',[])),
                    errors=errors,rows={k:len(v) for k,v in all_frames.items()},signature=sig,
                    public_score='NOT_MEASURED',note='No automatic promotion or Kaggle submission. Scales inherited from project manifests.'))
        atomic_json(run/'LATEST_EXPORT.json',dict(directory=str(out),signature=sig))
        log('export complete:',out)
        return out
    finally:
        del base,new;gc.collect()
        if device=='cuda':
            import torch
            torch.cuda.empty_cache()


def control_comparison(frame,control):
    x=frame.merge(control,on='image_id',suffixes=('','_control'),validate='one_to_one');rows=[]
    cols=[f'{m}_r12b_L_{k}' for m in ('r22','f71') for k in PA_KEYS]
    for col in cols:
        if col not in frame or col not in control:continue
        a=pd.to_numeric(x[col],errors='coerce');b=pd.to_numeric(x[col+'_control'],errors='coerce');ok=np.isfinite(a)&np.isfinite(b)
        e=(a-b)[ok].abs()
        rows.append(dict(column=col,n=int(ok.sum()),median_abs_diff=float(e.median()),p90_abs_diff=float(e.quantile(.9)),
                         fraction_within_005=float((e<=.05).mean())))
    return pd.DataFrame(rows,columns=['column','n','median_abs_diff','p90_abs_diff','fraction_within_005'])


def osf_report(frame,truth_path):
    gt=r45_infer.osf_truth(truth_path);x=frame.merge(gt,on='image_id',validate='one_to_one');rows=[]
    candidates={'pa':[c for c in frame if c.startswith('pa_med6_')],
                'mt':['r22_'+a+'_mt_inner_mean_mm' for a in ('r12b','v2','a76_raw','a76_safe','a76_v2mix')],
                'fl':[c for c in frame if c.endswith('lex5_fl') and not c.endswith('protocol_fragments')]}
    for target,cols in candidates.items():
        existing=[c for c in cols if c in x]
        anchor={'pa':'pa_med6_r12b','mt':'r22_v2_mt_inner_mean_mm','fl':'cmb_r22_f71_lex5_fl'}[target]
        anchor_values=pd.to_numeric(x[anchor],errors='coerce') if anchor in x else pd.Series(np.nan,index=x.index)
        shared=np.isfinite(x[f'gt_{target}'])
        for c in existing:shared=shared & np.isfinite(pd.to_numeric(x[c],errors='coerce'))
        for col in cols:
            if col not in x:continue
            a=pd.to_numeric(x[col],errors='coerce');b=x[f'gt_{target}'];ok=np.isfinite(a)&np.isfinite(b)
            err=(a-b)[ok].to_numpy()
            if len(err)<5:continue
            loo=np.array([err[i]-np.median(np.delete(err,i)) for i in range(len(err))])
            common=(a-b)[shared].to_numpy()
            pair_ok=ok & np.isfinite(anchor_values)
            ep=(a-b)[pair_ok].to_numpy();eb=(anchor_values-b)[pair_ok].to_numpy()
            rows.append(dict(target=target,column=col,n=len(err),mae=float(np.abs(err).mean()),
                             median_bias=float(np.median(err)),loo_centered_mae=float(np.abs(loo).mean()),
                             common_n=len(common),common_mae=float(np.abs(common).mean()) if len(common) else None,
                             common_centered_mae=float(np.abs(common-np.median(common)).mean()) if len(common) else None,
                             anchor=anchor,paired_n=len(ep),paired_mae=float(np.abs(ep).mean()) if len(ep) else None,
                             paired_anchor_mae=float(np.abs(eb).mean()) if len(eb) else None,
                             paired_mae_delta=float((np.abs(ep)-np.abs(eb)).mean()) if len(ep) else None,
                             caution='Auxiliary external comparison; no leaderboard gain inferred'))
    return pd.DataFrame(rows,columns=['target','column','n','mae','median_bias','loo_centered_mae',
                                     'common_n','common_mae','common_centered_mae','anchor','paired_n','paired_mae',
                                     'paired_anchor_mae','paired_mae_delta','caution'])
