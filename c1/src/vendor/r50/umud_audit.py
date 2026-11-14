"""R17 training data audit and persistent cache. SPDX-License-Identifier: GPL-3.0-or-later"""
from umud_core import *
import concurrent.futures, shutil

def audit_training(project,run_dir,assets,side=768,workers=4):
    project=Path(project);run_dir=Path(run_dir);assets=Path(assets);cache=run_dir/'cache';cache.mkdir(parents=True,exist_ok=True)
    folds=pd.read_csv(assets/'folds.csv');pairs=pd.read_csv(assets/'fasc_pairs.csv')
    groups=json.loads((assets/'group_gate.json').read_text())
    assert groups['VISUAL_SEQUENCE_GROUPED_FOLDS_READY'] and groups['exact_pixel_duplicate_cross_fold_n']==0
    assert len(pairs)==2761 and len(folds)==3809
    f=folds[folds.branch=='fascicle'].copy()
    assert not f.file_name.duplicated().any() and set(f.file_name)==set(pairs.name)
    assert (f.groupby('pixel_sha256').fold.nunique()==1).all()
    assert (f.groupby('group_id').fold.nunique()==1).all()
    # The same filename in the apo branch is NOT a cross-task pairing key.
    rows=[]
    for sha,g in f.groupby('pixel_sha256',sort=True):
        r=g.iloc[0];names=sorted(g.file_name.tolist())
        rows.append(dict(sample_id=sha,group_id=int(r.group_id),fold=int(r.fold),file_name=r.file_name,
                         image=str(project/'data/competition_train_v1/fasc_images'/r.file_name),
                         mask_paths=json.dumps([str(project/'data/competition_train_v1/fasc_masks'/name) for name in names]),
                         annotation_n=len(names)))
    samples=pd.DataFrame(rows);assert len(samples)==2046
    for p in samples.image:
        if not Path(p).is_file():raise FileNotFoundError(p)
    config=dict(version=VERSION,side=side,dilate=1,registration='STRETCH_FULL_NORMALIZED_COORDINATES',target='source_skeleton_forward_splat',
                folds_sha=sha256(assets/'folds.csv'),pairs_sha=sha256(assets/'fasc_pairs.csv'))
    identity=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if (run_dir/'cache_config.json').exists():
        if json.loads((run_dir/'cache_config.json').read_text())['identity']!=identity:raise RuntimeError('Cache settings changed; select a new RUN_NAME')
    atomic_json(run_dir/'cache_config.json',dict(identity=identity,config=config))
    def one(row):
        dst=cache/(row.sample_id+'.npz');qpath=cache/(row.sample_id+'.json')
        if dst.exists() and qpath.exists():
            q=json.loads(qpath.read_text())
            if q.get('identity')==identity:
                with np.load(dst,allow_pickle=False) as z:
                    assert z['image'].shape==(side,side) and z['line'].shape==(side,side)
                return q
        a=read_gray(row.image)
        if pixel_hash(a)!=row.sample_id:raise RuntimeError(f'Pixels changed since R3A2: {row.image}')
        paths=json.loads(row.mask_paths)
        target,line,valid=fasc_target(paths,a.shape,side)
        x,_=letterbox_image(a,side)
        mismatches=sum(read_binary(p).shape!=a.shape for p in paths)
        q=dict(identity=identity,sample_id=row.sample_id,fold=int(row.fold),group_id=int(row.group_id),height=a.shape[0],width=a.shape[1],
               annotations=len(paths),mask_shape_mismatch=mismatches,positive_pixels=int(target.sum()),line_pixels=int(line.sum()),
               foreground_fraction=float(target.sum()/max(valid.sum(),1)),empty=bool(line.sum()==0),
               image_orientation_deg=None,mask_sha256=[sha256(p) for p in paths])
        angle=image_orientation(line)
        q['image_orientation_deg']=float(angle) if np.isfinite(angle) else None
        tmp=dst.with_suffix('.tmp.npz');np.savez_compressed(tmp,image=np.rint(x*255).astype(np.uint8),target=target,line=line,valid=valid);os.replace(tmp,dst)
        atomic_json(qpath,q);return q
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for k,result in enumerate(pool.map(one,list(samples.itertuples(index=False))),1):
            results.append(result)
            if k%50==0:print('cached',k,'/',len(samples),flush=True)
    qc=pd.DataFrame(results);samples=samples.merge(qc[['sample_id','empty','foreground_fraction']],on='sample_id',validate='one_to_one')
    samples.to_csv(run_dir/'samples.csv',index=False);qc.to_csv(run_dir/'target_qc.csv',index=False)
    counts=samples.groupby('fold').agg(images=('sample_id','size'),groups=('group_id','nunique'),empty=('empty','sum'))
    counts.to_csv(run_dir/'fold_counts.csv')
    gate=dict(status='PASS',unique_fasc_images=len(samples),source_fasc_rows=len(f),exact_duplicate_rows_removed=len(f)-len(samples),
              cache_identity=identity,side=side,source_mask_shape_mismatch_count=int(qc.mask_shape_mismatch.sum()),empty_targets=int(qc['empty'].sum()),
              train_folds=[1,2,3],validation_fold=4,outer_fold=0,groups_are_subject_ids=False,
              note='Folds 0 and 1 have few large visual groups; report each fold and group count, not only pooled averages.')
    if qc['empty'].mean()>.01:gate['status']='BLOCKED_EMPTY_TARGETS'
    atomic_json(run_dir/'AUDIT.json',gate)
    montage=[]
    for fold in range(5):
        r=samples[samples.fold==fold].iloc[0]
        with np.load(cache/(r.sample_id+'.npz')) as z:
            rgb=np.repeat(z['image'][...,None],3,axis=-1);rgb[z['line']>0]=[255,50,50]
            rgb=cv2.resize(rgb,(320,320));cv2.putText(rgb,f'fold {fold}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2);montage.append(rgb)
    cv2.imwrite(str(run_dir/'training_overlay.png'),cv2.cvtColor(np.concatenate(montage,axis=1),cv2.COLOR_RGB2BGR))
    print(counts);print(gate)
    if gate['status']!='PASS':raise RuntimeError(gate)
    return samples

def stage_cache(samples,source,local):
    source=Path(source);local=Path(local);local.mkdir(parents=True,exist_ok=True)
    def one(sid):
        src=source/(sid+'.npz');dst=local/src.name
        if not dst.exists() or dst.stat().st_size!=src.stat().st_size:
            tmp=dst.with_suffix('.tmp.npz');shutil.copyfile(src,tmp);os.replace(tmp,dst)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for k,_ in enumerate(pool.map(one,samples.sample_id),1):
            if k%250==0:print('local cache',k,'/',len(samples),flush=True)
    return local

def audit_test(project,run_dir,registry):
    folder=Path(project)/'data/competition_test_v1';registry=pd.read_csv(registry);rows=[];thumbs=[]
    for r in registry.itertuples(index=False):
        path=folder/r.image_id
        if not path.is_file():raise FileNotFoundError(path)
        a=read_gray(path);x=normalize(a);ppm=float(r.SCALE_PX_PER_CM_FINAL)/10
        rows.append(dict(image_id=r.image_id,height=a.shape[0],width=a.shape[1],dtype=str(a.dtype),px_per_mm=ppm,
                         nonblack_fraction=float((x>.03).mean()),pixel_sha256=pixel_hash(a)))
    out=pd.DataFrame(rows);assert len(out)==309 and out.image_id.is_unique
    out.to_csv(Path(run_dir)/'test_image_audit.csv',index=False)
    png=out[out.image_id.str.lower().str.endswith('.png')]
    for name in png.image_id.iloc[np.linspace(0,len(png)-1,min(8,len(png))).astype(int)]:
        a=read_gray(folder/name);tile=np.repeat(np.rint(normalize(a)*255).astype(np.uint8)[...,None],3,axis=-1)
        tile=cv2.resize(tile,(360,240));cv2.putText(tile,name,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1);thumbs.append(tile)
    if thumbs:cv2.imwrite(str(Path(run_dir)/'png_contact_sheet.png'),np.concatenate([np.concatenate(thumbs[:4],axis=1),np.concatenate(thumbs[4:8],axis=1)],axis=0) if len(thumbs)==8 else np.concatenate(thumbs,axis=1))
    return out
