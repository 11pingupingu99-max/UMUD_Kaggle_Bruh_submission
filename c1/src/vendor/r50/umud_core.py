"""UMUD R17 v1.0.0. Geometry, data and inference; no leaderboard fit.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from pathlib import Path
import os, json, hashlib, math, random, time
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

VERSION = 'R17_v1.0.0'
TARGETS = ['pa_deg', 'fl_mm', 'mt_mm']

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False));os.replace(tmp,path)

def read_gray(path):
    a=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if a is None: raise ValueError(f'Unreadable image: {path}')
    if a.ndim==3: a=cv2.cvtColor(a[:,:,:3],cv2.COLOR_BGR2GRAY)
    if a.ndim!=2 or not np.isfinite(a).all(): raise ValueError(f'Invalid image: {path}')
    return a

def pixel_hash(a):
    h=hashlib.sha256();h.update(str(tuple(a.shape)).encode());h.update(str(a.dtype).encode())
    h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()

def normalize(a):
    # Same policy for train/test; uint8 keeps acquired intensity, 16-bit uses robust range.
    if a.dtype==np.uint8: return a.astype(np.float32)/255.
    x=a.astype(np.float32);lo,hi=np.percentile(x,[.5,99.5])
    return np.clip((x-lo)/max(float(hi-lo),1e-6),0,1)

def read_binary(path):
    a=read_gray(path)
    if a.max()<=1: return (a>0).astype(np.uint8)
    return (a>float(a.max())*.5).astype(np.uint8)

def letterbox_geometry(hw,side):
    h,w=hw;r=min(side/h,side/w);nh=max(1,round(h*r));nw=max(1,round(w*r))
    return nh,nw,(side-nh)//2,(side-nw)//2

def letterbox_image(a,side):
    nh,nw,y,x=letterbox_geometry(a.shape,side)
    z=np.zeros((side,side),np.float32)
    z[y:y+nh,x:x+nw]=cv2.resize(normalize(a),(nw,nh),interpolation=cv2.INTER_AREA if max(a.shape)>side else cv2.INTER_LINEAR)
    return z,(nh,nw,y,x)

def splat_mask(mask,output_hw):
    """Normalized-coordinate mask registration, with no thin-pixel deletion.
    Skeleton coordinates are mapped once; nearby source skeleton edges are drawn
    in the output so both downsampling and upsampling retain line continuity.
    """
    h,w=output_hw;mh,mw=mask.shape
    sk=skeletonize(mask>0);ys,xs=np.nonzero(sk)
    out=np.zeros((h,w),np.uint8)
    if not len(xs): return out
    ox=np.clip(np.rint((xs+.5)*w/mw-.5).astype(int),0,w-1)
    oy=np.clip(np.rint((ys+.5)*h/mh-.5).astype(int),0,h-1)
    out[oy,ox]=1
    coords={(int(y),int(x)):(int(xx),int(yy)) for y,x,xx,yy in zip(ys,xs,ox,oy)}
    for (y,x),q in coords.items():
        for dy,dx in ((0,1),(1,-1),(1,0),(1,1)):
            p=coords.get((y+dy,x+dx))
            if p is not None: cv2.line(out,q,p,1,1,cv2.LINE_8)
    return out

def fasc_target(mask_paths,hw,side,dilate=1):
    nh,nw,y,x=letterbox_geometry(hw,side)
    line=np.zeros((side,side),np.uint8)
    for path in mask_paths:
        line[y:y+nh,x:x+nw] |= splat_mask(read_binary(path),(nh,nw))
    target=cv2.dilate(line,np.ones((2*dilate+1,2*dilate+1),np.uint8)) if dilate else line.copy()
    valid=np.zeros_like(line);valid[y:y+nh,x:x+nw]=1
    target*=valid
    return target,line,valid

def restore_probability(p,hw):
    nh,nw,y,x=letterbox_geometry(hw,p.shape[-1])
    return cv2.resize(p[y:y+nh,x:x+nw],(hw[1],hw[0]),interpolation=cv2.INTER_LINEAR)

def axial_diff(a,b):
    return np.abs((np.asarray(a)-np.asarray(b)+90)%180-90)

def fit_line(points):
    p=np.asarray(points,np.float64)
    if len(p)<3: return None
    c=p.mean(0);_,s,v=np.linalg.svd(p-c,full_matrices=False);d=v[0]
    if d[0]<0:d=-d
    if abs(d[0])<1e-8:return None
    a=float(d[1]/d[0]);b=float(c[1]-a*c[0]);t=(p-c)@d
    return dict(a=a,b=b,angle=math.degrees(math.atan2(d[1],d[0])),
                length=float(np.ptp(t)),residual=float(np.sqrt(np.mean(((p-c)@v[1])**2))),
                x0=float(p[:,0].min()),x1=float(p[:,0].max()),n=len(p))

def fasc_lines(mask,min_length=15):
    lab,n=ndi.label(skeletonize(np.asarray(mask)>0),structure=np.ones((3,3)))
    out=[]
    for i,sl in enumerate(ndi.find_objects(lab),1):
        if sl is None: continue
        y,x=np.nonzero(lab[sl]==i)
        if len(x)<max(6,min_length*.5):continue
        z=fit_line(np.c_[x+sl[1].start,y+sl[0].start])
        if z and z['length']>=min_length and z['residual']<max(3,.05*z['length']):out.append(z)
    return out

def image_orientation(mask):
    lines=fasc_lines(mask,min_length=max(8,mask.shape[1]*.02))
    if not lines:return float('nan')
    a=np.radians([z['angle']*2 for z in lines]);w=np.array([z['length'] for z in lines])
    return math.degrees(math.atan2(np.sum(w*np.sin(a)),np.sum(w*np.cos(a))))/2

def segmentation_metrics(prob,target,line,valid,threshold=.5,tolerance=2):
    pr=(prob>=threshold)&(valid>0);gt=(target>0)&(valid>0)
    inter=np.count_nonzero(pr & gt);dice=(2*inter+1)/(pr.sum()+gt.sum()+1)
    ps=skeletonize(pr);gs=(line>0)&(valid>0)
    if not gs.any():return dict(dice=float(dice),center_f1=float(not ps.any()),angle_error=None,angle_valid=False)
    if not ps.any():return dict(dice=float(dice),center_f1=0.,angle_error=None,angle_valid=False)
    dtg=ndi.distance_transform_edt(~gs);dtp=ndi.distance_transform_edt(~ps)
    precision=float((dtg[ps]<=tolerance).mean());recall=float((dtp[gs]<=tolerance).mean())
    f1=2*precision*recall/max(precision+recall,1e-9)
    ap=image_orientation(pr);ag=image_orientation(gs);ok=bool(np.isfinite([ap,ag]).all())
    return dict(dice=float(dice),center_f1=f1,angle_error=float(axial_diff(ap,ag)) if ok else None,angle_valid=ok)

def score_arrays(truth,pred):
    truth=np.asarray(truth,float);pred=np.asarray(pred,float)
    if truth.shape!=pred.shape or truth.ndim!=2 or truth.shape[1]!=3:raise ValueError('Expected matching N x 3 arrays')
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():raise ValueError('Cannot silently drop failed predictions')
    error=np.abs(truth-pred);tol=np.array([6.,12.,3.])
    return dict(score_main=float((error.mean(0)/tol).mean()),MAE=dict(zip(TARGETS,error.mean(0).tolist())),
                note='Main MAE term only; official tie breakers are not reimplemented.')

def aponeurosis_lines(mask):
    """Select two long bands; measure centers AND their inner edges explicitly."""
    mask=np.asarray(mask)>0;h,w=mask.shape
    lab,n=ndi.label(mask,structure=np.ones((3,3)));components=[]
    for i,sl in enumerate(ndi.find_objects(lab),1):
        if sl is None:continue
        yy,xx=np.nonzero(lab[sl]==i);yy=yy+sl[0].start;xx=xx+sl[1].start
        if len(xx)<max(20,.0002*h*w) or np.ptp(xx)<.2*w:continue
        cols=np.unique(xx);center=[];top=[];bottom=[]
        for x in cols:
            ys=yy[xx==x];center.append(np.mean(ys));top.append(np.min(ys));bottom.append(np.max(ys))
        center_line=fit_line(np.c_[cols,center])
        if not center_line or abs(center_line['angle'])>45:continue
        components.append(dict(center=center_line,top=fit_line(np.c_[cols,top]),bottom=fit_line(np.c_[cols,bottom]),
            width=float(np.ptp(xx)),area=int(len(xx)),mid=float(np.mean(yy))))
    pairs=[]
    for i,u in enumerate(components):
        for l in components[i+1:]:
            if u['mid']>l['mid']:u0,l0=l,u
            else:u0,l0=u,l
            x0=max(u0['center']['x0'],l0['center']['x0']);x1=min(u0['center']['x1'],l0['center']['x1'])
            if x1-x0<.2*w:continue
            xs=np.linspace(x0,x1,31);a=u0['bottom'];b=l0['top']
            gap=(b['a']*xs+b['b'])-(a['a']*xs+a['b'])
            if gap.min()<=.025*h or gap.max()>=.85*h:continue
            quality=(x1-x0)/(1+u0['center']['residual']+l0['center']['residual'])
            pairs.append((quality,u0,l0,x0,x1))
    if not pairs:return None
    pairs.sort(key=lambda z:z[0],reverse=True)
    q,u,l,x0,x1=pairs[0]
    return dict(upper=u,lower=l,x0=x0,x1=x1,candidate_pairs=len(pairs),quality=q)

def measure_geometry(apo_prob,fasc_prob,px_per_mm,apo_thr=.5,fasc_thr=.5):
    """All intersections in native coordinates. Square-pixel scale is explicit.
    Extrapolation is allowed up to one image width beyond each image boundary;
    the ratio of total to visible fascicle length is recorded, not silently clipped.
    """
    if not np.isfinite(px_per_mm) or px_per_mm<=0:raise ValueError('Invalid px_per_mm')
    if apo_prob.shape!=fasc_prob.shape:raise ValueError('Probability maps must share native coordinates')
    h,w=apo_prob.shape;pair=aponeurosis_lines(apo_prob>=apo_thr)
    out=dict(pa_deg=None,fl_mm=None,mt_mm=None,mt_center_mm=None,band_correction_mm=None,
             n_fasc=0,apo_pairs=0,extrapolation_ratio=None,status='NO_APO_PAIR')
    if pair is None:return out
    u,l=pair['upper'],pair['lower'];xs=np.linspace(pair['x0'],pair['x1'],101)
    def yy(z):return z['a']*xs+z['b']
    inner_gap=yy(l['top'])-yy(u['bottom']);center_gap=yy(l['center'])-yy(u['center'])
    out.update(mt_mm=float(np.mean(inner_gap)/px_per_mm),mt_center_mm=float(np.mean(center_gap)/px_per_mm),
               band_correction_mm=float(np.mean(center_gap-inner_gap)/px_per_mm),apo_pairs=pair['candidate_pairs'],status='NO_FASCICLE')
    arr=[]
    for f in fasc_lines(fasc_prob>=fasc_thr,min_length=max(12,.025*w)):
        endpoints=[]
        for a in (u['bottom'],l['top']):
            den=f['a']-a['a']
            if abs(den)<1e-6:break
            x=(a['b']-f['b'])/den;y=f['a']*x+f['b'];endpoints.append((x,y))
        if len(endpoints)!=2:continue
        p,q=np.asarray(endpoints)
        if not (-w<=p[0]<=2*w and -w<=q[0]<=2*w):continue
        xm=(f['x0']+f['x1'])/2;ym=f['a']*xm+f['b']
        if not (u['bottom']['a']*xm+u['bottom']['b']<=ym<=l['top']['a']*xm+l['top']['b']):continue
        pa=float(axial_diff(f['angle'],l['top']['angle']))
        length=float(np.linalg.norm(p-q));fl=length/px_per_mm
        if not (1<pa<85 and 0<fl<500):continue
        arr.append((pa,fl,length/max(f['length'],1)))
    if arr:
        med=np.median(np.asarray(arr),axis=0)
        out.update(pa_deg=float(med[0]),fl_mm=float(med[1]),n_fasc=len(arr),extrapolation_ratio=float(med[2]),status='OK')
    return out

def compose_candidate(anchor,measurements,mode):
    """Preserve untouched CSV field strings exactly, including all MT values."""
    if mode not in ('pa_only','fl_only','pafl'):raise ValueError(mode)
    if anchor.image_id.duplicated().any() or measurements.image_id.duplicated().any():raise ValueError('Duplicate IDs')
    if set(anchor.image_id)!=set(measurements.image_id):raise ValueError('Image ID set mismatch')
    out=anchor.copy();m=measurements.set_index('image_id').loc[out.image_id]
    cols=['pa_deg'] if mode=='pa_only' else ['fl_mm'] if mode=='fl_only' else ['pa_deg','fl_mm']
    counts={}
    for col in cols:
        vals=pd.to_numeric(m[col],errors='coerce').to_numpy(float)
        ok=np.isfinite(vals)&(vals>0);counts[col]=int(ok.sum())
        out.loc[ok,col]=[f'{v:.10f}' for v in vals[ok]]
    for col in set(TARGETS)-set(cols):assert out[col].equals(anchor[col])
    if not np.isfinite(out[TARGETS].astype(float).to_numpy()).all():raise ValueError('Invalid candidate')
    return out,counts

def geometry_selftest():
    h,w=400,800;ppm=10.;apo=np.zeros((h,w),np.uint8);fa=np.zeros_like(apo)
    apo[78:83,20:780]=1;apo[238:243,20:780]=1
    # One line at 30 degrees, reaching both inner band edges.
    cv2.line(fa,(150,60),(550,291),1,2)
    z=measure_geometry(apo,fa,ppm)
    expected_mt=(238-82)/ppm
    assert z['status']=='OK',z
    assert abs(z['pa_deg']-30)<.7,z
    assert abs(z['mt_mm']-expected_mt)<.1,z
    assert abs(z['fl_mm']-expected_mt/math.sin(math.radians(z['pa_deg'])))<.2,z
    assert abs(z['band_correction_mm']-.4)<.1,z
    mirror=measure_geometry(apo[:,::-1],fa[:,::-1],ppm)
    assert abs(mirror['pa_deg']-z['pa_deg'])<.2
    assert abs(mirror['fl_mm']-z['fl_mm'])<.2
    z2=measure_geometry(apo,fa,ppm*2)
    assert abs(z2['pa_deg']-z['pa_deg'])<1e-6 and abs(z2['fl_mm']*2-z['fl_mm'])<1e-6
    return z
