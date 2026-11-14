"""Fast startup checks; synthetic examples validate contracts, not model quality.
SPDX-License-Identifier: GPL-3.0-or-later
"""
import numpy as np
import cv2
import torch
from r76_data import edge_coordinates
from r76_train import new_model, loss_fn, APO_CONFIG, FASC_CONFIG
from r76_export import guarded_probability, lex5_features
from r45_core import fasc_lines, clean_json
from r45_decode import decode_image


def run(log=print):
    h,w=96,160;apo=np.zeros((h,w),np.float32);p=np.zeros_like(apo)
    apo[15:21]=1;apo[70:77]=1
    for k in range(-80,180,28):cv2.line(p,(k,20),(k+120,70),1,1)
    c=np.ones_like(p);s=np.zeros_like(p)
    y,has=edge_coordinates(apo,np.ones_like(apo));assert np.allclose(y[:,20],[20.5,69.5])
    lines=fasc_lines(p>=.5,min_length=max(12,.025*w))
    a=decode_image(p,c,s,{'base':apo},10,False)
    b=decode_image(p,c,s,{'base':apo},10,False,fascicle_lines=lines)
    assert clean_json(a)==clean_json(b),'Baseline decoder changed'
    safe,info=guarded_probability(apo,apo);assert np.array_equal(safe,apo)
    assert lex5_features(p,c,s,apo,10)['lex5_n']>0
    log('PASS: subpixel edge coordinates, legacy decoder equality, anatomical guard, fixed lex5')
    # Small real-network forwards and backward passes; no pretrained downloads.
    old_threads=torch.get_num_threads();torch.set_num_threads(min(old_threads,2))
    try:
        size=64;t=np.zeros((size,size),np.float32);t[12:17]=1;t[45:50]=1
        y,has=edge_coordinates(t,np.ones_like(t))
        for task,cfg in [('apo',APO_CONFIG),('fasc',FASC_CONFIG)]:
            m=new_model(task).train();x=torch.rand(1,1,size,size);out=m(x)
            batch=dict(x=x,target=torch.from_numpy(t)[None,None],valid=torch.ones(1,1,size,size))
            if task=='apo':batch.update(edge=torch.ones_like(batch['valid']),edge_y=torch.from_numpy(y)[None],edge_visible=torch.from_numpy(has)[None])
            else:batch.update(cs=torch.cat([torch.ones_like(batch['valid']),torch.zeros_like(batch['valid'])],1),band=torch.from_numpy(t)[None,None])
            loss,_=loss_fn(task,out,batch,cfg,0);loss.backward()
            assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
            del m,out,loss;log('PASS:',task,'network forward/backward')
    finally:torch.set_num_threads(old_threads)
    return {'status':'PASS','meaning':'Code/geometry checks only, not a GPU benchmark or leaderboard result'}
