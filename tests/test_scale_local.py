"""Scale reader vs the frozen registry on the test images available locally (all six layouts)."""
import sys; sys.path.insert(0, "/home/claude/umud_final")
import numpy as np, pandas as pd
from PIL import Image
from umud_final.scale import read_scale
reg = pd.read_csv('/home/claude/c0_in/r4a/challenge_scale_registry_309.csv').set_index('image_id')
src = pd.read_csv('/home/claude/c1/local_images.csv', index_col=0).path.to_dict()
n = 0; lays = set()
for iid, p in sorted(src.items()):
    try:
        im = Image.open(p); im.load()
    except Exception:
        continue
    r = read_scale(im, is_png=iid.lower().endswith('.png'))
    assert r.layout == reg.loc[iid, 'layout'], (iid, r.layout)
    assert abs(r.px_per_cm - reg.loc[iid, 'SCALE_PX_PER_CM_FINAL']) < 1e-6, (iid, r.px_per_cm, reg.loc[iid, 'SCALE_PX_PER_CM_FINAL'], r.note)
    n += 1; lays.add(r.layout)
print(f"PASS: skala = rejestr na {n} obrazach, uklady: {sorted(lays)}")
