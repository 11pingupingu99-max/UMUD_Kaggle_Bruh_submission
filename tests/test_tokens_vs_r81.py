"""Regression: the extracted token pipeline must give exactly the same submission as notebook R81
for B10 and for every probe variant submitted since R79 (AQ, AR, AS, AT)."""
import json, sys
sys.path.insert(0, "/home/claude/r77"); sys.path.insert(0, "/home/claude/umud_final")
from pathlib import Path
import numpy as np, pandas as pd, nbrun
from umud_final.tokens import run_tokens

FIX = Path("/home/claude/umud_final/tests/fixtures")
R81 = "/home/claude/r81/UMUD_R81_KAGGLE.ipynb"

def load_inputs():
    t = json.loads((FIX / "tables_r81.json").read_text())
    t["base"] = pd.read_csv(FIX / "base_r23a.csv")
    return t

variants = {"B10": json.load(open("/home/claude/r82/probes_AT.json"))["B10"], "B9 (R81 default)": nbrun.default_variant(R81)}
for f in ("/home/claude/r79/probes_AQ.json", "/home/claude/r80/probes_AR.json", "/home/claude/r81/probes_AS.json", "/home/claude/r82/probes_AT.json"):
    for k, v in json.load(open(f)).items():
        variants[f"{Path(f).stem}:{k}"] = v
inp = load_inputs()
n_ok = 0
for name, v in variants.items():
    a, _, _ = nbrun.run(R81, v)
    b, _, _ = run_tokens(inp, v)
    assert a.equals(b), f"{name}: differs (max |d| {np.abs(a[['pa_deg','fl_mm','mt_mm']].values - b[['pa_deg','fl_mm','mt_mm']].values).max()})"
    n_ok += 1
ref = pd.read_csv(FIX / "B10_submission.csv")
b10, _, _ = run_tokens(inp, variants["B10"])
assert (b10.image_id == ref.image_id).all() and np.allclose(b10[['pa_deg','fl_mm','mt_mm']].values, ref[['pa_deg','fl_mm','mt_mm']].values, rtol=0, atol=1e-9)
print(f"PASS: {n_ok} wariantow identycznych z R81 (B10, B9 i sondy AQ/AR/AS/AT)")
