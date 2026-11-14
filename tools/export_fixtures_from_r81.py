"""Exports the input tables embedded in notebook R81 as regression fixtures (frozen copies of B10's inputs)."""
import json, sys
sys.path.insert(0, "/home/claude/r77")
from pathlib import Path
import numpy as np, nbrun
R81 = "/home/claude/r81/UMUD_R81_KAGGLE.ipynb"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
B10 = json.load(open("/home/claude/r82/probes_AT.json"))["B10"]
s, g, _ = nbrun.run(R81, B10)
g["base"].to_csv(out / "base_r23a.csv", index=False)
tabs = {k: g[k] for k in ("BAND", "PADEC", "FL22", "PNG_MUSCLE", "OR_INS", "V2MT", "V2OV", "R17DEV", "R17ENS", "FX_COLS", "FX")}
tabs["GROUP"] = [str(x) for x in np.asarray(g["GROUP"])]
tabs["GRYSUB"] = [str(x) for x in np.asarray(g["GRYSUB"])]
tabs["BID"] = [int(x) for x in np.asarray(g["BID"])]
(out / "tables_r81.json").write_text(json.dumps(tabs))
s.to_csv(out / "B10_submission.csv", index=False)
print("fixtures:", sorted(p.name for p in out.iterdir()))
