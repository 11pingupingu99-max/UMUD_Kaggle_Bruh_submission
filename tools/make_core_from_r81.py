"""Builds umud_final/tokens_core.py from cell 2 of the R81 Kaggle notebook.

Embedded tables are replaced by entries of the INPUTS mapping; groups, gry sub-groups and sequence blocks come
from INPUTS as well (they are computed from the images upstream). Everything else is copied verbatim, so the
token pipeline stays identical to the one that produced the leaderboard submissions.
"""
import json, re, sys
from pathlib import Path

NB = Path(sys.argv[1]); OUT = Path(sys.argv[2])
src = "".join(json.loads(NB.read_text())["cells"][2]["source"])
lines = src.split("\n")

def replace_line(prefix, new):
    hits = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    lines[hits[0]] = new

replace_line("PAYLOAD_SHA256 = ", 'PAYLOAD_SHA256 = INPUTS.get("payload_sha256", "computed")')
replace_line("PAYLOAD = ", "# base predictions come from INPUTS['base'] (computed upstream)")
for name in ("BAND", "PADEC", "FL22", "PNG_MUSCLE", "OR_INS", "V2MT", "V2OV", "R17DEV", "R17ENS", "FX_COLS", "FX"):
    replace_line(f"{name} = json.loads(", f'{name} = INPUTS["{name}"]')
replace_line("FX_TAG = ", 'FX_TAG = INPUTS.get("FX_TAG", "computed")')
replace_line("raw = zlib.decompress(", "# (payload decoding removed)")
replace_line("assert hashlib.sha256(raw).hexdigest() == PAYLOAD_SHA256", "# (payload hash check removed: base is computed)")
replace_line("base = pd.read_csv(io.BytesIO(raw))", 'base = INPUTS["base"][["image_id", "pa_deg", "fl_mm", "mt_mm"]].reset_index(drop=True).copy()')
replace_line("GROUP = np.array([id_group(n, d)", 'GROUP = np.asarray(INPUTS["GROUP"], dtype=object).astype(str)')
replace_line("BID = blocks(base)", 'BID = np.asarray(INPUTS["BID"], dtype=int)')
replace_line("GRYSUB = np.array(", 'GRYSUB = np.asarray(INPUTS["GRYSUB"], dtype=object).astype(str)')
replace_line("_paths = {Path(p).name: p for p in glob.glob(", "_paths = {}  # image checks live in the image stage")
# continuation line of the old _paths comprehension
i = [k for k, l in enumerate(lines) if l.startswith("_paths = {}")][0]
assert "if Path(p).suffix.lower()" in lines[i + 1], lines[i + 1]
lines[i + 1] = ""
replace_line("VARIANT", "VARIANT = VARIANT") if any(l.startswith("VARIANT") for l in lines) else None
header = ('"""Token pipeline of the UMUD submission (generated from notebook R81, cell 2 - do not edit by hand).\n'
          'Executed by umud_final.tokens.run_tokens() with INPUTS (tables) and VARIANT (constants) in the namespace.\n'
          'Licence: GPL-3.0-or-later."""\n')
OUT.write_text(header + "\n".join(lines) + "\n")
print("written", OUT, len(lines), "lines")
