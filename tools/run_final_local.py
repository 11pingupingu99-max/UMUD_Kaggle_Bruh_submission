"""Run a final notebook locally (no Kaggle):
    python tools/run_final_local.py F1|F3 [out_dir] [--no-img]
Executes the code cells of final/UMUD_FINAL_<name>_KAGGLE.ipynb and prints the SHA-256 of submission.csv.
`img` (device group, PNG label and scale read from the images) needs the 309 test images under /kaggle/input (any folder
layout; found by file name). --no-img drops that token: the CSV then depends on the embedded tables only (check values in
final/VARIANTS.json, `csv_sha256_without_img`)."""
import hashlib, json, os, re, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
args = [a for a in sys.argv[1:] if not a.startswith('--')]
name = args[0]
out = Path(args[1]) if len(args) > 1 else Path(tempfile.mkdtemp())
out.mkdir(parents=True, exist_ok=True); os.environ['R31_WORKDIR'] = str(out)
nb = ROOT / 'final' / f'UMUD_FINAL_{name}_KAGGLE.ipynb'
cells = [''.join(c['source']) for c in json.loads(nb.read_text())['cells'] if c['cell_type'] == 'code']
if '--no-img' in sys.argv:
    cells[0] = re.sub(r'(VARIANT = "[^"]*)\+img"', r'\1"', cells[0])
g = {}
for i, src in enumerate(cells):
    exec(compile(src, f'cell{i + 1}', 'exec'), g)
csv = out / 'submission.csv'
print('SUBMISSION', csv, 'SHA-256', hashlib.sha256(csv.read_bytes()).hexdigest())
