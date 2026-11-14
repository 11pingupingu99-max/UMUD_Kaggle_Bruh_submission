"""Public source/format checks. No private labels, predictions, images or run logs."""
import ast
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import tempfile
import unittest
import zipfile

C1 = Path(__file__).resolve().parents[1]
REPO = C1.parent
spec = importlib.util.spec_from_file_location('c1_public_support', C1 / 'tools/final_notebook_support.py')
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


class PublicRelease(unittest.TestCase):
    def test_frozen_source(self):
        self.assertEqual(support.frozen_source_hash(C1), support.SOURCE_SHA256)
        self.assertEqual(len(list((C1 / 'src').rglob('*.py'))), 54)

    def test_exact_checkpoint_metadata(self):
        m = json.loads((C1 / 'models/WEIGHTS_MANIFEST.json').read_text())
        recorded = json.loads((C1 / 'models/AUDITED_RUN_MODEL_SIGNATURE.json').read_text())['signature']['models']
        self.assertEqual(sum(map(len, m['models'].values())), 30)
        self.assertEqual(set(m['models']), set(recorded))
        paths = []
        for family, entries in m['models'].items():
            self.assertEqual({d['sha256'] for d in entries}, {d['sha256'] for d in recorded[family]})
            for d in entries:
                p = PurePosixPath(d['path'])
                self.assertFalse(p.is_absolute())
                self.assertNotIn('..', p.parts)
                self.assertEqual(len(d['sha256']), 64)
                paths.append(d['path'])
        self.assertEqual(len(paths), len(set(paths)))

    def test_both_notebook_packages(self):
        variants = json.loads((REPO / 'final/VARIANTS.json').read_text())
        for key in ('F1', 'F3'):
            with self.subTest(profile=key):
                p = REPO / variants[key]['notebook']
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), variants[key]['notebook_sha256'])
                notebook = json.loads(p.read_text())
                values = {}
                for cell in notebook['cells']:
                    if cell['cell_type'] != 'code':
                        continue
                    tree = ast.parse(''.join(cell['source']))
                    for node in tree.body:
                        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                            name = node.targets[0].id
                            if name in ('PROFILE', 'PAYLOAD_B64', 'PAYLOAD_SHA256'):
                                values[name] = ast.literal_eval(node.value)
                self.assertEqual(values['PROFILE'], variants[key]['profile'])
                raw = base64.b64decode(values['PAYLOAD_B64'], validate=True)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), values['PAYLOAD_SHA256'])
                with tempfile.TemporaryDirectory() as temp:
                    with zipfile.ZipFile(io.BytesIO(raw)) as z:
                        for name in z.namelist():
                            rel = PurePosixPath(name)
                            self.assertFalse(rel.is_absolute())
                            self.assertNotIn('..', rel.parts)
                            self.assertFalse(name.endswith(('.csv', '.pt', '.log')))
                        z.extractall(temp)
                    self.assertEqual(support.frozen_source_hash(Path(temp)), support.SOURCE_SHA256)

    def test_submission_io_on_synthetic_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / 'synthetic_run'
            run.mkdir()
            data = b'image_id,pa_deg,fl_mm,mt_mm\nsynthetic-a.tif,15,50,20\nsynthetic-b.png,17,60,22\n'
            name = 'submission_C1_F1.csv'
            source = run / name
            source.write_bytes(data)
            (run / 'IMAGES_FROM_PIXELS.csv').write_text('image_id\nsynthetic-a.tif\nsynthetic-b.png\n')
            (run / 'DONE.json').write_text(json.dumps({'source_sha256': support.SOURCE_SHA256, 'images': 2,
                'files': {name: hashlib.sha256(data).hexdigest()}, 'time_utc': 'SYNTHETIC_TEST_ONLY'}))
            dest = root / 'output/submission.csv'
            receipt = support.select_submission(run, 'C1_F1', dest)
            self.assertEqual(dest.read_bytes(), data)
            self.assertIsNone(receipt['new_run_leaderboard_score'])
            source.write_bytes(data + b'\n')
            with self.assertRaisesRegex(ValueError, 'DONE.json'):
                support.select_submission(run, 'C1_F1', root / 'rejected/submission.csv')
            self.assertFalse((root / 'rejected/submission.csv').exists())


if __name__ == '__main__':
    unittest.main()
