"""Fixed final profiles on newly computed measurements."""
from __future__ import annotations
import contextlib
import io
from pathlib import Path
import numpy as np
import pandas as pd

CORE = Path(__file__).with_name("r96_math.py")


def run_math(inputs, variant):
    # These were cache/reader switches in R96. The fresh driver already performed their work.
    parts = [p for p in variant.split("+") if p not in ("ctab", "rbasek", "img")]
    # Splitting '+' also splits positive numeric offsets; joining restores the exact original string.
    var = "+".join(parts)
    ns = {"INPUTS": inputs, "VARIANT": var, "__name__": "umud_fresh_math"}
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        exec(compile(CORE.read_text(), str(CORE), "exec"), ns)
    return ns["sub"].copy(), log.getvalue()


def compare(reference, actual):
    """Per-target comparison, never a claim about the unknown leaderboard labels."""
    if isinstance(reference, (str, Path)):
        reference = pd.read_csv(reference)
    if reference.image_id.duplicated().any() or actual.image_id.duplicated().any():
        raise ValueError("Duplicate IDs in comparison")
    joined = actual.merge(reference, on="image_id", how="inner", suffixes=("_new", "_ref"), validate="one_to_one")
    rows = []
    for c in ("pa_deg", "fl_mm", "mt_mm"):
        a = joined[c + "_new"].to_numpy(float); b = joined[c + "_ref"].to_numpy(float)
        d = a - b
        rows.append({"target": c, "n": len(d), "mean_delta": float(d.mean()) if len(d) else None,
                     "mae_between_predictions": float(np.abs(d).mean()) if len(d) else None,
                     "max_abs_delta": float(np.abs(d).max()) if len(d) else None,
                     "equal_1e_8": int((np.abs(d) <= 1e-8).sum()),
                     "missing_reference_rows": int(len(reference) - len(joined)),
                     "extra_prediction_rows": int(len(actual) - len(joined))})
    return pd.DataFrame(rows), joined
