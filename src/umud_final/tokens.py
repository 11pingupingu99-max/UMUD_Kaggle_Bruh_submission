"""Runs the token pipeline (constants applied to per-image measurements) on computed input tables."""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
from pathlib import Path

import pandas as pd

CORE = Path(__file__).with_name("tokens_core.py")
REQUIRED = ("base", "BAND", "PADEC", "FL22", "PNG_MUSCLE", "OR_INS", "V2MT", "V2OV", "R17DEV", "R17ENS",
            "FX_COLS", "FX", "GROUP", "GRYSUB", "BID")


def run_tokens(inputs: dict, variant: str, workdir: str | None = None, quiet: bool = True):
    """Returns (submission DataFrame, namespace, log text)."""
    missing = [k for k in REQUIRED if k not in inputs]
    if missing:
        raise KeyError(f"missing inputs: {missing}")
    wd = workdir or tempfile.mkdtemp()
    old = os.environ.get("R31_WORKDIR")
    os.environ["R31_WORKDIR"] = wd
    ns = {"INPUTS": inputs, "VARIANT": variant, "__name__": "umud_tokens_core"}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
            exec(compile(CORE.read_text(), str(CORE), "exec"), ns)
    finally:
        if old is None:
            os.environ.pop("R31_WORKDIR", None)
        else:
            os.environ["R31_WORKDIR"] = old
    sub = pd.read_csv(Path(wd) / "submission.csv")
    return sub, ns, buf.getvalue()
