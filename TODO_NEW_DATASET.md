# Remaining work for a one-command run on a new dataset

Status 13 Nov 2026. On the competition data the final CSVs are reproduced by `final/UMUD_FINAL_F1_KAGGLE.ipynb` / `_F3_` with the
`VARIANT` strings in `final/VARIANTS.json`. On new images, stages 5–12 of the README must run first; then the tables are rebuilt.

1. Restore the builders marked ✎ in `PROVENANCE.md`:
   - `build_rbase.py` (base PA / FL / MT priority rules);
   - FX columns 110 / 111 / 130 / 132 / 72 (formulas given there).
   Check each against the tables embedded in the final notebook: same values on the 309 test images, tolerance as for `ctab`.
2. Add a `build_tables.py` that writes all tables as one JSON and a loader cell in the final notebook, so the notebook can read
   tables from a file instead of embedded literals.
3. Add a single driver notebook running inference stages 5–12 in sequence on a folder of new images (weights reused, no
   training).
4. Test: rebuild the tables from the original exports and confirm that F1 and F3 are identical to the submitted CSVs.
5. Publish the weights as a Kaggle Dataset with the SHA-256 list.
6. Before publishing (GitHub, GPL-3.0): move the notebooks from the Drive folder `pipeline_z_Drive` into `pipeline/`.
