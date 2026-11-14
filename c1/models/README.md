# Checkpoints required for C1

WEIGHTS_MANIFEST.json lists the30 exact expected files by portable relative path. The binary checkpoints are not in this
archive. Use tools/export_weights.py against the completed C1 run while the user's Drive is mounted, then publish that
verified folder as a Dataset if requested. tools/configure_inference.py maps this layout to an arbitrary --weights-dir.
A weights folder alone is insufficient to establish that a public Dataset exists; release/FINALIZATION.json tracks its URL.
