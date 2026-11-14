# Frozen C1 image-inference runtime

This isolated directory preserves the executed C1 source SHA-256 af622512b9883b87aa1fe95f026ec882f95791266467abbc4acf49e6f93771c9.
From repository root, use `python c1/tools/configure_inference.py ...` then `python c1/tools/run_fresh.py --config ...`.
The runtime reads `c1/final/VARIANTS.json`, never the publication metadata in root `final/VARIANTS.json`.
The source hash covers 54 active Python files, the driver, PNG templates and frozen profiles; extra historical vendor CSV assets are archived outside this directory.
Tests: `python -m unittest discover -s c1/tests_c1 -v`. Binary weights are not bundled.
