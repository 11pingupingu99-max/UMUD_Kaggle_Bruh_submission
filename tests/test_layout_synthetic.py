"""Synthetic check of block detection: a 5-frame clip of near-identical frames between unrelated images."""
import sys; sys.path.insert(0, "/home/claude/umud_final")
import numpy as np
from umud_final.layout import blocks_from_images, group_of, grysub_of
rng = np.random.default_rng(0)
base = rng.normal(100, 30, (800, 1200)).clip(0, 255)
imgs = [rng.normal(100, 30, (800, 1200)).clip(0, 255)]
imgs += [np.clip(base + rng.normal(0, 3, base.shape), 0, 255) for _ in range(5)]
imgs += [rng.normal(100, 30, (800, 1200)).clip(0, 255) for _ in range(2)]
nums = np.arange(10, 18); lays = ["1200_left_ticks"] * 8
bid, sims = blocks_from_images(nums, lays, lambda k: imgs[k])
assert list(bid) == [-1, 0, 0, 0, 0, 0, -1, -1], bid
assert group_of("644x1088") == "teln" and grysub_of("512_fixed") == "gry_b" and group_of("png_variable") == "png"
print("PASS: bloki syntetyczne", list(bid))
