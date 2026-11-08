"""KITTI-STEP label encoding (vseg01, 2026-10-31). Torch-free, shared by the dataset and the CPU evaluation tools.

STEP panoptic PNG: R = semantic class (Cityscapes 19 trainIds, 255 = void), G * 256 + B = instance id
(car / person only, 0 = no instance, consistent over time = track id).
"""

from pathlib import Path

import cv2
import numpy as np

IGNORE = 255
NUM_CLASSES = 19
PERSON, CAR = 11, 13                           # STEP things (trainIds)
SPLIT_DIR = {"trainval": "train", "train": "train", "test": "val", "val": "val"}


def read_panoptic(path):
    """STEP PNG -> (semantic uint8 [h, w] with 255 void, instance int32 [h, w])."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return bgr[..., 2].copy(), bgr[..., 1].astype(np.int32) * 256 + bgr[..., 0].astype(np.int32)


def sequence_frames(root, split):
    """-> {seq: [frame stems]} for one split (sorted)."""
    d = Path(root) / "panoptic_maps" / SPLIT_DIR[split]
    if not d.is_dir():
        raise FileNotFoundError(d)
    return {s.name: sorted(p.stem for p in s.glob("*.png")) for s in sorted(p for p in d.iterdir() if p.is_dir())}
