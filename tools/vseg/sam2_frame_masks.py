#!/usr/bin/env python3
"""
vseg02 (2026-10-31): masks for box-tracker outputs with SAM2 prompted frame by frame (no video memory).

Purpose: the comparison partner of Seg2Track-SAM2 at mask level. Same SAM2.1 weights, same detector, but SAM2 only
turns each tracked box into a mask in that frame; the identities come from the box tracker (v1.0 / v1.1 TbD
trackers). The gap to Seg2Track is then what SAM2's video propagation adds.

    conda activate mot_sam2
    python tools/vseg/sam2_frame_masks.py --tracks ~/mot_logs/track/kittimots_cascade_rcnn/tracktrack_tracks.npz
    # tracks: tools/mot/run_tracker.py ... --save-tracks  (object_det env)

Input : <tracker>_tracks.npz, per sequence rows [frame, x1, y1, x2, y2, id, class, score] (original px, class 0 = car,
        1 = pedestrian); images KITTI_MOTS/training/image_02/<seq>/<frame>.png.
SAM2  : official facebookresearch/sam2 SAM2ImagePredictor, sam2.1_hiera_large (as Seg2Track), one set_image per
        frame, all boxes of the frame in one predict(box=[N, 4], multimask_output=False).
Masks : KITTI-MOTS forbids overlapping masks -> painted in ascending score order (the higher score keeps a shared
        pixel); empty masks are dropped. MOTS class = class + 1. MOTS id: vseg02b - tracker ids are renumbered per
        sequence in order of first appearance (1, 2, ...), because some trackers encode the class in the id
        (e.g. 100003) and STQ needs ids < 65536; identity structure is unchanged.
Output: ~/mot_logs/sam2_frame/<run>_<tracker>/annotations_mots/SAM2frame/data/<seq>.txt (KITTI-MOTS RLE) +
        run_info.json (SAM2 ms per frame). Resumable per sequence. Evaluate with
        python tools/vseg/eval_seg2track_mots.py --root ~/mot_logs/sam2_frame --tracker-name SAM2frame \\
            --out ~/mot_logs/sam2_frame_mots
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HOME = Path("~").expanduser()
CKPT = {"l": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
        "b+": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
        "s": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
        "t": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", nargs="+", required=True)
    ap.add_argument("--size", default="l", choices=list(CKPT))
    ap.add_argument("--ckpt-dir", default=str(HOME / "Physical_AI/refs_mot/sam2/checkpoints"))
    ap.add_argument("--images", default=str(HOME / "Physical_AI/datasets/KITTI_MOTS/training/image_02"))
    ap.add_argument("--out", default=str(HOME / "mot_logs/sam2_frame"))
    ap.add_argument("--min-score", type=float, default=0.0, help="drop track boxes below this score (default keep)")
    a = ap.parse_args()

    import torch
    import sam2
    from pycocotools import mask as mu
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print(f"sam2 package: {os.path.dirname(sam2.__file__)}")
    cfg, ck = CKPT[a.size]
    predictor = SAM2ImagePredictor(build_sam2(cfg, str(Path(a.ckpt_dir) / ck), device="cuda"))

    for tp in a.tracks:
        tp = Path(tp).expanduser()
        z = np.load(tp)
        run = f"{tp.parent.name.replace('kittimots_', '')}_{tp.stem.replace('_tracks', '')}"
        od = Path(a.out) / run / "annotations_mots" / "SAM2frame" / "data"
        od.mkdir(parents=True, exist_ok=True)
        info_p = Path(a.out) / run / "run_info.json"
        info = json.loads(info_p.read_text()) if info_p.exists() else {"tracks": str(tp), "size": a.size, "sec": {}}
        for seq in sorted(z.files):
            if (od.parent.parent.parent / f"done_{seq}").exists():
                continue
            rows = z[seq]
            frames = sorted(Path(a.images, seq).glob("*.png"))
            remap = {}
            for t in rows[np.argsort(rows[:, 0], kind="stable"), 5] if len(rows) else []:
                remap.setdefault(int(t), len(remap) + 1)
            lines, t_sum = [], 0.0
            for f, fp in enumerate(frames):
                r = rows[(rows[:, 0] == f) & (rows[:, 7] >= a.min_score)] if len(rows) else rows
                if not len(r):
                    continue
                img = cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    predictor.set_image(img)
                    masks, _, _ = predictor.predict(box=r[:, 1:5].astype(np.float32), multimask_output=False)
                torch.cuda.synchronize()
                t_sum += time.time() - t0
                masks = np.asarray(masks).reshape(len(r), -1, h, w)[:, 0] > 0
                idmap = np.full((h, w), -1, np.int64)
                for k in np.argsort(r[:, 7], kind="stable"):          # ascending score: higher score wins overlaps
                    idmap[masks[k]] = k
                for k in range(len(r)):
                    m = idmap == k
                    if not m.any():
                        continue
                    tid, cls = remap[int(r[k, 5])], int(r[k, 6]) + 1
                    if not 0 < tid < 65536:
                        sys.exit(f"STOP: track id {tid} out of range")
                    rle = mu.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode()
                    lines.append(f"{f} {tid} {cls} {h} {w} {rle}")
            (od / f"{seq}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            info["sec"][seq] = t_sum
            info["ms_per_frame"] = 1000 * sum(info["sec"].values()) / max(1, sum(
                len(list(Path(a.images, s).glob("*.png"))) for s in info["sec"]))
            info_p.write_text(json.dumps(info, indent=1))
            (od.parent.parent.parent / f"done_{seq}").touch()
            print(f"{run} {seq}: {len(frames)} frames, {len(lines)} masks, {1000 * t_sum / len(frames):.0f} ms/frame",
                  flush=True)
        print(f"-> {od}  ({info['ms_per_frame']:.0f} ms/frame SAM2)")


if __name__ == "__main__":
    main()
