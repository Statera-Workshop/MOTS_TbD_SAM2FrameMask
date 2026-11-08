#!/usr/bin/env python3
"""
vseg03 (2026-11-01): MOTS results from an instance-segmentation model + a box tracker (tracking-by-detection with the
model's own masks), the third family next to Seg2Track-SAM2 and "box tracker + SAM2 per frame".

    python tools/vseg/tracks_to_mots.py --dets ~/mot_logs/dets/kittimots/mask_rcnn_km.npz \
        --tracks ~/mot_logs/track_vseg/kittimots_mask_rcnn_km/hybridsort_tracks.npz

Each output track box takes the mask of the detection of the same frame and class with the highest box IoU
(>= --iou, default 0.5; trackers report the matched detection's box or its Kalman update, so the match is
normally near 1). Track rows with no such detection (predicted-only boxes) get no mask. Masks are made
non-overlapping as KITTI-MOTS requires: painted in ascending track score, the higher score keeps a shared pixel.
Ids are renumbered per sequence in order of first appearance (as tools/vseg/sam2_frame_masks.py).
Output: ~/mot_logs/tbd_masks/<det>_<tracker>/annotations_mots/TbDmask/data/<seq>.txt, then
    python tools/vseg/eval_seg2track_mots.py --root ~/mot_logs/tbd_masks --tracker-name TbDmask --out ~/mot_logs/tbd_masks_mots
"""

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu


def box_iou(a, b):
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--tracks", nargs="+", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default="~/mot_logs/tbd_masks")
    a = ap.parse_args()
    dz = np.load(Path(a.dets).expanduser())
    for tp in a.tracks:
        tp = Path(tp).expanduser()
        tz = np.load(tp)
        run = f"{tp.parent.name.replace('kittimots_', '')}_{tp.stem.replace('_tracks', '')}"
        od = Path(a.out).expanduser() / run / "annotations_mots" / "TbDmask" / "data"
        od.mkdir(parents=True, exist_ok=True)
        stats = {"tracks": str(tp), "dets": str(a.dets), "rows": 0, "matched": 0, "masks": 0}
        for seq in sorted(tz.files):
            trows, drows, rles = tz[seq], dz[seq], dz[f"{seq}__rle"]
            h, w = (int(v) for v in dz[f"{seq}__hw"])
            if len(drows) != len(rles):
                raise SystemExit(f"STOP: {seq}: {len(drows)} detections but {len(rles)} masks")
            remap = {}
            for t in (trows[np.argsort(trows[:, 0], kind="stable"), 5] if len(trows) else []):
                remap.setdefault(int(t), len(remap) + 1)
            lines = []
            for f in np.unique(trows[:, 0]).astype(int) if len(trows) else []:
                tr = trows[trows[:, 0] == f]
                di = np.where(drows[:, 0] == f)[0]
                stats["rows"] += len(tr)
                if not len(di):
                    continue
                iou = box_iou(tr[:, 1:5], drows[di, 1:5])
                iou[tr[:, 6][:, None] != drows[di, 6][None, :]] = 0.0          # same class only
                best = iou.argmax(1)
                cand = [(k, di[best[k]]) for k in range(len(tr)) if iou[k, best[k]] >= a.iou]
                if not cand:
                    continue
                stats["matched"] += len(cand)
                masks = {k: mu.decode({"size": [h, w], "counts": rles[j].encode()}).astype(bool) for k, j in cand}
                idmap = np.full((h, w), -1, np.int64)
                for k in sorted(masks, key=lambda q: tr[q, 7]):                 # ascending score
                    idmap[masks[k]] = k
                for k in masks:
                    m = idmap == k
                    if not m.any():
                        continue
                    rle = mu.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode()
                    lines.append(f"{f} {remap[int(tr[k, 5])]} {int(tr[k, 6]) + 1} {h} {w} {rle}")
                    stats["masks"] += 1
            (od / f"{seq}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        (od.parent.parent.parent / "run_info.json").write_text(json.dumps(stats, indent=1))
        print(f"{run}: {stats['rows']} track rows, {stats['matched']} matched to a detection mask, "
              f"{stats['masks']} masks written -> {od}")


if __name__ == "__main__":
    main()
