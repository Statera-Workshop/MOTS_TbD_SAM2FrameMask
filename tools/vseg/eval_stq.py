#!/usr/bin/env python3
"""
vseg01 (2026-10-31): STQ (Segmentation and Tracking Quality, the KITTI-STEP metric) on the 9 val sequences.

    python tools/vseg/eval_stq.py --check                                   # gt + gt -> STQ 1.000
    python tools/vseg/eval_stq.py --sem gt --inst seg2track:cascade_rcnn_l  # Seg2Track association, gt semantics
    python tools/vseg/eval_stq.py --sem gt --inst seg2track:all             # every finished Seg2Track run
    python tools/vseg/eval_stq.py --sem deeplabv3plus_in --inst seg2track:cascade_rcnn_l
    python tools/vseg/eval_stq.py --sem deeplabv3plus_in --inst none        # semantics only (AQ of "no tracking")

Metric: google-research/deeplab2 numpy STQuality, copied verbatim (common/seg/stq_deeplab2.py): 19 classes,
things = person (11) and car (13), void 255, panoptic id = (class << 16) + track id. STQ = sqrt(AQ x SQ);
SQ = 19-class IoU over the whole split, AQ = track-level association quality over car / person tubes
(crowd = thing pixels without id are ignored for AQ, as in the official code).

Panoptic prediction = semantic map + instance tracks:
    --sem gt          KITTI-STEP semantic ground truth (isolates the tracking side)
    --sem <version>   ~/vseg_logs/sem/<version>/<seq>/<frame>.png (tools/vseg/dump_semseg.py, trainIds)
    --inst none       no ids: every car / person pixel is one tube per class (id 0)
    --inst gt         ground-truth ids (with --sem gt this is the --check)
    --inst mots:<dir> vseg03: any KITTI-MOTS txt folder <dir>/<seq>.txt (SAM2 per frame, model + tracker), e.g.
                      mots:~/mot_logs/sam2_frame/cascade_rcnn_reid_tracktrack/annotations_mots/SAM2frame/data
    --inst seg2track:<run>   ~/mot_logs/seg2track/<run>/annotations_mots/Seg2Track/data/<seq>.txt (KITTI-MOTS
                      RLE): every mask overwrites its pixels with car (MOTS class 1) / person (class 2) and its id,
                      in file order. [Unified] semantic thing pixels not covered by any mask keep their class with
                      id 0 (one "unassigned" tube per class), the usual way of merging instance and semantic outputs.
Output: ~/vseg_logs/stq/<sem>__<inst>.json and one line per run in ~/vseg_logs/stq/summary.md
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.seg.step_labels import CAR, NUM_CLASSES, PERSON, read_panoptic  # noqa: E402
from common.seg.stq_deeplab2 import STQuality  # noqa: E402

HOME = Path("~").expanduser()
STEP = HOME / "Physical_AI/datasets/kitti-step/panoptic_maps/val"
SEM_ROOT = HOME / "vseg_logs/sem"
S2T_ROOT = HOME / "mot_logs/seg2track"
OUT = HOME / "vseg_logs/stq"
SHIFT, OFFSET = 16, 1 << 24
MOTS_TO_STEP = {1: CAR, 2: PERSON}


def new_metric():
    return STQuality(NUM_CLASSES, [PERSON, CAR], 255, SHIFT, OFFSET)


def read_mots(path):
    """KITTI-MOTS txt -> {frame: [(id, step_class, rle dict)]}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        p = line.split(" ")
        if len(p) < 6 or int(p[2]) not in MOTS_TO_STEP:
            continue
        tid = int(p[1])
        if not 0 < tid < (1 << SHIFT):
            raise ValueError(f"{path}: track id {tid} does not fit in 16 bits")
        out.setdefault(int(p[0]), []).append(
            (tid, MOTS_TO_STEP[int(p[2])], {"size": [int(p[3]), int(p[4])], "counts": p[5].encode()}))
    return out


def eval_sequence(job):
    seq, sem_src, inst_src = job
    from pycocotools import mask as mu
    metric = new_metric()
    frames = sorted(STEP.joinpath(seq).glob("*.png"))
    if inst_src.startswith("mots:"):
        tracks = read_mots(Path(inst_src[5:]).expanduser() / f"{seq}.txt")
    elif inst_src not in ("none", "gt"):
        tracks = read_mots(S2T_ROOT / inst_src / "annotations_mots/Seg2Track/data" / f"{seq}.txt")
    else:
        tracks = {}
    n_masks = 0
    for fp in frames:
        gsem, ginst = read_panoptic(fp)
        gt = (gsem.astype(np.int64) << SHIFT) + ginst
        if sem_src == "gt":
            sem = gsem.astype(np.int64)
        else:
            p = SEM_ROOT / sem_src / seq / fp.name
            s = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if s is None or s.shape != gsem.shape:
                raise RuntimeError(f"{p}: missing or shape {None if s is None else s.shape} != {gsem.shape}")
            sem = s.astype(np.int64)
        inst = np.zeros_like(sem)
        if inst_src == "gt":
            thing = (gsem == CAR) | (gsem == PERSON)
            inst[thing] = ginst[thing]
            sem[thing] = gsem[thing]
        else:
            for tid, cls, rle in tracks.get(int(fp.stem), []):
                if tuple(rle["size"]) != gsem.shape:
                    raise RuntimeError(f"{seq}/{fp.stem}: mask size {rle['size']} != label {gsem.shape}")
                m = mu.decode(rle).astype(bool)
                sem[m], inst[m] = cls, tid
                n_masks += 1
        metric.update_state(gt, (sem << SHIFT) + inst, sequence_id=seq)
    return seq, metric, len(frames), n_masks


def run(sem_src, inst_src, cores, overwrite):
    OUT.mkdir(parents=True, exist_ok=True)
    if inst_src.startswith("mots:"):                       # name = <run>-<tracker name> from .../<run>/annotations_mots/<name>/data
        d = Path(inst_src[5:]).expanduser()
        label = f"{d.parent.parent.parent.name}-{d.parent.name}"
    else:
        label = inst_src.replace(":", "-")
    name = f"{sem_src}__{label}"
    js = OUT / f"{name}.json"
    if js.exists() and not overwrite:
        r = json.loads(js.read_text())
        print(f"{name:<44} STQ {r['STQ']:.4f}  AQ {r['AQ']:.4f}  SQ {r['SQ']:.4f}  (cached)")
        return r
    src = inst_src.split(":", 1)[1] if inst_src.startswith("seg2track:") else inst_src
    seqs = sorted(p.name for p in STEP.iterdir() if p.is_dir())
    with Pool(min(cores, len(seqs))) as pool:
        parts = pool.map(eval_sequence, [(s, sem_src, src) for s in seqs])
    metric = new_metric()
    metric.merge_state([m for _, m, _, _ in parts])
    res = metric.result()
    r = {"sem": sem_src, "inst": inst_src, "STQ": float(res["STQ"]), "AQ": float(res["AQ"]), "SQ": float(res["IoU"]),
         "frames": sum(n for *_, n, _ in parts), "masks": sum(k for *_, k in parts),
         "per_seq": {str(s): {"STQ": float(a), "AQ": float(b), "SQ": float(c), "frames": int(n)}
                     for s, a, b, c, n in zip(res["ID_per_seq"], res["STQ_per_seq"], res["AQ_per_seq"],
                                              res["IoU_per_seq"], res["Length_per_seq"])}}
    js.write_text(json.dumps(r, indent=1))
    print(f"{name:<44} STQ {r['STQ']:.4f}  AQ {r['AQ']:.4f}  SQ {r['SQ']:.4f}  ({r['frames']} frames, "
          f"{r['masks']} masks)", flush=True)
    with open(OUT / "summary.md", "a") as f:
        f.write(f"| {sem_src} | {inst_src} | {r['STQ']:.4f} | {r['AQ']:.4f} | {r['SQ']:.4f} |\n")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sem", default="gt")
    ap.add_argument("--inst", default="none", help="none | gt | seg2track:<run> | seg2track:all")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cores", type=int, default=9)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.check:
        r = run("gt", "gt", a.cores, True)
        if min(r["STQ"], r["AQ"], r["SQ"]) < 0.9999:
            sys.exit("STOP: ground truth as prediction is not 1.000")
        print("stq-check OK")
        return
    insts = [a.inst]
    if a.inst == "seg2track:all":
        insts = [f"seg2track:{p.parent.parent.parent.name}"
                 for p in sorted(S2T_ROOT.glob("*/annotations_mots/Seg2Track/data"))]
    for inst in insts:
        run(a.sem, inst, a.cores, a.overwrite)


if __name__ == "__main__":
    main()
