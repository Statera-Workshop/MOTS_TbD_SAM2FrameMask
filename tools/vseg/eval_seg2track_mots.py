"""Seg2Track-SAM2 결과를 mask 수준(MOTS)으로 평가합니다 (vseg01a, 코드 묶음과 독립인 단일 파일).

- 입력: ~/mot_logs/seg2track/<det>_<size>/annotations_mots/Seg2Track/data/<seq>.txt (KITTI-MOTS RLE txt)
- 정답: KITTI_MOTS/instances/<seq>/<frame>.png (uint16, class*1000+id, 10000 = 무시 영역).
  이 PNG는 KITTI-STEP에서 만든 것입니다(tools/mot/step_to_mots_png.py, 10-22). TrackEval이 읽는 KITTI-MOTS txt로
  한 번 변환해 <out>/gt_txt/에 둡니다. 무시 영역은 class 10, id 10000 줄 하나로 씁니다(KITTI-MOTS txt 규약).
  instances_txt 폴더가 있으면 그것을 그대로 씁니다.
- 지표: TrackEval KittiMOTS (pinned 12c8791b) HOTA / DetA / AssA / MOTSA / sMOTSA / IDF1, car·pedestrian 평균
- 시퀀스 길이는 image_02의 프레임 수로 넘깁니다(SEQ_INFO), seqmap 파일은 쓰지 않습니다.
- --gt-check: 정답을 그대로 예측으로 넣어 HOTA 1.000이 나오는지 먼저 확인합니다(경로·형식 점검).
- summary.md에는 box 수준 HOTA(tools/mot/eval_seg2track.py, ~/mot_logs/track/kittimots_<det>/seg2track.json)를 함께 둡니다.
출력: ~/mot_logs/seg2track_mots/<det>_<size>.json, summary.md
vseg02: 다른 MOTS 결과도 같은 방식으로 평가합니다(--root, --tracker-name, --out). JSON에 TP·FP·FN·IDSW 개수를 남깁니다.
    python tools/vseg/eval_seg2track_mots.py --root ~/mot_logs/sam2_frame --tracker-name SAM2frame \
        --out ~/mot_logs/sam2_frame_mots
정답 txt는 항상 ~/mot_logs/seg2track_mots/gt_txt 하나를 씁니다.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

HOME = os.path.expanduser("~")
VAL = ["0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018"]
TRAIN = ["0000", "0001", "0003", "0004", "0005", "0009", "0011", "0012", "0015", "0017", "0019", "0020"]

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="*", default=None, help="예: cascade_rcnn_l v6_l (기본: 전부)")
ap.add_argument("--root", default=f"{HOME}/mot_logs/seg2track")
ap.add_argument("--mots", default=f"{HOME}/Physical_AI/datasets/KITTI_MOTS")
ap.add_argument("--out", default=f"{HOME}/mot_logs/seg2track_mots")
ap.add_argument("--cores", type=int, default=max(1, (os.cpu_count() or 2) - 2))
ap.add_argument("--gt-check", action="store_true")
ap.add_argument("--tracker-name", default="Seg2Track", help="<root>/<run>/annotations_mots/<name>/data/<seq>.txt")
ap.add_argument("--split", default="val", choices=["val", "train", "hold2", "hold2b"],
                help="vseg04: train = the 12 training sequences (Seg2Track threshold selection)")
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()

try:
    import numpy as np
    import pycocotools  # noqa: F401  (TrackEval KittiMOTS가 mask IoU에 사용)
    for _n, _t in (("int", int), ("float", float), ("bool", bool)):  # pinned TrackEval은 np.int 등을 씀 (numpy>=1.24에서 제거)
        if _n not in np.__dict__:
            setattr(np, _n, _t)
    import trackeval
except ImportError as e:
    sys.exit(f"STOP: {e}. pip install pycocotools tabulate 후 다시 실행하세요.")


def png_to_txt(png_dir, txt_dir, info):
    """instances PNG -> KITTI-MOTS txt (frame id class h w rle). 이미 있고 프레임 수가 맞으면 건너뜁니다."""
    import cv2
    from pycocotools import mask as mu
    os.makedirs(txt_dir, exist_ok=True)
    for s, n in info.items():
        dst = f"{txt_dir}/{s}.txt"
        pngs = sorted(glob.glob(f"{png_dir}/{s}/*.png"))
        if len(pngs) != n:
            sys.exit(f"STOP: {png_dir}/{s} 정답 PNG {len(pngs)}장, 이미지 {n}장")
        if os.path.isfile(dst) and os.path.isfile(dst + ".ok"):
            continue
        lines = []
        for p in pngs:
            f = int(os.path.splitext(os.path.basename(p))[0])
            m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if m is None or m.dtype != np.uint16:
                sys.exit(f"STOP: {p} 가 uint16 PNG가 아닙니다")
            h, w = m.shape
            for v in np.unique(m):
                v = int(v)
                if v == 0:
                    continue
                cls = 10 if v == 10000 else v // 1000
                if cls not in (1, 2, 10):
                    sys.exit(f"STOP: {p} 에 알 수 없는 값 {v}")
                rle = mu.encode(np.asfortranarray((m == v).astype(np.uint8)))["counts"].decode()
                lines.append(f"{f} {v} {cls} {h} {w} {rle}")
        with open(dst, "w") as g:
            g.write("\n".join(lines) + "\n")
        open(dst + ".ok", "w").close()
        print(f"  gt_txt {s}: {n} frames, {len(lines)} masks", flush=True)


def find_gt(info):
    for c in ["instances_txt", "training/instances_txt", "label_02", "training/label_02"]:
        if os.path.isfile(f"{a.mots}/{c}/0002.txt"):
            return f"{a.mots}/{c}"
    if os.path.isdir(f"{a.mots}/instances/0002"):
        txt = f"{HOME}/mot_logs/seg2track_mots/gt_txt"
        png_to_txt(f"{a.mots}/instances", txt, info)
        return txt
    sys.exit(f"STOP: {a.mots} 아래에서 정답(instances_txt / label_02 / instances PNG)을 찾지 못했습니다.")


def box_hota(run):
    if a.tracker_name != "Seg2Track" or a.split != "val":
        return None
    det = run.rsplit("_", 1)[0]
    p = f"{HOME}/mot_logs/track/kittimots_{det}/seg2track.json"
    try:
        return json.load(open(p))["mean"]["HOTA"]
    except (OSError, KeyError, ValueError):
        return None


def seq_info():
    info = {}
    if a.split in ("hold2", "hold2b"):        # vseg06/07: held-out train sequences (tools/vseg/make_holdout.py)
        js = "~/vseg_logs/holdout.json" if a.split == "hold2" else "~/vseg_logs/holdout_b.json"
        seqs = json.load(open(os.path.expanduser(js)))["hold"]
    else:
        seqs = VAL if a.split == "val" else TRAIN
    for s in seqs:
        n = len(glob.glob(f"{a.mots}/training/image_02/{s}/*.png"))
        if n == 0:
            sys.exit(f"STOP: {a.mots}/training/image_02/{s} 프레임 없음")
        info[s] = n
    return info


def evaluate(trk_root, name, gt_dir, info):
    ec = trackeval.Evaluator.get_default_eval_config()
    ec.update(USE_PARALLEL=a.cores > 1, NUM_PARALLEL_CORES=a.cores, PRINT_RESULTS=False, PRINT_CONFIG=False,
              TIME_PROGRESS=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False,
              BREAK_ON_ERROR=True, RETURN_ON_ERROR=False, LOG_ON_ERROR=os.path.join(a.out, "trackeval_error.log"))
    dc = trackeval.datasets.KittiMOTS.get_default_dataset_config()
    dc.update(GT_FOLDER=gt_dir, GT_LOC_FORMAT="{gt_folder}/{seq}.txt", TRACKERS_FOLDER=trk_root,
              TRACKERS_TO_EVAL=[name], TRACKER_SUB_FOLDER="data", OUTPUT_FOLDER=os.path.join(a.out, "_trackeval"),
              SEQ_INFO=dict(info), CLASSES_TO_EVAL=["car", "pedestrian"], PRINT_CONFIG=False)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    res, _ = trackeval.Evaluator(ec).evaluate([trackeval.datasets.KittiMOTS(dc)], metrics)
    comb = res["KittiMOTS"][name]["COMBINED_SEQ"]
    out = {}
    for c in ["car", "pedestrian"]:
        r = comb[c]
        out[c] = {"HOTA": float(np.mean(r["HOTA"]["HOTA"])), "DetA": float(np.mean(r["HOTA"]["DetA"])),
                  "AssA": float(np.mean(r["HOTA"]["AssA"])), "MOTSA": float(r["CLEAR"]["MOTA"]),
                  "sMOTSA": float(r["CLEAR"]["sMOTA"]), "IDF1": float(r["Identity"]["IDF1"]),
                  "IDSW": int(r["CLEAR"]["IDSW"]), "TP": int(r["CLEAR"]["CLR_TP"]),
                  "FP": int(r["CLEAR"]["CLR_FP"]), "FN": int(r["CLEAR"]["CLR_FN"])}
    out["mean"] = {k: float(np.mean([out["car"][k], out["pedestrian"][k]]))
                   for k in ["HOTA", "DetA", "AssA", "MOTSA", "sMOTSA", "IDF1"]}
    return out


def gt_check(gt_dir, info):
    tmp = tempfile.mkdtemp(prefix="gtcheck_")
    os.makedirs(f"{tmp}/GT/data")
    for s in info:
        with open(f"{gt_dir}/{s}.txt") as f, open(f"{tmp}/GT/data/{s}.txt", "w") as g:
            for line in f:
                p = line.split(" ")
                if len(p) >= 6 and p[2] in ("1", "2"):
                    g.write(line)
    r = evaluate(tmp, "GT", gt_dir, info)
    shutil.rmtree(tmp)
    print(f"gt-check HOTA car {r['car']['HOTA']:.3f} ped {r['pedestrian']['HOTA']:.3f}")
    if min(r["car"]["HOTA"], r["pedestrian"]["HOTA"]) < 0.999:
        sys.exit("STOP: 정답을 예측으로 넣었는데 HOTA가 1이 아닙니다. 형식/경로를 확인하세요.")
    print("gt-check OK")


def main():
    os.makedirs(a.out, exist_ok=True)
    info = seq_info()
    gt_dir = find_gt(info)
    print(f"GT {gt_dir}  frames {sum(info.values())} ({len(info)} seqs)  cores {a.cores}")
    if a.gt_check:
        gt_check(gt_dir, info)
        return
    runs = a.runs or sorted(os.path.basename(os.path.dirname(os.path.dirname(p)))
                            for p in glob.glob(f"{a.root}/*/annotations_mots/{a.tracker_name}"))
    rows = []
    for run in runs:
        trk_root = f"{a.root}/{run}/annotations_mots"
        miss = [s for s in info if not os.path.isfile(f"{trk_root}/{a.tracker_name}/data/{s}.txt")]
        if miss:
            print(f"{run:<20} SKIP (없는 시퀀스 {miss})")
            continue
        js = f"{a.out}/{run}.json"
        if os.path.isfile(js) and not a.overwrite:
            r = json.load(open(js))
        if not os.path.isfile(js) or a.overwrite or "TP" not in r["car"]:     # vseg02: older JSONs lack TP / FP / FN
            r = evaluate(trk_root, a.tracker_name, gt_dir, info)
            json.dump(r, open(js, "w"), indent=1)
        m = r["mean"]
        tp, fp, fn = (sum(r[c][k] for c in ("car", "pedestrian")) for k in ("TP", "FP", "FN"))
        print(f"{run:<20} HOTA {m['HOTA']:.3f}  DetA {m['DetA']:.3f}  AssA {m['AssA']:.3f}  "
              f"sMOTSA {m['sMOTSA']:.3f}  car {r['car']['HOTA']:.3f}  ped {r['pedestrian']['HOTA']:.3f}  "
              f"TP {tp} FP {fp} FN {fn}", flush=True)
        rows.append((run, r))
    # vseg04: the summary lists every run evaluated into this folder so far, not only this call's --runs
    seen = {r for r, _ in rows}
    for js in sorted(glob.glob(f"{a.out}/*.json")):
        run = os.path.splitext(os.path.basename(js))[0]
        if run in seen:
            continue
        r = json.load(open(js))
        if isinstance(r, dict) and "mean" in r and "TP" in r.get("car", {}):
            rows.append((run, r))
    rows.sort(key=lambda x: -x[1]["mean"]["HOTA"])
    with open(f"{a.out}/summary.md", "w") as f:
        f.write("| run | mask HOTA | DetA | AssA | MOTSA | sMOTSA | IDF1 | car HOTA | ped HOTA | box HOTA | mask - box "
                "| TP | FP | FN |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for run, r in rows:
            m, b = r["mean"], box_hota(run)
            bs, ds = ("-", "-") if b is None else (f"{b:.3f}", f"{m['HOTA'] - b:+.3f}")
            f.write(f"| {run} | {m['HOTA']:.3f} | {m['DetA']:.3f} | {m['AssA']:.3f} | {m['MOTSA']:.3f} | "
                    f"{m['sMOTSA']:.3f} | {m['IDF1']:.3f} | {r['car']['HOTA']:.3f} | {r['pedestrian']['HOTA']:.3f} | "
                    f"{bs} | {ds} | {r['car']['TP'] + r['pedestrian']['TP']} | "
                    f"{r['car']['FP'] + r['pedestrian']['FP']} | {r['car']['FN'] + r['pedestrian']['FN']} |\n")
    print(f"-> {a.out}/summary.md ({len(rows)} runs)")


if __name__ == "__main__":
    main()
