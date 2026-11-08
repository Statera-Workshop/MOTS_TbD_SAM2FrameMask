# MOTS_TbD_SAM2FrameMask

**TbD + SAM2 frame masks** - multi-object tracking and segmentation (MOTS) on KITTI-MOTS: a box tracker (tracking by
detection) gives the identities, SAM2.1-large prompted frame by frame with the tracked boxes gives the masks.
Best mask HOTA of the video-segmentation stage: **0.616** (Cascade R-CNN +
TrackTrack).

Part of the Physical AI comparison series of the Statera Working Group: same data, protocol and metrics, only the
method changes. Detectors and trackers come from the MOT repositories (`Statera-Workshop/MOT_Detectors_KITTIMOTS`,
`MOT_TbD_*`), mask models from the SEG repositories (`SEG_Instance_*`, v1.1), Seg2Track-SAM2 from
`MOTS_SAM2_Seg2Track`.

## Method

1. Detections of the MOT-stage detectors (1216 x 384, KITTI-MOTS train, 30 epochs) and, for TrackTrack / BoT-SORT,
   their ReID features (`MOT_Detectors_KITTIMOTS`, Release v1.0: `<detector>_reid.npz`).
2. Box tracker: `tools/mot/run_tracker.py --save-tracks` (the v1.0 trackers, thresholds as in the MOT stage).
3. SAM2 per frame: `tools/vseg/sam2_frame_masks.py` - `SAM2ImagePredictor` (sam2.1_hiera_large, as Seg2Track), one
   `set_image` per frame, all tracked boxes of the frame in one call, no video memory. KITTI-MOTS forbids overlapping
   masks: painted in ascending score order. Track ids renumbered per sequence (STQ needs ids < 65536).
   SAM2 time: about 122 ms per frame (RTX 4070 Laptop), offline use.
4. Comparison partners: the same tracks with the mask of the matched detection of an instance-segmentation model
   (`tools/vseg/tracks_to_mots.py`, "own masks"), and Seg2Track-SAM2 (tracking by segmentation with SAM2 memory).

## Results (KITTI-MOTS val, 9 sequences, 2,981 frames)

Mask HOTA / DetA / AssA / sMOTSA: TrackEval KittiMOTS (pinned 12c8791b), car / pedestrian mean. STQ: deeplab2
definition with the FCN (ImageNet) KITTI-STEP semantic map of `SEG_Semantic_CNN_FCN` v1.1 (`tools/vseg/eval_stq.py`).
Box HOTA of the tracks: the same tracker output at box level (MOT-stage evaluation).

| method | detector | tracker / threshold | **mask HOTA** | DetA | AssA | sMOTSA | car / ped. HOTA | FP | STQ (FCN semantic) | box HOTA of the tracks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM2 frame masks | Cascade R-CNN | TrackTrack | **0.616** | 0.558 | 0.686 | 0.519 | 0.696 / 0.537 | 473 | 0.545 | 0.625 |
| Seg2Track-SAM2 | Cascade R-CNN | thr 0.8 | **0.602** | 0.542 | 0.678 | 0.418 | 0.666 / 0.539 | 2,313 | 0.559 | - |
| SAM2 frame masks | Cascade Mask R-CNN | Hybrid-SORT | **0.591** | 0.539 | 0.652 | 0.514 | 0.694 / 0.488 | 395 | 0.513 | 0.598 |
| SAM2 frame masks | Mask R-CNN | Hybrid-SORT | **0.590** | 0.533 | 0.659 | 0.486 | 0.699 / 0.482 | 700 | 0.519 | 0.584 |
| SAM2 frame masks | Cascade R-CNN | Hybrid-SORT | **0.588** | 0.538 | 0.646 | 0.498 | 0.686 / 0.490 | 500 | 0.512 | 0.595 |
| own masks | Cascade Mask R-CNN | BoT-SORT | **0.578** | 0.543 | 0.618 | 0.484 | 0.695 / 0.460 | 713 | 0.553 | 0.612 |
| Seg2Track-SAM2 | Cascade R-CNN | thr 0.7 | **0.572** | 0.506 | 0.654 | 0.304 | 0.656 / 0.488 | 3,347 | - | - |
| own masks | Mask R-CNN | BoT-SORT | **0.568** | 0.518 | 0.629 | 0.402 | 0.705 / 0.432 | 1,437 | 0.556 | 0.593 |
| own masks | Cascade Mask R-CNN | ByteTrack | **0.564** | 0.528 | 0.604 | 0.482 | 0.684 / 0.443 | 559 | 0.513 | 0.571 |
| Seg2Track-SAM2 | Cascade R-CNN | thr 0.5 | **0.560** | 0.491 | 0.648 | 0.215 | 0.658 / 0.462 | 4,312 | 0.560 | - |
| own masks | Cascade Mask R-CNN | Hybrid-SORT | **0.558** | 0.507 | 0.618 | 0.471 | 0.668 / 0.448 | 504 | 0.504 | 0.598 |
| own masks | Mask R-CNN | ByteTrack | **0.557** | 0.512 | 0.609 | 0.427 | 0.688 / 0.426 | 1,050 | 0.508 | 0.560 |
| own masks | Mask R-CNN | Hybrid-SORT | **0.554** | 0.499 | 0.620 | 0.435 | 0.670 / 0.437 | 834 | 0.507 | 0.584 |
| Seg2Track-SAM2 | YOLOv6 | thr 0.5 | **0.538** | 0.461 | 0.634 | 0.250 | 0.632 / 0.443 | 3,993 | - | - |
| own masks | Cascade Mask R-CNN | OC-SORT | **0.530** | 0.485 | 0.581 | 0.463 | 0.655 / 0.404 | 257 | 0.485 | 0.566 |
| SAM2 frame masks | YOLOv6 | Hybrid-SORT | **0.522** | 0.469 | 0.585 | 0.437 | 0.656 / 0.389 | 467 | 0.484 | 0.501 |
| own masks | Mask R-CNN | OC-SORT | **0.518** | 0.481 | 0.559 | 0.435 | 0.666 / 0.370 | 539 | 0.490 | 0.548 |
| SAM2 frame masks | YOLO11-seg | Hybrid-SORT | **0.478** | 0.412 | 0.557 | 0.401 | 0.624 / 0.332 | 135 | 0.460 | 0.465 |
| own masks | YOLO11-seg | BoT-SORT | **0.425** | 0.360 | 0.508 | 0.335 | 0.600 / 0.250 | 224 | 0.476 | 0.452 |
| own masks | YOLO11-seg | Hybrid-SORT | **0.425** | 0.361 | 0.505 | 0.334 | 0.578 / 0.271 | 271 | 0.440 | 0.465 |
| own masks | YOLO11-seg | ByteTrack | **0.424** | 0.373 | 0.487 | 0.345 | 0.594 / 0.254 | 285 | 0.443 | 0.439 |
| own masks | YOLO11-seg | OC-SORT | **0.360** | 0.290 | 0.460 | 0.283 | 0.556 / 0.164 | 54 | 0.427 | 0.381 |

- Same Hybrid-SORT tracks, masks from SAM2 instead of the model's own mask head: Cascade Mask R-CNN +0.033, Mask R-CNN +0.037, YOLO11-seg +0.053 mask HOTA. SAM2 turns a box into a mask with almost no loss; a learned mask head loses more.
- Best: Cascade R-CNN + TrackTrack + SAM2 frame masks (0.616). Seg2Track-SAM2 on the same detections: official 0.5 0.560 (gap 0.056); chosen on train (0.8) 0.602 (gap 0.014); mean of two held-out folds (0.7) 0.572 (gap 0.044). How close it comes depends on how its detection threshold is chosen - see `Statera-Workshop/MOTS_SAM2_Seg2Track` (v1.2 section).
- With a weak detector and a simple tracker (YOLOv6 + Hybrid-SORT) Seg2Track wins (0.538 vs 0.522; AssA 0.634 vs 0.585): SAM2's video memory repairs associations a weak tracker breaks.
- STQ ranks differently from mask HOTA: its association term rewards recall and hardly penalises false-positive tubes, so recall-heavy combinations (BoT-SORT, Seg2Track at 0.5) score higher there.

Result JSONs: `results/sam2_frame/`, `results/own_masks/`, `results/seg2track/`, `results/stq/`.

## Protocol (all MOT_* repositories)

- Online / causal only: every method sees the current frame and its own state; checked by re-running prefixes of
  every sequence (the output of frame t must not change when later frames are removed).
- Input 1216 x 384 (KITTI 1242 x 375 letterboxed), detectors trained from scratch on KITTI-MOTS train (12 sequences),
  30 epochs, Adam 1e-4, effective batch 8, seed 777 - the protocol of the *Physical AI Object Detection* series.
- Metrics: HOTA / DetA / AssA (TrackEval classes, car / pedestrian mean), MOTA, IDF1, IDSW.
- Latency: RTX 4070 Laptop GPU, batch 1, fp32, medians over val; pipeline = detector (forward + post-processing) +
  ReID (appearance trackers) + tracker (CPU); image decoding / letterboxing not included.
- Trackers run with their official default parameters; frame-based values given for 30 fps are scaled to 10 fps.
  A sensitivity table with one common score threshold for every tracker is reported separately (chosen on val, so it
  is analysis, not ranking).

## Usage

Data (KITTI tracking images, KITTI-STEP -> KITTI-MOTS instance PNGs and VOC layout):

```bash
bash tools/mot/get_kitti_mots.sh
python tools/mot/step_to_mots_png.py && python tools/mot/kittimots_to_voc.py
```

Tracks (or take `tracks_*.npz` from Release v1.0 and skip this step):

```bash
gh release download v1.0 --repo Statera-Workshop/MOT_Detectors_KITTIMOTS --pattern cascade_rcnn_reid.npz
python tools/mot/run_tracker.py --source kittimots --dets cascade_rcnn_reid.npz --trackers tracktrack --save-tracks
```

SAM2 frame masks and evaluation (SAM2 in its own environment, as in `MOTS_SAM2_Seg2Track`):

```bash
git clone https://github.com/facebookresearch/sam2 && (cd sam2 && pip install -e . && cd checkpoints && bash download_ckpts.sh)
python tools/vseg/sam2_frame_masks.py --tracks ~/mot_logs/track/kittimots_cascade_rcnn_reid/tracktrack_tracks.npz \
    --ckpt-dir sam2/checkpoints
pip install pycocotools tabulate git+https://github.com/JonathonLuiten/TrackEval@12c8791b
python tools/vseg/eval_seg2track_mots.py --gt-check
python tools/vseg/eval_seg2track_mots.py --root ~/mot_logs/sam2_frame --tracker-name SAM2frame --out ~/mot_logs/sam2_frame_mots
```

Own masks: `dets_<model>.npz` (Release v1.0: detections + masks of the three `SEG_Instance_*` KITTI models) ->
`run_tracker.py --dets dets_<model>.npz --save-tracks` -> `tools/vseg/tracks_to_mots.py` -> the same evaluation with
`--root ~/mot_logs/tbd_masks --tracker-name TbDmask`.

STQ: unpack `sem_fcn_in_kittistep_val.tar.gz` (Release) to `~/vseg_logs/sem/fcn_in`, then
`python tools/vseg/eval_stq.py --check` and
`python tools/vseg/eval_stq.py --sem fcn_in --inst mots:<folder with the MOTS txt files>`.

Release v1.0: the tracks of every row above (`tracks_<run>_<tracker>.npz`), detections with masks of the three
instance models (`dets_*.npz`), the MOTS output of the best row (`mots_cascade_rcnn_tracktrack_SAM2frame.tar.gz`) and
the FCN semantic maps (`sem_fcn_in_kittistep_val.tar.gz`). KITTI / KITTI-MOTS / KITTI-STEP are CC BY-NC-SA 3.0
(non-commercial); these files inherit the condition.

## Files

```
tools/mot/run_tracker.py, common/mot/   box trackers (v1.0 of the series) and their evaluation
tools/vseg/sam2_frame_masks.py          SAM2 per frame on tracked boxes -> KITTI-MOTS txt
tools/vseg/tracks_to_mots.py            tracked boxes + a model's own masks -> KITTI-MOTS txt
tools/vseg/eval_seg2track_mots.py       mask-level MOTS evaluation (TrackEval KittiMOTS)
tools/vseg/eval_stq.py, common/seg/     STQ (deeplab2 code, copied verbatim)
tools/mot/*kitti*                       data preparation
results/                                result JSONs of the tables
THIRD_PARTY.md                          sources and licences
```

## Global Collaboration

Hills Robotics is currently participating in multiple projects in Korea and internationally.

We are actively interested in building relationships with organizations that can complement our technology with global sales networks, distribution channels, system integration capabilities, manufacturing ecosystems, AI infrastructure, robotics technologies, research capabilities, or strategic investment.

Several technology and business collaborations are already underway. Some partners, projects, system architectures, and commercial programs cannot be publicly disclosed because they are subject to confidentiality and business agreements.

We believe that the emerging Physical AI ecosystem cannot be built by a single company.
It requires a global network connecting robotics companies, AI researchers, semiconductor and computing companies, sensor manufacturers, software developers, universities, system integrators, industrial customers, investors, and global distribution partners.

We have now begun our serious exploration of the Physical AI frontier, and we welcome connections with people and organizations who want to build this ecosystem together.

## Research, Contribution, and Donation

The repositories of the Statera Working Group are intended to contribute to the broader technical discussion around Physical AI.

Researchers, engineers, developers, companies, universities, and independent contributors interested in these subjects are welcome to connect with us.

If you would like to collaborate on Physical AI or autonomous robotics, participate in research or technology development, discuss international business cooperation, propose technology or platform integration, contribute technical knowledge, or support this GitHub knowledge initiative through sponsorship or donation, please contact us.

## Contact

**Richard Minsoo Go**
Head of AI, Hills Robotics
Statera Working Group

Personal / Research Contact:
[gominsoo888@gmail.com](mailto:gominsoo888@gmail.com)

Hills Robotics / Business Contact:
[minsoo.go@hillslogis.com](mailto:minsoo.go@hillslogis.com)

Hills Robotics:
https://hillslogis.com/

For inquiries regarding next-generation outdoor autonomous robotic platforms, Physical AI Spider Grid, Intelligence Cabinet, Edge AI, or On-Premise AI architectures that are not yet publicly presented on the company website, please contact us directly.
