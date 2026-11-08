"""
MOT v1.1 (2026-10-30): CAMELTrack - learned association (Somers et al., arXiv 2505.01257, TrackingLaboratory/CAMELTrack).

CAMEL replaces the hand-made association rules of a tracking-by-detection tracker with a small trainable module:
  Temporal Encoder (one per cue)  each detection / tracklet (last <= 50 observations) -> one token
                                  (linear projection of the cue, + sinusoidal encoding of the observation age,
                                  transformer encoder over the observations, cls token -> linear to emb_dim)
  merge                           tokens of the cues are summed
  GAFFE                           transformer encoder over [detections, tracklets, cls] of one frame
  similarity                      1 - ||a - b|| / 2 on L2-normalised embeddings ("norm_euclidean")
  association                     Hungarian on the similarity, pairs below sim_threshold are not matched
Training (association-centric): one sample = one frame; tracklets = past detections of every GT identity in the last
50 frames, detections = the detection of the same identities in the frame; loss = InfoNCE (NT-Xent, cosine, tau 0.07,
average over non-zero pair losses) on all tracklet and detection embeddings with the GT identities as labels.

Official configuration kept (configs/modules/track/camel.yaml): GAFFE 1024-d / 8 heads / 4 layers / ff 2048,
bbox encoder 64-d / ff 128, appearance encoder 512-d / ff 1024 (4 layers, 8 heads), dropout 0.1, AdamW 1e-4 /
wd 1e-4, cosine schedule with 5 % warm-up, augmentations BBoxShake 0.05, AppEmbNoise 0.4 (tracklets only),
DropoutFeatures (app 0.4, bbox 0.1), one of SwapSporadic 0.2 / DropoutSporadic 0.2 / SwapOccluded 0.5 (p 0.4 / 0.4 / 0.1),
online logic min_det_conf 0.4, min_init_det_conf 0.6, min_num_hits 0, gallery 50.
[Unified] decisions:
  * cues = box (ltwh / image size + score) and the shared KITTI-MOTS ReID feature (512-d, common/mot/reid.py) in place
    of the official BPBReID part features and keypoints (the global-appearance variant camel_bbox_app_global).
  * max_wo_hits 150 (30 fps) -> 50 at 10 fps, the framework's frame-rate rule (trackers.py).
  * one tracker instance per class (PerClass, like every tracker here); one model shared by the classes.
  * sim_threshold: computed after training on held-out train sequences (best balanced accuracy of the pair
    similarities), as the official use_computed_sim_threshold option; the validation split is never used.
  * the shared ReID feature is L2-normalised (about 0.04 per dimension); it is multiplied by sqrt(512) before the
    linear projection, otherwise the age encoding (amplitude 1) drowns it and all embeddings collapse (seen in the
    synthetic test). The official BPBReID part embeddings are not normalised. For the same reason the box input
    (ltwh / image size, score) is standardised with the mean / std of the train detections (model buffers box_mu,
    box_sd, stored in the checkpoint): with raw values (KITTI widths ~0.05) a box-only model did not learn at all.
  * sporadic swap / dropout pick observations uniformly (official: Gaussian weighting by age, sigma 50 frames).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

APP_DIM = 512
APP_SCALE = math.sqrt(APP_DIM)   # [Unified] unit-norm ReID feature -> unit variance per dimension (see docstring)
MAX_LEN = 50
DEFAULTS = dict(min_det_conf=0.4, min_init_det_conf=0.6, min_num_hits=0, max_wo_hits_30fps=150, gallery=50)


# ====================================================================== model
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=1000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x, age, mask):                     # x [B, N, S, E], age [B, N, S], mask [B, N, S]
        a = age.long().clamp(0, self.pe.shape[0] - 1)
        return x + self.pe[a] * mask.unsqueeze(-1)


class TemporalEncoder(nn.Module):
    """official TemporalEncoder + LinProj tokenizer: obs -> linear (dropped obs -> 0) -> +PE -> LN -> drop ->
    transformer (seq-first, as official) with a cls token -> linear_out(cls)."""

    def __init__(self, in_dim, hidden, out_dim, ff, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.lin, self.lin_drop = nn.Linear(in_dim, hidden), nn.Dropout(dropout)
        self.pe = PositionalEncoding(hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.norm, self.drop = nn.LayerNorm(hidden), nn.Dropout(dropout)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden, n_heads, ff, dropout), n_layers,
                                             enable_nested_tensor=False)
        self.out = nn.Linear(hidden, out_dim)
        self.hidden, self.out_dim = hidden, out_dim

    def forward(self, x, age, fmask, dropped):          # x [B, N, S, F]; masks [B, N, S] bool
        B, N, S, _ = x.shape
        tok = self.lin_drop(self.lin(x)) * (fmask & ~dropped).unsqueeze(-1)
        tok = self.drop(self.norm(self.pe(tok, age, fmask)))
        src = torch.cat([tok.permute(2, 0, 1, 3).reshape(S, B * N, self.hidden),
                         self.cls.expand(1, B * N, -1)], 0)
        key_mask = torch.cat([fmask.reshape(B * N, S), torch.ones(B * N, 1, dtype=torch.bool, device=x.device)], 1)
        h = self.encoder(src, src_key_padding_mask=~key_mask)[-1]           # cls [B*N, E]
        out = self.out(h).reshape(B, N, self.out_dim)
        return out * fmask.any(-1, keepdim=True)


class GAFFE(nn.Module):
    def __init__(self, dim=1024, n_heads=8, n_layers=4, ff=2048, dropout=0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(dim))
        self.norm, self.drop = nn.LayerNorm(dim), nn.Dropout(dropout)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, n_heads, ff, dropout, batch_first=True),
                                             n_layers, enable_nested_tensor=False)

    def forward(self, t_tok, t_mask, d_tok, d_mask):
        B, D = d_mask.shape
        src = self.drop(self.norm(torch.cat([d_tok, t_tok, self.cls.expand(B, 1, -1)], 1)))
        m = torch.cat([d_mask, t_mask, torch.ones(B, 1, dtype=torch.bool, device=d_mask.device)], 1)
        x = self.encoder(src, src_key_padding_mask=~m)
        return x[:, D:D + t_mask.shape[1]], x[:, :D]


class CAMEL(nn.Module):
    def __init__(self, emb_dim=1024):
        super().__init__()
        self.bbox_enc = TemporalEncoder(5, 64, emb_dim, 128)
        self.app_enc = TemporalEncoder(APP_DIM, 512, emb_dim, 1024)
        self.gaffe = GAFFE(emb_dim)
        self.register_buffer("sim_threshold", torch.tensor(0.5))   # official default before the computed one
        self.register_buffer("box_mu", torch.zeros(5))             # [Unified] box standardisation, set from the
        self.register_buffer("box_sd", torch.ones(5))              # train detections before training (see docstring)

    def tokens(self, s):
        """s: dict bbox [B,N,S,5], app [B,N,S,512], age [B,N,S], fmask, drop_bbox, drop_app [B,N,S] bool."""
        box = (s["bbox"] - self.box_mu) / self.box_sd * s["fmask"].unsqueeze(-1)
        return (self.bbox_enc(box, s["age"], s["fmask"], s["drop_bbox"])
                + self.app_enc(s["app"] * APP_SCALE, s["age"], s["fmask"], s["drop_app"]))

    def forward(self, trk, det):
        t_mask, d_mask = trk["fmask"].any(-1), det["fmask"].any(-1)
        t_emb, d_emb = self.gaffe(self.tokens(trk), t_mask, self.tokens(det), d_mask)
        sim = 1 - torch.cdist(F.normalize(t_emb, dim=-1), F.normalize(d_emb, dim=-1)) / 2
        return t_emb, d_emb, sim, t_mask, d_mask


def ntxent(emb, labels, tau=0.07):
    """pytorch_metric_learning NTXentLoss(CosineSimilarity) + AvgNonZeroReducer: for every positive pair (a, p)
    -log(exp(s_ap/tau) / (exp(s_ap/tau) + sum_n exp(s_an/tau))), mean over pairs with loss > 0."""
    if len(emb) < 2:
        return None
    e = F.normalize(emb, dim=-1)
    s = e @ e.t() / tau
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(e), dtype=torch.bool, device=e.device)
    pos, neg = same & ~eye, ~same
    if not pos.any():
        return None
    neg_exp = (torch.exp(s - s.max(1, keepdim=True).values) * neg).sum(1, keepdim=True)   # stable per row
    sp = s - s.max(1, keepdim=True).values
    loss = -(sp - torch.log(torch.exp(sp) + neg_exp))
    loss = loss[pos]
    nz = loss > 1e-12
    return loss[nz].mean() if nz.any() else loss.sum() * 0


def batch_loss(t_emb, d_emb, t_mask, d_mask, t_lab, d_lab):
    losses = []
    for b in range(t_emb.shape[0]):
        tv, dv = t_mask[b], d_mask[b]          # v1.1-b: false positives (unique negative labels) act as negatives
        l = ntxent(torch.cat([t_emb[b, tv], d_emb[b, dv]]), torch.cat([t_lab[b, tv], d_lab[b, dv]]))
        if l is not None:
            losses.append(l)
    return torch.stack(losses).mean() if losses else None


def hungarian(sim, thr):
    """official hungarian_algorithm: values below thr set to thr - 1e-5, assignment, then pairs < thr unmatched."""
    s = sim.copy()
    s[s < thr] = thr - 1e-5
    r, c = linear_sum_assignment(-s)
    ok = sim[r, c] >= thr
    return np.stack([r[ok], c[ok]], 1) if ok.any() else np.empty((0, 2), int)


# ====================================================================== online tracker
_MODELS = {}


def load_camel(path, device="cuda"):
    key = (str(path), device)
    if key not in _MODELS:
        ck = torch.load(path, map_location=device, weights_only=False)
        m = CAMEL().to(device)
        m.load_state_dict(ck["model_state_dict"], strict=True)
        m.eval()
        _MODELS[key] = m
    return _MODELS[key]


class _Tracklet:
    def __init__(self, tid, obs, gallery):
        self.id, self.obs, self.gallery = tid, [obs], gallery     # obs = (frame, bbox5, app, xyxy, score)
        self.state, self.hits, self.hit_streak, self.time_wo_hits = "init", 0, 0, 0

    def forward(self):
        self.time_wo_hits += 1
        if self.time_wo_hits > 1:
            self.hit_streak = 0

    def update(self, obs):
        self.obs = (self.obs + [obs])[-self.gallery:]
        self.hits += 1
        self.hit_streak += 1
        self.time_wo_hits = 0


class CAMELTracker:
    """official CAMELTrack.process(): forward -> CAMEL association -> update -> new tracklets (conf >= init) ->
    state machine (init / active / lost / dead) -> output the active tracklets matched in this frame."""
    needs_feats, needs_gmc = True, False

    def __init__(self, frame_rate=10, weights=None, img_wh=(1242, 375), device="cuda", sim_threshold=None, model=None,
                 **kw):
        p = dict(DEFAULTS, **kw)
        self.min_det_conf, self.min_init = p["min_det_conf"], p["min_init_det_conf"]
        self.min_num_hits, self.gallery = p["min_num_hits"], p["gallery"]
        self.max_wo_hits = max(1, int(round(p["max_wo_hits_30fps"] * frame_rate / 30.0)))
        self.weights, self.device, self.wh = weights, device, np.array(img_wh, np.float32)
        self.model = model if model is not None else (None if weights is None else load_camel(weights, device))
        if model is not None:
            self.device = next(model.parameters()).device
        self.thr = sim_threshold if sim_threshold is not None else (
            float(self.model.sim_threshold) if self.model is not None else 0.5)
        self.tracks, self.frame, self.count = [], 0, 0

    def _obs(self, row, feat):
        x1, y1, x2, y2, s = row[:5]
        b5 = np.array([x1 / self.wh[0], y1 / self.wh[1], (x2 - x1) / self.wh[0], (y2 - y1) / self.wh[1], s],
                      np.float32)
        return (self.frame, b5, np.asarray(feat, np.float32), row[:4].copy(), float(s))

    @torch.no_grad()
    def _similarity(self, dets):
        T, S = len(self.tracks), max(len(t.obs) for t in self.tracks)
        tb, ta = np.zeros((1, T, S, 5), np.float32), np.zeros((1, T, S, APP_DIM), np.float32)
        tg, tm = np.zeros((1, T, S), np.float32), np.zeros((1, T, S), bool)
        for i, t in enumerate(self.tracks):
            for j, o in enumerate(reversed(t.obs)):                    # newest first, as official padded_features
                tb[0, i, j], ta[0, i, j], tg[0, i, j], tm[0, i, j] = o[1], o[2], self.frame - o[0], True
        D = len(dets)
        db = np.stack([o[1] for o in dets])[None, :, None]
        da = np.stack([o[2] for o in dets])[None, :, None]
        dev = self.device
        z = lambda a: torch.from_numpy(a).to(dev)                       # noqa: E731
        trk = dict(bbox=z(tb), app=z(ta), age=z(tg), fmask=z(tm), drop_bbox=torch.zeros_like(z(tm)),
                   drop_app=torch.zeros_like(z(tm)))
        dm = torch.ones(1, D, 1, dtype=torch.bool, device=dev)
        det = dict(bbox=z(db), app=z(da), age=torch.zeros(1, D, 1, device=dev), fmask=dm,
                   drop_bbox=torch.zeros_like(dm), drop_app=torch.zeros_like(dm))
        return self.model(trk, det)[2][0].float().cpu().numpy()

    def update(self, dets, feats=None):
        dets = np.asarray(dets, np.float64).reshape(-1, 6)
        keep = dets[:, 4] >= self.min_det_conf
        obs = [self._obs(r, f) for r, f in zip(dets[keep], np.asarray(feats)[keep])]
        for t in self.tracks:
            t.forward()
        matched_det = set()
        if self.tracks and obs:
            sim = self._similarity(obs)
            for ti, di in hungarian(sim, self.thr):
                self.tracks[ti].update(obs[di])
                matched_det.add(di)
        for di, o in enumerate(obs):
            if di not in matched_det and o[4] >= self.min_init:
                self.count += 1
                self.tracks.append(_Tracklet(self.count, o, self.gallery))
        out = []
        for t in self.tracks:
            if t.state == "init":
                t.state = "active" if t.hit_streak >= self.min_num_hits else "dead" if t.time_wo_hits >= 1 else "init"
            elif t.state in ("active", "lost"):
                t.state = "active" if t.time_wo_hits == 0 else "lost" if t.time_wo_hits < self.max_wo_hits else "dead"
            if t.state == "active" or (t.state == "init" and self.frame < self.min_num_hits):
                o = t.obs[-1]
                if o[0] == self.frame:
                    out.append([*o[3], t.id, 0, o[4]])
        self.tracks = [t for t in self.tracks if t.state != "dead"]
        self.frame += 1
        return np.array(out, np.float64).reshape(-1, 7)
