from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np

LABELS = ["walking", "running", "standing", "resting", "eating", "drinking", "grooming",
          "marking", "urination", "defecation", "unknown"]
UNKNOWN = LABELS.index("unknown")
CAMS = ["CAM01", "CAM02", "CAM03", "CAM04"]
NOSE, EYES, NECK, MID, TAIL, TIP = range(6)
N_FEATURES = 44
N_LONG = 6
TICK_S = 1 / 7.5


def _angle(a, b, c) -> tuple[float, float]:
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return 0.0, 0.0
    cos = float(np.dot(v1, v2) / n)
    sin = float((v1[0] * v2[1] - v1[1] * v2[0]) / n)
    return cos, sin


class FeatureMaker:

    def __init__(self, zones: dict, kpt_conf: float = 0.5):
        self.zones = zones
        self.kpt_conf = kpt_conf
        self.prev: tuple | None = None
        self.last_seen: datetime | None = None

    def update(self, t: datetime, o) -> np.ndarray:
        f = np.zeros(N_FEATURES, np.float32)
        since = (t - self.last_seen).total_seconds() if self.last_seen else 10.0
        f[43] = min(since, 10.0) / 10.0
        if o is None:
            self.prev = None
            return f
        self.last_seen = t
        x1, y1, x2, y2 = o.box
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        s = max(w, h)
        c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        f[0] = 1.0
        f[1 + CAMS.index(o.cam_id)] = 1.0 if o.cam_id in CAMS else 0.0
        f[5] = 1.0 if o.place.kind == "door" else 0.0
        f[6] = o.conf
        f[7], f[8], f[9] = w / 1920, h / 1080, math.log(h / w)
        kp = o.kpts if o.kpts is not None else np.zeros((6, 3), np.float32)
        xy, vis = kp[:, :2], kp[:, 2]
        ok = vis >= self.kpt_conf
        rel = np.where(ok[:, None], (xy - c) / s, 0.0)
        f[10:22] = rel.reshape(-1)
        f[22:28] = vis
        if ok[NOSE] and ok[NECK]:
            f[28] = (xy[NOSE, 1] - xy[NECK, 1]) / s
        if ok[NECK] and ok[MID] and ok[TAIL]:
            f[29], f[30] = _angle(xy[NECK], xy[MID], xy[TAIL])
        if ok[MID] and ok[TAIL] and ok[TIP]:
            f[31], f[32] = _angle(xy[MID], xy[TAIL], xy[TIP])
        if ok[NOSE] and ok[TAIL]:
            f[33] = np.linalg.norm(xy[NOSE] - xy[TAIL]) / s
        p = self.prev
        if p and p[1] == o.cam_id and p[2] == o.track_id and 0 < (t - p[0]).total_seconds() <= 1.0:
            dt = (t - p[0]).total_seconds()
            v = (c - p[3]) / s / dt
            f[34], f[35], f[36] = v[0], v[1], float(np.linalg.norm(v))
            f[37] = (s - p[4]) / s / dt
            both = ok & p[6]
            if both.any():
                f[38] = float(np.linalg.norm(xy[both] - p[5][both], axis=1).mean() / s / dt)
        z = self.zones[o.cam_id]
        head = xy[NOSE] if ok[NOSE] else c
        gp = (c[0], y2)
        for sid, stype, poly in z.stations:
            if stype in ("food", "water", "food_or_water"):
                d = z._dist(poly, head)
                f[39] = max(f[39], 1.0 if d >= 0 else 0.0)
                f[40] = max(f[40], float(np.clip(1 + d / s, 0, 1)))
            elif stype == "rest" and z._dist(poly, gp) >= -10:
                f[41] = 1.0
        f[42] = 1.0 if o.place.station else 0.0
        self.prev = (t, o.cam_id, o.track_id, c, s, xy.copy(), ok.copy())
        return f


class LongTerm:

    def __init__(self, dt: float = TICK_S):
        self.a10, self.a30, self.a60 = dt / 10.0, dt / 30.0, dt / 60.0
        self.s = np.zeros(7, np.float64)

    def update(self, f: np.ndarray) -> np.ndarray:
        s = self.s
        s[6] += self.a60 * (f[0] - s[6])
        if f[0] > 0:
            sp, kp = min(float(f[36]), 5.0), min(float(f[38]), 5.0)
            s[0] += self.a10 * (sp - s[0])
            s[1] += self.a60 * (sp - s[1])
            s[2] += self.a10 * (kp - s[2])
            s[3] += self.a60 * (kp - s[3])
            s[4] += self.a30 * (np.clip(f[34], -5, 5) - s[4])
            s[5] += self.a30 * (np.clip(f[35], -5, 5) - s[5])
        return np.array([s[0], s[1], s[2], s[3], math.hypot(s[4], s[5]), s[6]], np.float32)


def with_long(X: np.ndarray) -> np.ndarray:
    lt = LongTerm()
    return np.concatenate([X, np.stack([lt.update(f) for f in X])], 1)


def civet_observation(state, obs):
    if state.room is None:
        return None
    kinds = ("floor", "door") if state.state == "VISIBLE" else ("floor", "door", "outside", "gate")
    cands = [o for o in obs if o.room == state.room and o.place.kind in kinds and o.track_age >= 2]
    if not cands:
        return None
    return max(cands, key=lambda o: (o.cam_id == state.cam_id, o.conf))


class Observer:

    def __init__(self, fallback: bool = False, need: int = 3, of: int = 5):
        self.fallback, self.need = fallback, need
        self.recent: deque = deque(maxlen=of)

    def __call__(self, state, obs, raw=()):
        o = civet_observation(state, obs)
        if not self.fallback:
            return o
        cands = []
        if o is None and state.state == "HIDDEN" and state.room is not None:
            cands = [r for r in raw if r.room == state.room and r.place.kind in ("floor", "door")]
        self.recent.append(bool(cands))
        if o is not None or not cands or sum(self.recent) < self.need:
            return o
        return max(cands, key=lambda r: (r.cam_id == state.cam_id, r.conf))


def build_net(n_features: int = N_FEATURES, n_classes: int = len(LABELS), width: int = 64, delay: int = 0):
    import torch
    import torch.nn as nn

    class CausalConv(nn.Conv1d):
        def __init__(self, cin, cout, k, dil):
            super().__init__(cin, cout, k, dilation=dil)
            self.left = (k - 1) * dil

        def forward(self, x):
            return super().forward(nn.functional.pad(x, (self.left, 0)))

    class BehaviourNet(nn.Module):

        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                CausalConv(n_features, width, 3, 1), nn.ReLU(), nn.Dropout(0.2),
                CausalConv(width, width, 3, 2), nn.ReLU(), nn.Dropout(0.2),
                CausalConv(width, width, 3, 4), nn.ReLU())
            self.head = nn.Linear((3 if delay else 2) * width, n_classes)

        def forward(self, x):
            h = self.body(x.transpose(1, 2))
            parts = [h[:, :, -1], h.mean(2)] + ([h[:, :, -1 - delay]] if delay else [])
            return self.head(torch.cat(parts, 1))

    return BehaviourNet()


@dataclass
class Bout:
    label: str
    start: datetime
    end: datetime
    conf: float
    camera: str | None


class BoutBuilder:

    def __init__(self, min_s: float = 2.0):
        self.min_s = min_s
        self.cur: Bout | None = None
        self.cand: tuple | None = None

    def update(self, t: datetime, label: str, conf: float, camera: str | None) -> Bout | None:
        if self.cur and label == self.cur.label:
            self.cur.end, self.cand = t, None
            return None
        if self.cand is None or self.cand[0] != label:
            self.cand = (label, t, [conf], camera)
            return None
        self.cand[2].append(conf)
        if (t - self.cand[1]).total_seconds() < self.min_s:
            return None
        closed = self.cur
        if closed:
            closed.end = self.cand[1]
        self.cur = Bout(label, self.cand[1], t, float(np.mean(self.cand[2])), self.cand[3])
        self.cand = None
        return closed

    def close(self) -> Bout | None:
        b, self.cur = self.cur, None
        return b


class BehaviourRunner:

    def __init__(self, weights: str, zones: dict, device: str = "cpu"):
        import torch
        ck = torch.load(weights, map_location=device, weights_only=False)
        self.window, self.stride, self.smooth = ck["window"], ck["stride"], ck["smooth"]
        self.mean, self.std = ck["mean"], ck["std"]
        self.delay = int(ck.get("delay", 0))
        self.hold_s = float(ck.get("hold_s", 0.0))
        self.last_seen: datetime | None = None
        self.long = LongTerm() if ck.get("long") else None
        self.nets = []
        for st in ck.get("states") or [ck["state"]]:
            net = build_net(n_features=len(self.mean), delay=self.delay)
            net.load_state_dict(st)
            self.nets.append(net.eval())
        self.torch, self.device = torch, device
        self.fm = FeatureMaker(zones)
        self.observer = Observer(bool(ck.get("fallback", False)))
        self.buf: deque = deque(maxlen=self.window)
        self.past: deque = deque(maxlen=self.delay + 1)
        self.preds: deque = deque(maxlen=self.smooth)
        self.bouts = BoutBuilder(ck.get("min_bout_s", 2.0))
        self.n = 0
        self.camera: str | None = None
        self.label: tuple[int, float] | None = None

    def update(self, t: datetime, state, obs, raw=()) -> Bout | None:
        o = self.observer(state, obs, raw)
        if o is not None:
            self.camera = o.cam_id
        return self.step(t, self.fm.update(t, o), o is not None)

    def step(self, t: datetime, feat: np.ndarray, visible: bool) -> Bout | None:
        if self.long is not None:
            feat = np.concatenate([feat, self.long.update(feat)])
        self.buf.append((feat - self.mean) / self.std)
        self.past.append((t, visible, self.camera))
        if self.past[0][1]:
            self.last_seen = self.past[0][0]
        self.n += 1
        if self.n % self.stride or len(self.buf) < self.window:
            return None
        t, visible, camera = self.past[0]
        x = self.torch.from_numpy(np.stack(self.buf)[None].astype(np.float32))
        with self.torch.no_grad():
            p = np.mean([self.torch.softmax(net(x), 1)[0].numpy() for net in self.nets], 0)
        k, c = int(p.argmax()), float(p.max())
        if not visible:
            if self.hold_s and self.preds and self.last_seen and (t - self.last_seen).total_seconds() < self.hold_s:
                k, c = self.preds[-1]
            else:
                k, c = UNKNOWN, 1.0
        self.preds.append((k, c))
        votes = Counter(k for k, _ in self.preds)
        lab = votes.most_common(1)[0][0]
        conf = float(np.mean([c for k, c in self.preds if k == lab]))
        self.label = (lab, conf)
        return self.bouts.update(t, LABELS[lab], conf, camera)
