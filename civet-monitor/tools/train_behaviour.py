import argparse
import csv
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from civetwatch.behaviour import (LABELS, UNKNOWN, BehaviourRunner, BoutBuilder, FeatureMaker,
                                  Observer, build_net, civet_observation, with_long)
from civetwatch.config import load_config

DAY = "2026-04-05"
PIECES = {"train_0035_0114": ("00:35:00", "01:14:00"),
          "train_0300_0350": ("03:00:00", "03:50:00"),
          "train_0114_0140": ("01:14:00", "01:40:00"),
          "train_0350_0450": ("03:50:00", "04:50:00"),
          "train_1540_1620": ("15:40:00", "16:20:00"),
          "test_2300_2330": ("23:00:00", "23:30:00")}
OUT = ROOT / "outputs" / "behaviour"
FEAT = OUT
RES = OUT
WINDOW, STRIDE, SMOOTH, VAL_SHARE = 40, 4, 3, 0.2


ALIASES = {"hidden": "unknown"}
FLIP_NEG = [10, 12, 14, 16, 18, 20, 30, 32, 34]


def clock(t: str) -> datetime:
    return datetime.fromisoformat(f"{DAY}T{t.strip()}")


def read_labels(path: Path) -> tuple[list[tuple[datetime, int]], datetime, datetime] | None:
    rows = [r for r in csv.DictReader(open(path, newline="", encoding="utf-8")) if (r.get("time") or "").strip()]
    start = next((clock(r["time"]) for r in rows if r["behaviour"].strip().upper() == "START"), None)
    end = next((clock(r["time"]) for r in rows if r["behaviour"].strip().upper() == "END"), None)
    pts = []
    for r in rows:
        b = r["behaviour"].strip().lower()
        b = ALIASES.get(b, b)
        if b in ("start", "end"):
            continue
        if b not in LABELS:
            sys.exit(f"{path.name} {r['time']}: '{b}' is not one of {LABELS}")
        pts.append((clock(r["time"]), LABELS.index(b)))
    if not pts or start is None or end is None:
        return None
    return sorted(pts), start, end


def labels_at(ts: np.ndarray, lab) -> np.ndarray:
    pts, start, end = lab
    t_pts = np.array([p[0].timestamp() for p in pts])
    k = np.searchsorted(t_pts, ts, side="right") - 1
    y = np.where(k >= 0, np.array([p[1] for p in pts])[np.clip(k, 0, None)], -1)
    return np.where((ts >= start.timestamp()) & (ts < end.timestamp()), y, -1)


def extract(a) -> None:
    from civetwatch.pipeline import Pipeline
    from civetwatch.run import CachedSource, _file_start
    cfg = load_config(str(ROOT / a.config))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (s0, e0) in PIECES.items():
        if a.piece and name != a.piece:
            continue
        cfg["storage"]["room_log_csv"] = str(FEAT / f"room_log_{name}.csv")
        cfg["storage"]["db_path"] = str(FEAT / f"replay_{name}.sqlite")
        Path(cfg["storage"]["db_path"]).unlink(missing_ok=True)
        p = Pipeline(cfg)
        st = {c: CachedSource(c, str(Path(a.cache) / f"{c}.npz"), _file_start(cc)) for c, cc in cfg["cameras"].items()}
        fm = FeatureMaker(p.zones)
        pick = Observer(a.fallback)
        t, end = clock(s0), clock(e0)
        ts, X, vis, cams, rooms = [], [], [], [], []
        while t < end:
            frames = {c: s.frame_at((t - s.file_start).total_seconds()) for c, s in st.items()}
            state, obs = p.tick(t, frames, set())
            o = pick(state, obs, p.last_raw)
            ts.append(t.timestamp())
            X.append(fm.update(t, o))
            vis.append(o is not None)
            cams.append(o.cam_id if o else "")
            rooms.append(state.room or "")
            t += timedelta(seconds=p.active_dt)
        p.close()
        np.savez_compressed(FEAT / f"feat_{name}.npz", ts=np.array(ts), X=np.array(X, np.float32),
                            visible=np.array(vis), cam=np.array(cams), room=np.array(rooms), fallback=a.fallback)
        print(f"{name}: {len(ts)} ticks, civet seen in {100 * np.mean(np.array(X)[:, 0] > 0):.0f}%")


def features_fallback() -> bool:
    flags = {bool(np.load(FEAT / f"feat_{n}.npz").get("fallback", False)) for n in train_pieces()}
    if len(flags) > 1:
        sys.exit(f"{FEAT}: pieces made with and without --fallback")
    return flags.pop()


def load_piece(name: str):
    z = np.load(FEAT / f"feat_{name}.npz")
    lab = read_labels(ROOT / "labelling" / f"behaviour_{name}.csv")
    return z, (labels_at(z["ts"], lab) if lab else None)


def is_draft(name: str) -> bool:
    a, b = ROOT / "labelling" / f"behaviour_{name}.csv", OUT / f"prelabel_{name}.csv"
    return b.exists() and a.read_bytes() == b.read_bytes()


def train_pieces() -> list[str]:
    out = []
    for name in PIECES:
        if not name.startswith("train") or not (FEAT / f"feat_{name}.npz").exists():
            continue
        if is_draft(name):
            print(f"{name}: still the uncorrected draft, skipped")
        elif read_labels(ROOT / "labelling" / f"behaviour_{name}.csv") is None:
            print(f"{name}: no labels yet, skipped")
        else:
            out.append(name)
    return out


def windows(X: np.ndarray, y: np.ndarray, idx: np.ndarray, delay: int = 0) -> tuple[np.ndarray, np.ndarray]:
    return np.stack([X[i - WINDOW + 1 + delay:i + 1 + delay] for i in idx]), y[idx]


def val_start(y: np.ndarray) -> int:
    ok = np.where(y >= 0)[0]
    return int(ok[int(len(ok) * (1 - VAL_SHARE))])


def f1_table(y: np.ndarray, p: np.ndarray) -> tuple[dict, float]:
    out = {}
    for k, name in enumerate(LABELS):
        tp, fp, fn = int(((p == k) & (y == k)).sum()), int(((p == k) & (y != k)).sum()), int(((p != k) & (y == k)).sum())
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out[name] = (prec, rec, 2 * prec * rec / max(prec + rec, 1e-9), tp + fn)
    present = [v[2] for v in out.values() if v[3] > 0]
    return out, float(np.mean(present)) if present else 0.0


def print_scores(title: str, y: np.ndarray, p: np.ndarray) -> float:
    tab, macro = f1_table(y, p)
    print(f"\n{title}: {len(y)} steps of {STRIDE / 7.5:.2f} s, macro F1 {macro:.3f} (over {sum(v[3] > 0 for v in tab.values())} labels present)")
    print(f"  {'label':11} {'n':>5} {'prec':>6} {'rec':>6} {'F1':>6}")
    for name, (pr, rc, f1, n) in tab.items():
        if n or pr:
            print(f"  {name:11} {n:5} {pr:6.2f} {rc:6.2f} {f1:6.2f}")
    return macro


def flip(X: np.ndarray) -> np.ndarray:
    Y = X.copy()
    Y[..., FLIP_NEG] *= -1
    return Y


def gather(a) -> tuple:
    Xtr, ytr, Xva, yva = [], [], [], []
    for name in train_pieces():
        z, y = load_piece(name)
        X = with_long(z["X"]) if a.long else z["X"]
        ok = np.where(y >= 0)[0]
        ok = ok[(ok >= WINDOW - 1) & (ok < len(y) - a.delay)]
        cut = val_start(y)
        tr = ok[::2] if a.all_data else ok[ok < cut - WINDOW - a.delay][::2]
        va = ok[:0] if a.all_data else ok[ok >= cut][::STRIDE]
        for (Xs, ys), idx in (((Xtr, ytr), tr), ((Xva, yva), va)):
            if len(idx):
                w, l = windows(X, y, idx, a.delay)
                Xs.append(w); ys.append(l)
        since = "" if a.all_data else f" (from {datetime.fromtimestamp(z['ts'][cut]):%H:%M:%S})"
        print(f"{name}: train {len(tr)} windows, val {len(va)}{since}")
    if not Xtr:
        sys.exit("no labelled train pieces yet")
    cat = lambda v: np.concatenate(v) if v else None
    return cat(Xtr), cat(ytr), cat(Xva), cat(yva)


def fit(Xtr, ytr, Xva, yva, mean, std, a, seed: int) -> tuple[dict, int, float | None]:
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_net(n_features=Xtr.shape[2], delay=a.delay).to(dev)
    counts = np.bincount(ytr, minlength=len(LABELS)).astype(float)
    wts = np.where(counts > 0, (counts.sum() / np.maximum(counts, 1)) ** a.wpow, 0.0)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(wts / wts[wts > 0].mean(), dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-3)
    Xt = torch.tensor((Xtr - mean) / std, dtype=torch.float32)
    Xf = torch.tensor((flip(Xtr) - mean) / std, dtype=torch.float32) if a.flip else None
    yt = torch.tensor(ytr)
    Xv = None if Xva is None else torch.tensor((Xva - mean) / std, dtype=torch.float32, device=dev)
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            xb = Xt[b]
            if Xf is not None:
                xb = torch.where((torch.rand(len(b)) < 0.5)[:, None, None], Xf[b], xb)
            xb = xb.to(dev) + 0.05 * torch.randn(xb.shape, device=dev)
            loss = loss_fn(net(xb), yt[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        if Xv is None:
            continue
        net.eval()
        with torch.no_grad():
            pv = net(Xv).argmax(1).cpu().numpy()
        _, macro = f1_table(yva, pv)
        if macro > best:
            best, best_ep, bad = macro, ep + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if bad >= a.patience:
            break
    if Xv is None:
        return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, a.epochs, None
    return best_state, best_ep, best


def train(a) -> dict | None:
    import torch
    Xtr, ytr, Xva, yva = gather(a)
    flat = Xtr.reshape(-1, Xtr.shape[2])
    mean, std = flat.mean(0), flat.std(0) + 1e-6
    print("train labels:", {LABELS[k]: int(v) for k, v in sorted(Counter(ytr.tolist()).items())})
    if yva is not None:
        print("val labels:  ", {LABELS[k]: int(v) for k, v in sorted(Counter(yva.tolist()).items())})
    states, eps = [], []
    for s in range(a.seed, a.seed + a.seeds):
        st, ep, best = fit(Xtr, ytr, Xva, yva, mean, std, a, s)
        states.append(st); eps.append(ep)
        print(f"seed {s}: " + (f"{ep} epochs on all labels" if best is None else f"best epoch {ep}, val macro F1 {best:.3f} (raw windows)"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"states": states, "mean": mean, "std": std, "window": WINDOW, "stride": STRIDE, "smooth": SMOOTH,
                "min_bout_s": 2.0, "labels": LABELS, "delay": a.delay, "long": a.long, "flip": a.flip, "fallback": features_fallback(), "hold_s": a.hold,
                "epochs": eps, "all_data": a.all_data}, a.out)
    print(f"-> {a.out}")
    return None if a.all_data else val_report(a.out)


def runs(ts: np.ndarray, y: np.ndarray, t0: float, t1: float) -> list[tuple[float, float, int]]:
    dt = float(ts[1] - ts[0])
    out = []
    for i in np.where((ts >= t0) & (ts < t1) & (y >= 0))[0]:
        if out and out[-1][2] == y[i] and out[-1][3] == i - 1:
            out[-1][1], out[-1][3] = ts[i], i
        else:
            out.append([ts[i], ts[i], int(y[i]), i])
    return [(s, e + dt, k) for s, e, k, _ in out]


def bout_f1(hand: list, system: list) -> float:
    used, tp = set(), 0
    for s, e, k in system:
        for j, (hs, he, hk) in enumerate(hand):
            if j in used or hk != k:
                continue
            inter = min(e, he) - max(s, hs)
            if inter > 0 and inter >= 0.5 * (max(e, he) - min(s, hs)):
                used.add(j)
                tp += 1
                break
    p, r = tp / max(len(system), 1), tp / max(len(hand), 1)
    return 2 * p * r / max(p + r, 1e-9)


def val_report(weights: str) -> dict:
    ys, ps, hand, system = [], [], [], []
    for name in train_pieces():
        z, y = load_piece(name)
        lab, step, bouts = replay(weights, z)
        cut = val_start(y)
        keep = step & (y >= 0) & (lab >= 0) & (np.arange(len(y)) >= cut)
        ys.append(y[keep]); ps.append(lab[keep])
        t0 = z["ts"][cut]
        t1 = z["ts"][np.where(y >= 0)[0][-1]] + (z["ts"][1] - z["ts"][0])
        hand += runs(z["ts"], y, t0, t1)
        system += [(max(b.start.timestamp(), t0), min(b.end.timestamp(), t1), LABELS.index(b.label)) for b in bouts
                   if b.end.timestamp() > t0 and b.start.timestamp() < t1]
        print(f"  {name} val part: macro F1 {f1_table(y[keep], lab[keep])[1]:.3f}")
    y, p = np.concatenate(ys), np.concatenate(ps)
    macro = print_scores("val parts pooled (smoothed, as live)", y, p)
    bf = bout_f1(hand, system)
    print(f"bouts: hand {len(hand)}, system {len(system)}, bout F1 {bf:.3f} (same label, overlap >= half the union)")
    missed, false = int(((p == UNKNOWN) & (y != UNKNOWN)).sum()), int(((p != UNKNOWN) & (y == UNKNOWN)).sum())
    print(f"accuracy {np.mean(p == y):.3f}; unknown on a civet the labeller saw {missed}, behaviour where the labeller had unknown {false}")
    tab, _ = f1_table(y, p)
    return {"macro": macro, "bout_f1": bf, "n_bouts": len(system), "f1": {k: v[2] for k, v in tab.items() if v[3]},
            "acc": float(np.mean(p == y)), "missed": missed, "false": false}


def replay(weights: str, z) -> tuple[np.ndarray, np.ndarray, list]:
    from civetwatch.zones import load_zones
    r = BehaviourRunner(weights, load_zones(load_config(str(ROOT / "config/april5_24h.yaml"))))
    lab, conf, bouts, step = np.full(len(z["ts"]), -1), np.zeros(len(z["ts"])), [], np.zeros(len(z["ts"]), bool)
    for i, (t, f, v, c) in enumerate(zip(z["ts"], z["X"], z["visible"], z["cam"])):
        if c:
            r.camera = str(c)
        b = r.step(datetime.fromtimestamp(t), f, bool(v))
        step[i] = r.n % r.stride == 0 and len(r.buf) >= r.window
        if b:
            bouts.append(b)
        if r.label:
            lab[i], conf[i] = r.label
    last = r.bouts.close()
    if last:
        bouts.append(last)
    if r.delay:
        lab = np.concatenate([lab[r.delay:], np.full(r.delay, -1)])
        step = np.concatenate([step[r.delay:], np.zeros(r.delay, bool)])
    return lab, step, bouts


def run_piece(weights: str, name: str, val_only: bool = False) -> None:
    z, y = load_piece(name)
    if y is None:
        sys.exit(f"{name}: no labels")
    lab, step, bouts = replay(weights, z)
    keep = step & (y >= 0) & (lab >= 0)
    if val_only:
        ok = np.where(y >= 0)[0]
        keep &= np.arange(len(y)) >= ok[int(len(ok) * (1 - VAL_SHARE))]
    title = f"{name} {'val part' if val_only else ''} (smoothed, as live)"
    print_scores(title, y[keep], lab[keep])
    if val_only:
        return
    dt = float(np.median(np.diff(z["ts"])))
    m = (y >= 0) & (lab >= 0)
    print(f"\ntime budget ({m.sum() * dt / 60:.1f} min): minutes per behaviour, hand vs system")
    for k, n in enumerate(LABELS):
        h, s = (y[m] == k).sum() * dt / 60, (lab[m] == k).sum() * dt / 60
        if h or s:
            print(f"  {n:11} hand {h:6.1f}  system {s:6.1f}  diff {100 * (s - h) / h if h else float('inf'):+5.0f}%")
    from civetwatch.db import Store
    cfg = load_config(str(ROOT / "config/april5_24h.yaml"))
    db = RES / f"events_{name}.sqlite"
    db.unlink(missing_ok=True)
    store = Store(str(db), cfg)
    with open(RES / f"events_{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["start", "end", "duration_s", "behaviour", "camera", "confidence"])
        for b in bouts:
            store.add_event(b.camera or "", b.label, b.start, b.end, b.conf)
            w.writerow([f"{b.start:%H:%M:%S}", f"{b.end:%H:%M:%S}", round((b.end - b.start).total_seconds(), 1), b.label, b.camera, round(b.conf, 2)])
    store.close()
    print(f"\n{len(bouts)} bouts -> {db.name}, events_{name}.csv")
    timeline(name, z, y, lab)


def prelabel(weights: str, name: str) -> None:
    if not name.startswith("train"):
        sys.exit("prelabel is for train pieces only: the test list must be labelled blind")
    path = ROOT / "labelling" / f"behaviour_{name}.csv"
    if read_labels(path):
        sys.exit(f"{path.name} already has behaviour rows: not overwritten")
    s0, e0 = PIECES[name]
    _, _, bouts = replay(weights, np.load(FEAT / f"feat_{name}.npz"))
    rows = [(s0, "START", "")]
    for i, b in enumerate(bouts):
        rows.append((s0 if i == 0 else f"{b.start:%H:%M:%S}", b.label, "?" if b.conf < 0.6 else ""))
    rows.append((e0, "END", ""))
    for p in (path, OUT / f"prelabel_{name}.csv"):
        with open(p, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows([("time", "behaviour", "note"), *rows])
    print(f"{len(bouts)} draft bouts ({sum(r[2] == '?' for r in rows)} marked '?') -> {path}")


def timeline(name: str, z, y: np.ndarray, lab: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    t = np.array([datetime.fromtimestamp(v) for v in z["ts"]])
    fig, ax = plt.subplots(figsize=(16, 3.2))
    rooms = sorted({r for r in z["room"] if r})
    for row, (vals, title) in enumerate([(y, "hand"), (lab, "system")]):
        for k in range(len(LABELS)):
            m = vals == k
            if m.any():
                ax.fill_between(t, row + 0.1, row + 0.9, where=m, color=cmap(k), step="post", linewidth=0,
                                label=LABELS[k] if row == 0 else None)
    room_idx = np.array([rooms.index(r) if r else -1 for r in z["room"]])
    ax.plot(t, 2.1 + 0.8 * (room_idx + 0.5) / max(len(rooms), 1), color="black", linewidth=0.8)
    ax.set_yticks([0.5, 1.5, 2.5], ["hand", "system", "room (log)"])
    ax.set_title(f"Behaviour timeline {name}")
    ax.legend(ncol=11, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(RES / f"timeline_{name}.png", dpi=120)
    print(f"timeline -> {RES / f'timeline_{name}.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "train", "val", "eval", "prelabel"])
    ap.add_argument("--piece", help="one piece (extract / eval)")
    ap.add_argument("--config", default="config/april5_24h.yaml")
    ap.add_argument("--cache", default=str(ROOT / "outputs" / "behaviour_kpts"))
    ap.add_argument("--out", default=str(ROOT / "models" / "behaviour_v2.pt"))
    ap.add_argument("--epochs", type=int, default=60, help="max epochs (fixed count with --all-data)")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--delay", type=int, default=0, help="ticks of look-ahead: the label describes this many ticks back")
    ap.add_argument("--long", action="store_true", help="append LongTerm features (60 s speed, drift, ...)")
    ap.add_argument("--flip", action="store_true", help="mirror augmentation")
    ap.add_argument("--seeds", type=int, default=1, help="nets averaged")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-data", action="store_true", help="train on every label, no val (final model)")
    ap.add_argument("--fallback", action="store_true", help="extract: untracked detections stand in while A01 is HIDDEN (Observer)")
    ap.add_argument("--features", default=str(OUT), help="folder of the feat_<piece>.npz files")
    ap.add_argument("--hold", type=float, default=0.0, help="s out of sight that keep the label (labelling rule: 2)")
    ap.add_argument("--results", default=str(OUT), help="eval: folder for events and timeline")
    ap.add_argument("--wpow", type=float, default=0.5, help="class weight = inverse frequency ** wpow")
    a = ap.parse_args()
    global FEAT, RES
    FEAT, RES = Path(a.features), Path(a.results)
    FEAT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    if a.cmd == "extract":
        extract(a)
    elif a.cmd == "train":
        train(a)
    elif a.cmd == "val":
        val_report(a.out)
    else:
        if not a.piece:
            sys.exit(f"{a.cmd} needs --piece")
        (prelabel if a.cmd == "prelabel" else run_piece)(a.out, a.piece)


if __name__ == "__main__":
    main()
