from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Placement:
    kind: str
    zone_id: str | None = None
    to_room: str | None = None
    station: str | None = None
    occluder: str | None = None


def _poly(points) -> np.ndarray:
    return np.asarray(points, np.float32).reshape(-1, 1, 2)


class CameraZones:
    def __init__(self, cam_id: str, spec: dict, zone_margin: float = 10.0, door_margin: float = 60.0):
        self.cam_id = cam_id
        self.room = spec["room_id"]
        self.zone_margin = zone_margin
        self.door_margin = door_margin
        self.floor = _poly(spec["floor"])
        self.doors = [(d["id"], d["to_room"], _poly(d["poly"])) for d in spec.get("doors", [])]
        self.gates = [(g["id"], _poly(g["poly"])) for g in spec.get("gates", [])]
        self.seethrough = [(s["id"], s.get("sees"), _poly(s["poly"])) for s in spec.get("seethrough", [])]
        self.ignore = [(z["id"], _poly(z["poly"])) for z in spec.get("ignore", [])]
        self.stations = [(s["id"], s.get("type"), _poly(s["poly"])) for s in spec.get("stations", [])]
        self.occluders = [(o["id"], _poly(o["poly"])) for o in spec.get("occluders", [])]
        self.hind = []
        cx, cy = self.floor.reshape(-1, 2).mean(0)
        for did, to_room, poly in self.doors:
            line = next((d.get("hind_line") for d in spec.get("doors", []) if d["id"] == did), None)
            if line:
                (x1, y1), (x2, y2) = line
                n = np.array([y2 - y1, x1 - x2], float)
                n /= np.linalg.norm(n)
                if np.dot([cx - x1, cy - y1], n) < 0:
                    n = -n
                self.hind.append((did, to_room, poly, np.array([x1, y1], float), n))
        self.hind_kpt, self.hind_min_conf, self.hind_margin, self.hind_near = 4, 0.5, 15.0, 80.0

    @staticmethod
    def _dist(poly: np.ndarray, pt) -> float:
        return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), True)

    @staticmethod
    def ground_point(box) -> tuple[float, float]:
        x1, y1, x2, y2 = box[:4]
        return (0.5 * (x1 + x2), float(y2))

    def hind_room(self, pt, place: Placement, kpts) -> tuple[str, str, str] | None:
        if kpts is None or not self.hind or len(kpts) <= self.hind_kpt:
            return None
        x, y, v = kpts[self.hind_kpt]
        if v < self.hind_min_conf:
            return None
        for did, to_room, poly, p0, n in self.hind:
            if place.zone_id != did and not (place.kind == "floor" and self._dist(poly, pt) >= -self.hind_near):
                continue
            d = float(np.dot([x - p0[0], y - p0[1]], n))
            if abs(d) < self.hind_margin:
                return (None, did, None)
            return (self.room, did, to_room) if d > 0 else (to_room, did, self.room)
        return None

    def place(self, pt) -> Placement:
        in_seethrough = any(self._dist(poly, pt) >= 0 for _, _, poly in self.seethrough)
        best = None
        for did, to_room, poly in self.doors:
            d = self._dist(poly, pt)
            margin = 0.0 if in_seethrough else self.door_margin
            if d >= -margin and (best is None or d > best[0]):
                best = (d, did, to_room)
        if best is not None:
            p = Placement("door", best[1], best[2])
        elif self._dist(self.floor, pt) >= -self.zone_margin:
            p = Placement("floor")
        elif any(self._dist(poly, pt) >= 0 for _, poly in self.gates):
            gid = next(g for g, poly in self.gates if self._dist(poly, pt) >= 0)
            return Placement("gate", gid)
        elif any(self._dist(poly, pt) >= 0 for _, _, poly in self.seethrough):
            sid = next(s for s, _, poly in self.seethrough if self._dist(poly, pt) >= 0)
            return Placement("seethrough", sid)
        elif any(self._dist(poly, pt) >= 0 for _, poly in self.ignore):
            return Placement("ignore")
        else:
            return Placement("outside")
        for sid, _, poly in self.stations:
            if self._dist(poly, pt) >= -self.zone_margin:
                p.station = sid
                break
        for oid, poly in self.occluders:
            if self._dist(poly, pt) >= -3 * self.zone_margin:
                p.occluder = oid
                break
        return p

    def counts_for_room(self, p: Placement) -> bool:
        return p.kind in ("floor", "door")


def load_zones(cfg: dict) -> dict[str, CameraZones]:
    h = cfg.get("handoff", {})
    zm, dm = float(h.get("zone_margin_px", 10)), float(h.get("door_margin_px", 60))
    out = {cid: CameraZones(cid, spec, zm, dm) for cid, spec in cfg["zones"]["cameras"].items()}
    for z in out.values():
        z.hind_kpt = int(h.get("hind_keypoint", 4))
        z.hind_min_conf = float(h.get("hind_min_conf", 0.5))
        z.hind_margin = float(h.get("hind_margin_px", 15))
        z.hind_near = float(h.get("hind_near_px", 80))
    return out
