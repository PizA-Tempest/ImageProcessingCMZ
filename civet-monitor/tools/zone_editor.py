import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

COLORS = {"floor": (0, 200, 0), "doors": (0, 160, 255), "gates": (0, 0, 255), "occluders": (255, 160, 0),
          "stations": (200, 0, 200), "seethrough": (160, 160, 160), "ignore": (90, 90, 90)}
KEYS = {ord("d"): "doors", ord("g"): "gates", ord("o"): "occluders", ord("s"): "stations",
        ord("t"): "seethrough", ord("i"): "ignore"}


def load_frame(a):
    if a.frame:
        return cv2.imread(a.frame)
    cap = cv2.VideoCapture(a.video, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_POS_MSEC, a.at * 1000.0)
    ok, img = cap.read()
    if not ok:
        raise SystemExit("cannot read frame")
    return img


def draw(img, cam, pts, scale):
    out = img.copy()
    over = img.copy()
    zones = [("floor", "floor", cam["floor"])] if cam.get("floor") else []
    for kind in ("doors", "gates", "occluders", "stations", "seethrough", "ignore"):
        for z in cam.get(kind, []):
            zones.append((kind, z["id"] + (f"->{z['to_room']}" if "to_room" in z else ""), z["poly"]))
    for kind, label, poly in zones:
        p = np.int32(poly)
        cv2.fillPoly(over, [p], COLORS[kind])
        cv2.polylines(out, [p], True, COLORS[kind], 2)
        c = p.mean(0).astype(int)
        cv2.putText(out, label, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    out = cv2.addWeighted(over, 0.25, out, 0.75, 0)
    if pts:
        cv2.polylines(out, [np.int32(pts)], False, (0, 255, 255), 2)
        for p in pts:
            cv2.circle(out, tuple(map(int, p)), 5, (0, 255, 255), -1)
    return cv2.resize(out, None, fx=scale, fy=scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", default="config/zones.json")
    ap.add_argument("--cam", required=True)
    ap.add_argument("--frame")
    ap.add_argument("--video")
    ap.add_argument("--at", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=0.6, help="display scale (1920x1080 -> 1152x648)")
    a = ap.parse_args()
    path = Path(a.zones)
    data = json.loads(path.read_text(encoding="utf-8"))
    cam = data["cameras"].setdefault(a.cam, {"room_id": input("room id for this camera: "), "floor": []})
    img = load_frame(a)
    pts: list[list[int]] = []
    win = f"zones {a.cam}"

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append([int(x / a.scale), int(y / a.scale)])
        elif ev == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    dirty = False
    while True:
        cv2.imshow(win, draw(img, cam, pts, a.scale))
        k = cv2.waitKey(30) & 0xFF
        quit_after = False
        if k == ord("q"):
            if dirty and input("save changes? [y/N] ").lower().startswith("y"):
                k, quit_after = ord("w"), True
            else:
                break
        if k == ord("c"):
            pts.clear()
        elif k == ord("f") and len(pts) >= 3:
            cam["floor"] = pts.copy()
            pts.clear()
            dirty = True
        elif k in KEYS and len(pts) >= 3:
            kind = KEYS[k]
            z = {"id": input(f"{kind} id: ").strip(), "poly": pts.copy()}
            if kind == "doors":
                z["to_room"] = input("leads to room id: ").strip()
            elif kind == "stations":
                z["type"] = input("type (food/water/sleep_box/...): ").strip()
            elif kind == "seethrough":
                z["sees"] = input("which room/area it shows: ").strip()
            lst = cam.setdefault(kind, [])
            lst[:] = [o for o in lst if o["id"] != z["id"]] + [z]
            pts.clear()
            dirty = True
        elif k == ord("x"):
            zid = input("zone id to delete: ").strip()
            for kind in KEYS.values():
                cam[kind] = [o for o in cam.get(kind, []) if o["id"] != zid]
            dirty = True
        if k == ord("w"):
            shutil.copy(path, path.with_suffix(".json.bak"))
            path.write_text(json.dumps(data, indent=1), encoding="utf-8")
            print(f"saved {path} (backup: {path.with_suffix('.json.bak')})")
            dirty = False
            if quit_after:
                break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
