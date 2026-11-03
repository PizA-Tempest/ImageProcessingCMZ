from __future__ import annotations

import json
from pathlib import Path

import yaml


def _resolve(base: Path, value):
    if value is None:
        return None
    s = str(value)
    if "://" in s:
        return s
    p = Path(s)
    return str(p if p.is_absolute() else (base / p).resolve())


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    base = path.parent.parent
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for cam in cfg["cameras"].values():
        cam["source"] = _resolve(base, cam.get("source"))
        cam["substream"] = _resolve(base, cam.get("substream"))
    st = cfg["storage"]
    st["db_path"] = _resolve(base, st["db_path"])
    st["room_log_csv"] = _resolve(base, st["room_log_csv"])
    cfg["debug"]["video_dir"] = _resolve(base, cfg["debug"]["video_dir"])
    cfg["_base"] = str(base)
    zones_path = cfg.get("zones_path") or str(base / "config" / "zones.json")
    with open(zones_path, encoding="utf-8") as f:
        cfg["zones"] = json.load(f)
    return cfg
