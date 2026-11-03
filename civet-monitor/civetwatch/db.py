from __future__ import annotations

import csv
import sqlite3
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS room   (room_id TEXT PRIMARY KEY, room_name TEXT);
CREATE TABLE IF NOT EXISTS camera (camera_id TEXT PRIMARY KEY, room_id TEXT REFERENCES room, location TEXT);
CREATE TABLE IF NOT EXISTS behavior_event (
  event_id        TEXT PRIMARY KEY,
  camera_id       TEXT REFERENCES camera,
  behavior        TEXT,
  timestamp_start TIMESTAMP,
  timestamp_end   TIMESTAMP,
  duration        REAL,
  confidence      REAL
);
CREATE INDEX IF NOT EXISTS ix_event_start ON behavior_event(timestamp_start);
"""


class Store:
    def __init__(self, db_path: str, cfg: dict):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(db_path, check_same_thread=False)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.executescript(SCHEMA)
        for rid, r in cfg["rooms"].items():
            self.con.execute("INSERT OR REPLACE INTO room VALUES (?, ?)", (rid, r.get("name", rid)))
        for cid, c in cfg["cameras"].items():
            self.con.execute("INSERT OR REPLACE INTO camera VALUES (?, ?, ?)", (cid, c["room"], c.get("location")))
        self.con.commit()

    def add_event(self, camera_id: str, behavior: str, start, end, confidence: float) -> str:
        eid = uuid.uuid4().hex[:16]
        self.con.execute("INSERT INTO behavior_event VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (eid, camera_id, behavior, start.isoformat(), end.isoformat(),
                          (end - start).total_seconds(), confidence))
        self.con.commit()
        return eid

    def close(self):
        self.con.close()


class RoomLog:

    FIELDS = ["ts", "state", "room", "camera", "reason", "door"]

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        self.w.writeheader()

    def write(self, rows):
        for r in rows:
            self.w.writerow(r)
        self.f.flush()

    def close(self):
        self.f.close()
