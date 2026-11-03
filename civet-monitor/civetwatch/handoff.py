from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .zones import Placement

VISIBLE, PENDING, HIDDEN, UNKNOWN = "VISIBLE", "PENDING", "HIDDEN", "UNKNOWN"


@dataclass
class Observation:
    cam_id: str
    room: str
    track_id: int
    conf: float
    box: tuple
    place: Placement
    track_age: int = 1
    hind: tuple | None = None
    kpts: object = None


@dataclass
class RoomState:
    state: str = UNKNOWN
    room: str | None = None
    cam_id: str | None = None
    reason: str | None = None
    since: datetime | None = None
    door: str | None = None
    to_room: str | None = None


@dataclass
class HandoffConfig:
    t_handoff_s: float = 10.0
    lost_after_s: float = 1.0
    recover_after_s: float = 5.0
    min_track_age: int = 2
    door_min_age: int = 4
    min_dwell_s: float = 2.0
    hind_window_s: float = 0.0
    hind_min_votes: int = 3
    door_track_s: float = 0.0


class HandoffManager:
    def __init__(self, adjacency: list[list[str]], cfg: HandoffConfig | None = None):
        self.cfg = cfg or HandoffConfig()
        self.nb: dict[str, set[str]] = {}
        for a, b in adjacency:
            self.nb.setdefault(a, set()).add(b)
            self.nb.setdefault(b, set()).add(a)
        self.s = RoomState()
        self.last_seen: datetime | None = None
        self.last_place: Placement | None = None
        self.last_cam: str | None = None
        self._elsewhere: dict[str, datetime] = {}
        self._hind_votes: list[tuple[datetime, str]] = []
        self._door_tracks: dict[tuple[str, int], tuple[datetime, str]] = {}
        self.changes: list[dict] = []

    def _set(self, t: datetime, state: str, room: str | None, cam: str | None = None,
             reason: str | None = None, door: str | None = None, to_room: str | None = None):
        same = self.s.state == state and self.s.room == room and (
            state == VISIBLE or (self.s.reason == reason and self.s.to_room == to_room))
        if same:
            if cam:
                self.s.cam_id = cam
            return
        self.s = RoomState(state, room, cam, reason, t, door, to_room)
        self.changes.append({"ts": t.isoformat(timespec="milliseconds"), "state": state, "room": room,
                             "camera": cam, "reason": reason or "", "door": door or ""})

    @staticmethod
    def _best(obs: list[Observation]) -> Observation | None:
        return max(obs, key=lambda o: (o.place.kind == "floor", o.conf), default=None)

    def _hidden_reason(self) -> str:
        p = self.last_place
        if p is None:
            return "unknown"
        if p.station == "sleep_box":
            return "sleep_box"
        if p.kind == "gate":
            return "outside_door"
        if p.occluder:
            return "occluder"
        return "unknown"

    def update(self, t: datetime, observations: list[Observation], lost_rooms: set[str] | None = None) -> RoomState:
        cfg = self.cfg
        lost_rooms = lost_rooms or set()
        obs = [o for o in observations if o.track_age >= cfg.min_track_age and o.place.kind in ("floor", "door")]
        by_room: dict[str, list[Observation]] = {}
        for o in obs:
            by_room.setdefault(o.room, []).append(o)
            if o.place.kind == "door":
                self._door_tracks[(o.cam_id, o.track_id)] = (t, o.place.to_room)
        if len(self._door_tracks) > 100:
            self._door_tracks = {k: v for k, v in self._door_tracks.items() if (t - v[0]).total_seconds() <= cfg.door_track_s}

        cur = self.s
        if cur.state == UNKNOWN:
            best = self._best(obs)
            if best:
                self._seen(t, best)
                self._set(t, VISIBLE, best.room, best.cam_id)
            return self.s

        room = cur.room
        here = by_room.get(room, [])

        v, active = self._hind_vote(t, obs)
        if active and v is None and cur.state == VISIBLE:
            src = next((o for o in obs if o.hind), None)
            if src is not None:
                self.last_seen, self.last_cam = t, src.cam_id
            return self.s
        if v is not None and (v == room or v in self.nb.get(room, ())):
            src = next((o for o in obs if o.hind and o.hind[0] == v), None)
            if src is not None:
                self.last_seen, self.last_cam = t, src.cam_id
                self.last_place = Placement("door", src.hind[1], next(r for r in (src.room, src.hind[2]) if r != v))
            if v != room or cur.state != VISIBLE:
                self._set(t, VISIBLE, v, self.last_cam, "hind_end")
            self._elsewhere.clear()
            return self.s

        if cur.state == VISIBLE:
            if here:
                best = self._best(here)
                self._seen(t, best)
                dwell_ok = cur.since is None or (t - cur.since).total_seconds() >= cfg.min_dwell_s
                for nroom in (self.nb.get(room, ()) if dwell_ok else ()):
                    for o in by_room.get(nroom, []):
                        if o.place.kind == "floor" and best.place.kind == "door" and best.place.to_room == nroom:
                            self._seen(t, o)
                            self._set(t, VISIBLE, nroom, o.cam_id, "door_overlap")
                            return self.s
                self._set(t, VISIBLE, room, best.cam_id)
                self._elsewhere.clear()
                return self.s
            p = self.last_place
            dwell_ok = cur.since is None or (t - cur.since).total_seconds() >= cfg.min_dwell_s
            if dwell_ok and p is not None and p.kind == "door" and p.to_room in self.nb.get(room, ()):
                cands = [o for o in by_room.get(p.to_room, []) if o.place.kind == "floor"]
                if cands:
                    best = self._best(cands)
                    self._seen(t, best)
                    self._set(t, VISIBLE, p.to_room, best.cam_id, "door")
                    self._elsewhere.clear()
                    return self.s
            if dwell_ok and self._door_track(t, room, by_room):
                return self.s
            if room in lost_rooms:
                self._set(t, HIDDEN, room, self.last_cam, "stream_lost")
                self._recover(t, by_room)
                return self.s
            if self.last_seen and (t - self.last_seen).total_seconds() >= cfg.lost_after_s:
                if p is not None and p.kind == "door":
                    self._set(t, PENDING, room, self.last_cam, "door", p.zone_id, p.to_room)
                else:
                    self._set(t, HIDDEN, room, self.last_cam, self._hidden_reason())
            self._recover(t, by_room)
            return self.s

        if cur.state == PENDING:
            if here:
                best = self._best(here)
                self._seen(t, best)
                self._set(t, VISIBLE, room, best.cam_id)
                return self.s
            cands = [o for o in by_room.get(cur.to_room, [])
                     if o.place.kind == "floor" or o.track_age >= cfg.door_min_age]
            if cands:
                best = self._best(cands)
                self._seen(t, best)
                self._set(t, VISIBLE, cur.to_room, best.cam_id, "door_pending")
                self._elsewhere.clear()
                return self.s
            if self._try_handoff(t, room, by_room, require_door=True):
                return self.s
            if (t - cur.since).total_seconds() > cfg.t_handoff_s:
                self._set(t, HIDDEN, room, self.last_cam, "door")
            elif cur.to_room in lost_rooms:
                self._set(t, HIDDEN, cur.to_room, None, "stream_lost", cur.door)
            self._recover(t, by_room)
            return self.s

        if cur.state == HIDDEN:
            if here:
                best = self._best(here)
                self._seen(t, best)
                self._set(t, VISIBLE, room, best.cam_id)
                return self.s
            if self._try_handoff(t, room, by_room, require_door=True) or self._door_track(t, room, by_room):
                return self.s
            self._recover(t, by_room)
            return self.s
        return self.s

    def _seen(self, t: datetime, o: Observation):
        self.last_seen = t
        self.last_place = o.place
        self.last_cam = o.cam_id

    def _try_handoff(self, t, room, by_room, require_door: bool = True) -> bool:
        for nroom in sorted(self.nb.get(room, ())):
            cands = by_room.get(nroom, [])
            if require_door:
                cands = [o for o in cands if o.place.kind == "door" and o.place.to_room == room
                         and o.track_age >= self.cfg.door_min_age]
            if cands:
                best = self._best(cands)
                self._seen(t, best)
                self._set(t, VISIBLE, nroom, best.cam_id, "door_unseen")
                self._elsewhere.clear()
                return True
        return False

    def _hind_vote(self, t, obs) -> tuple[str | None, bool]:
        if self.cfg.hind_window_s <= 0:
            return None, False
        now = [o.hind[0] for o in obs if o.hind]
        if now:
            self._hind_votes.append((t, max(set(now), key=lambda r: (now.count(r), r is not None))))
        self._hind_votes = [(ts, r) for ts, r in self._hind_votes if (t - ts).total_seconds() <= self.cfg.hind_window_s]
        rooms = [r for _, r in self._hind_votes if r]
        active = bool(self._hind_votes)
        if len(rooms) < self.cfg.hind_min_votes:
            return None, active
        best = max(set(rooms), key=rooms.count)
        return (best if rooms.count(best) >= 0.7 * len(self._hind_votes) else None), active

    def _door_track(self, t, room, by_room) -> bool:
        if self.cfg.door_track_s <= 0:
            return False
        cands = []
        for nroom in self.nb.get(room, ()):
            for o in by_room.get(nroom, []):
                seen = self._door_tracks.get((o.cam_id, o.track_id))
                if o.place.kind == "floor" and seen and seen[1] == room and (t - seen[0]).total_seconds() <= self.cfg.door_track_s:
                    cands.append(o)
        if not cands:
            return False
        best = self._best(cands)
        self._seen(t, best)
        self._set(t, VISIBLE, best.room, best.cam_id, "door_track")
        self._elsewhere.clear()
        return True

    def _recover(self, t, by_room):
        seen_now = set(by_room)
        for r in list(self._elsewhere):
            if r not in seen_now:
                del self._elsewhere[r]
        for r in seen_now:
            if r == self.s.room:
                continue
            first = self._elsewhere.setdefault(r, t)
            if (t - first).total_seconds() >= self.cfg.recover_after_s:
                best = self._best(by_room[r])
                self._seen(t, best)
                self._set(t, VISIBLE, r, best.cam_id, "recovered")
                self._elsewhere.clear()
                return
