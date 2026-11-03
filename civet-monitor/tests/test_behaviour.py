from datetime import datetime, timedelta

from civetwatch.behaviour import BoutBuilder

T0 = datetime(2026, 4, 5, 0, 0, 0)


def feed(b, seq, step=0.5):
    out, t = [], 0.0
    for until, label in seq:
        while t < until - 1e-9:
            bout = b.update(T0 + timedelta(seconds=t), label, 0.9, "CAM01")
            if bout:
                out.append(bout)
            t += step
    return out


def test_bout_opens_after_min_duration_and_closes_on_change():
    b = BoutBuilder(min_s=2.0)
    out = feed(b, [(10, "walking"), (20, "resting")])
    assert [x.label for x in out] == ["walking"]
    assert out[0].start == T0 and out[0].end == T0 + timedelta(seconds=10)
    assert b.cur.label == "resting"


def test_short_flicker_does_not_open_a_bout():
    b = BoutBuilder(min_s=2.0)
    out = feed(b, [(10, "walking"), (11, "standing"), (20, "walking")])
    assert out == [] and b.cur.label == "walking"
    assert b.cur.end == T0 + timedelta(seconds=19.5)


from civetwatch.behaviour import Observer
from civetwatch.handoff import Observation, RoomState
from civetwatch.zones import Placement


def raw(cam="CAM03", room="ROOM03", kind="floor", conf=0.4):
    return Observation(cam, room, -1, conf, (0, 0, 10, 10), Placement(kind), 0)


def test_observer_off_ignores_raw():
    o = Observer(fallback=False)
    hidden = RoomState("HIDDEN", "ROOM03", "CAM03")
    assert all(o(hidden, [], [raw()]) is None for _ in range(5))


def test_observer_needs_three_of_five_ticks():
    o = Observer(fallback=True)
    hidden = RoomState("HIDDEN", "ROOM03", "CAM03")
    got = [o(hidden, [], [raw()]) for _ in range(4)]
    assert got[:2] == [None, None] and got[2] is not None and got[3] is not None


def test_observer_only_that_rooms_floor_or_door():
    o = Observer(fallback=True)
    hidden = RoomState("HIDDEN", "ROOM03", "CAM03")
    others = [raw(cam="CAM02", room="ROOM02"), raw(kind="seethrough"), raw(kind="ignore"), raw(kind="outside")]
    assert all(o(hidden, [], others) is None for _ in range(6))


def test_observer_never_while_visible():
    o = Observer(fallback=True)
    visible = RoomState("VISIBLE", "ROOM03", "CAM03")
    assert all(o(visible, [], [raw()]) is None for _ in range(6))
