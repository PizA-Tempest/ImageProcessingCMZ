from datetime import datetime, timedelta

from civetwatch.handoff import HIDDEN, PENDING, VISIBLE, HandoffConfig, HandoffManager, Observation
from civetwatch.zones import Placement

ADJ = [["R1", "R2"], ["R2", "R3"], ["R3", "R4"]]
T0 = datetime(2026, 4, 5, 0, 0, 0)
DT = 1 / 7.5


def ob(room, kind="floor", to=None, age=10, conf=0.8, station=None, occluder=None, cam=None):
    return Observation(cam or "C" + room[1], room, 1, conf, (0, 0, 10, 10),
                       Placement(kind, f"D{to}" if to else None, to, station, occluder), age)


def run(h, seq):
    t = 0.0
    states = []
    for i, (until, obs) in enumerate(seq):
        while t < until - 1e-9:
            states.append((t, h.update(T0 + timedelta(seconds=t), obs)))
            t += DT
    return h.s


def test_first_sighting_sets_room():
    h = HandoffManager(ADJ)
    s = run(h, [(1, [ob("R2")])])
    assert (s.state, s.room) == (VISIBLE, "R2")


def test_door_crossing_with_overlap():
    h = HandoffManager(ADJ)
    run(h, [(2, [ob("R1")]),
            (3, [ob("R1", "door", "R2")]),
            (3.5, [ob("R1", "door", "R2"), ob("R2", "door", "R1")]),
            (4.5, [ob("R1", "door", "R2"), ob("R2")])])
    assert (h.s.state, h.s.room) == (VISIBLE, "R2")


def test_door_crossing_gap_goes_pending_then_neighbour():
    h = HandoffManager(ADJ)
    run(h, [(2, [ob("R2")]), (3, [ob("R2", "door", "R3")]), (4.5, [])])
    assert (h.s.state, h.s.room, h.s.to_room) == (PENDING, "R2", "R3")
    run(h, [(5, [ob("R3")])])
    assert (h.s.state, h.s.room) == (VISIBLE, "R3")


def test_pending_times_out_to_hidden_door():
    h = HandoffManager(ADJ, HandoffConfig(t_handoff_s=10))
    run(h, [(1, [ob("R3")]), (2, [ob("R3", "door", "R4")]), (15, [])])
    assert (h.s.state, h.s.room, h.s.reason) == (HIDDEN, "R3", "door")


def test_turn_back_at_door():
    h = HandoffManager(ADJ)
    run(h, [(1, [ob("R2")]), (2, [ob("R2", "door", "R1")]), (4, []), (5, [ob("R2")])])
    assert (h.s.state, h.s.room) == (VISIBLE, "R2")


def test_lost_away_from_door_is_hidden_with_reason():
    h = HandoffManager(ADJ)
    run(h, [(1, [ob("R2", station="sleep_box")]), (3, [])])
    assert (h.s.state, h.s.room, h.s.reason) == (HIDDEN, "R2", "sleep_box")
    h = HandoffManager(ADJ)
    run(h, [(1, [ob("R3", occluder="plant1")]), (3, [])])
    assert (h.s.state, h.s.reason) == (HIDDEN, "occluder")


def test_one_tick_dropout_does_not_hand_off():
    h = HandoffManager(ADJ)
    run(h, [(2, [ob("R2")]), (2.2, [ob("R3", "door", "R2", age=5)]), (3, [ob("R2")])])
    assert (h.s.state, h.s.room) == (VISIBLE, "R2")
    assert all(c["room"] == "R2" for c in h.changes)


def test_no_jump_to_non_neighbour():
    h = HandoffManager(ADJ)
    run(h, [(1, [ob("R1")]), (3, []), (4, [ob("R3", "door", "R2")])])
    assert h.s.room == "R1"


def test_hidden_slips_through_door():
    h = HandoffManager(ADJ)
    run(h, [(1, [ob("R4")]), (3, []), (4, [ob("R3", "door", "R4", age=8)])])
    assert (h.s.state, h.s.room) == (VISIBLE, "R3")


def test_hidden_ignores_young_door_track():
    h = HandoffManager(ADJ)
    run(h, [(1, [ob("R4")]), (3, []), (3.2, [ob("R3", "door", "R4", age=2)])])
    assert (h.s.state, h.s.room) == (HIDDEN, "R4")


def test_recover_after_missed_handoff():
    h = HandoffManager(ADJ, HandoffConfig(recover_after_s=5))
    run(h, [(1, [ob("R1")]), (3, []), (9.5, [ob("R3")])])
    assert (h.s.state, h.s.room, h.s.reason) == (VISIBLE, "R3", "recovered")


def test_stream_lost():
    h = HandoffManager(ADJ)
    run(h, [(1, [ob("R2")])])
    h.update(T0 + timedelta(seconds=1.2), [], lost_rooms={"R2"})
    assert (h.s.state, h.s.room, h.s.reason) == (HIDDEN, "R2", "stream_lost")


def test_young_tracks_ignored():
    h = HandoffManager(ADJ)
    run(h, [(1, [ob("R2", age=1)])])
    assert h.s.state == "UNKNOWN"


def test_door_track_takes_over_when_our_room_loses_it():
    h = HandoffManager(ADJ, HandoffConfig(door_track_s=3))
    run(h, [(2, [ob("R2")]),
            (4, [ob("R2"), ob("R3", "door", "R2")]),
            (4.5, [ob("R3", "door", "R2")]),
            (5, [ob("R3")])])
    assert (h.s.state, h.s.room, h.s.reason) == (VISIBLE, "R3", "door_track")


def test_door_track_off_goes_hidden():
    h = HandoffManager(ADJ, HandoffConfig(door_track_s=0))
    run(h, [(2, [ob("R2")]),
            (4, [ob("R2"), ob("R3", "door", "R2")]),
            (4.5, [ob("R3", "door", "R2")]),
            (6, [ob("R3")])])
    assert (h.s.state, h.s.room) == (HIDDEN, "R2")


def hob(room, hind, kind="door", to=None):
    o = ob(room, kind, to)
    o.hind = (hind, "D23", "R2" if hind == "R3" else "R3")
    return o


def test_hind_end_decides_at_the_gate():
    h = HandoffManager(ADJ, HandoffConfig(hind_window_s=1.0, hind_min_votes=3))
    run(h, [(2, [ob("R3")]), (4, [hob("R3", "R2", to="R2"), ob("R2", "door", "R3", cam="C2")])])
    assert (h.s.state, h.s.room, h.s.reason) == (VISIBLE, "R2", "hind_end")


def test_hind_end_blocks_overlap_switch():
    h = HandoffManager(ADJ, HandoffConfig(hind_window_s=1.0, hind_min_votes=3))
    run(h, [(2, [ob("R2")]), (6, [ob("R2", "door", "R3"), hob("R3", "R2", kind="floor")])])
    assert (h.s.state, h.s.room) == (VISIBLE, "R2")


def test_hind_end_cant_tell_holds_room():
    h = HandoffManager(ADJ, HandoffConfig(hind_window_s=1.0, hind_min_votes=3))
    run(h, [(2, [ob("R2")]), (6, [ob("R2", "door", "R3"), hob("R3", None, kind="floor")])])
    assert (h.s.state, h.s.room) == (VISIBLE, "R2")
