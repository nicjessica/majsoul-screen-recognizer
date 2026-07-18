import unittest

from recognizer.models import (
    ObservedTileRecognition,
    PlayerRiverRecognition,
    RecognitionResult,
    TileMatch,
)
from recognizer.river_events import RiverDiscardEvent, RiverEventTracker


def _tile(name, row=0, column=0, **kwargs):
    return ObservedTileRecognition(name, kwargs.pop("match", None), row=row, column=column, **kwargs)


def _result(*rivers):
    return RecognitionResult([], None, [], [], 1.0, [], rivers=list(rivers))


def _river(seat, *tiles, error=None):
    return PlayerRiverRecognition(seat, list(tiles), error)


class RiverEventTrackerTests(unittest.TestCase):
    def test_first_frame_only_establishes_baseline(self):
        tracker = RiverEventTracker()
        initial = _result(_river("right", _tile("3m", 1, 2)))

        self.assertIsNone(tracker.observe(initial))
        self.assertIsNone(tracker.observe(initial))

    def test_unknown_to_known_is_confidence_jitter_and_does_not_emit(self):
        tracker = RiverEventTracker()
        tracker.observe(_result(_river("across", _tile(None, 2, 4))))

        event = tracker.observe(_result(_river("across", _tile("5pr", 2, 4))))

        self.assertIsNone(event)
        self.assertIsNone(tracker.observe(_result(_river("across", _tile("5pr", 2, 4)))))

    def test_absent_coordinate_becoming_known_emits_event(self):
        tracker = RiverEventTracker()
        tracker.observe(_result(_river("left")))

        event = tracker.observe(_result(_river("left", _tile("east", 0, 5))))

        self.assertEqual(event, RiverDiscardEvent(1, "east", "left", 0, 5))

    def test_absent_then_unknown_then_known_never_emits_delayed_event(self):
        tracker = RiverEventTracker()
        tracker.observe(_result(_river("left")))

        self.assertIsNone(tracker.observe(
            _result(_river("left", _tile(None, 0, 5)))
        ))
        self.assertIsNone(tracker.observe(
            _result(_river("left", _tile("east", 0, 5)))
        ))

    def test_self_river_never_emits_event(self):
        tracker = RiverEventTracker()
        tracker.observe(_result(_river("self", _tile(None, 0, 0))))

        self.assertIsNone(tracker.observe(_result(_river("self", _tile("1m", 0, 0)))))

    def test_multiple_additions_do_not_emit_and_advance_baseline(self):
        tracker = RiverEventTracker()
        tracker.observe(_result(_river("right", _tile(None, 0, 0), _tile(None, 0, 1))))
        ambiguous = _result(_river("right", _tile("1p", 0, 0), _tile("2p", 0, 1)))

        self.assertIsNone(tracker.observe(ambiguous))
        self.assertIsNone(tracker.observe(ambiguous))

        later = _result(_river(
            "right", _tile("1p", 0, 0), _tile("2p", 0, 1), _tile("3p", 0, 2)
        ))
        self.assertEqual(tracker.observe(later), RiverDiscardEvent(1, "3p", "right", 0, 2))

    def test_addition_mixed_with_another_slot_change_does_not_emit(self):
        tracker = RiverEventTracker()
        tracker.observe(_result(_river(
            "right", _tile(None, 0, 0), _tile("7m", 0, 1)
        )))
        mixed = _result(_river(
            "right", _tile("6m", 0, 0), _tile("8m", 0, 1)
        ))

        self.assertIsNone(tracker.observe(mixed))
        self.assertIsNone(tracker.observe(mixed))

    def test_known_name_change_does_not_emit_and_advances_baseline(self):
        tracker = RiverEventTracker()
        tracker.observe(_result(_river("left", _tile("4s", 1, 1))))

        self.assertIsNone(tracker.observe(_result(_river("left", _tile("5s", 1, 1)))))
        self.assertIsNone(tracker.observe(_result(_river("left", _tile("5s", 1, 1)))))

    def test_diagnostic_and_confidence_jitter_do_not_emit(self):
        tracker = RiverEventTracker()
        tracker.observe(_result(_river("right", _tile(
            "south", 0, 3, match=TileMatch("south", 0.81), error="old",
            candidates=[TileMatch("west", 0.8)],
        ), error="river old")))
        jitter = _result(_river("right", _tile(
            "south", 0, 3, match=TileMatch("south", 0.95), error=None,
            candidates=[TileMatch("north", 0.7)], is_riichi=True,
        ), error=None))

        self.assertIsNone(tracker.observe(jitter))

    def test_event_ids_are_monotonic_across_reset(self):
        tracker = RiverEventTracker()
        tracker.observe(_result())
        self.assertEqual(tracker.observe(
            _result(_river("right", _tile("white", 0, 0)))
        ).event_id, 1)

        tracker.reset()
        self.assertIsNone(tracker.observe(_result()))
        self.assertEqual(tracker.observe(
            _result(_river("left", _tile("red", 1, 0)))
        ).event_id, 2)


if __name__ == "__main__":
    unittest.main()
