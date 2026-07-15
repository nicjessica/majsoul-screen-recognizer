import unittest

import numpy as np

from recognizer.config import LayoutConfig, RelativeRegion
from recognizer.models import RecognitionResult
from recognizer.stability import KeyRegionSnapshot, RecognitionStabilizer


def _result(tile):
    return RecognitionResult(
        hand=[tile],
        draw=None,
        dora_indicators=[],
        meld_tiles=[],
        confidence=0.9,
        matches=[],
    )


def _snapshot(value):
    return KeyRegionSnapshot(
        regions={"hand": np.full((32, 32), value, dtype=np.uint8)}
    )


class StabilityRegressionTests(unittest.TestCase):
    def test_a_a_b_b_b_publishes_only_b_after_third_observation(self):
        stabilizer = RecognitionStabilizer(required_observations=3)
        result_a = _result("1m")
        result_b = _result("2m")

        updates = [
            stabilizer.observe_success(_snapshot(1), result_a),
            stabilizer.observe_success(_snapshot(1), result_a),
            stabilizer.observe_success(_snapshot(2), result_b),
            stabilizer.observe_success(_snapshot(2), result_b),
            stabilizer.observe_success(_snapshot(2), result_b),
        ]

        self.assertTrue(all(update.published_result is None for update in updates[:4]))
        self.assertIs(updates[4].published_result, result_b)
        self.assertTrue(updates[4].just_published)

    def test_published_a_remains_while_b_has_fewer_than_three_observations(self):
        stabilizer = RecognitionStabilizer(required_observations=3)
        result_a = _result("1m")
        result_b = _result("2m")
        for _ in range(3):
            published_a = stabilizer.observe_success(_snapshot(1), result_a)

        first_b = stabilizer.observe_success(_snapshot(2), result_b)
        second_b = stabilizer.observe_success(_snapshot(2), result_b)

        self.assertIs(published_a.published_result, result_a)
        self.assertIs(first_b.published_result, result_a)
        self.assertIs(second_b.published_result, result_a)
        self.assertEqual(second_b.pending_count, 2)

    def test_error_does_not_advance_pending_and_forces_next_recognition(self):
        stabilizer = RecognitionStabilizer(required_observations=3)
        snapshot = _snapshot(1)
        result = _result("1m")
        stabilizer.observe_success(snapshot, result)
        before_error = stabilizer.observe_success(snapshot, result)

        after_error = stabilizer.observe_error()

        self.assertEqual(before_error.pending_count, 2)
        self.assertEqual(after_error.pending_count, 2)
        self.assertIsNone(after_error.published_result)
        self.assertTrue(stabilizer.needs_recognition(snapshot))

    def test_small_draw_or_meld_change_marks_snapshot_changed(self):
        layout = LayoutConfig(
            hand_region=RelativeRegion(0.0, 0.0, 0.8, 1.0),
            draw_region=RelativeRegion(0.8, 0.0, 0.2, 0.2),
            dora_region=RelativeRegion(0.0, 0.0, 0.1, 0.1),
            meld_region=RelativeRegion(0.8, 0.8, 0.2, 0.2),
            hand_tile_count=1,
            draw_tile_count=1,
            dora_tile_count=0,
            meld_tile_count=1,
        )
        base = np.zeros((100, 100, 3), dtype=np.uint8)
        original = KeyRegionSnapshot.from_frame(base, layout)

        draw_changed = base.copy()
        draw_changed[0:20, 80:100] = 255
        meld_changed = base.copy()
        meld_changed[80:100, 80:100] = 255

        self.assertFalse(original.is_equivalent(KeyRegionSnapshot.from_frame(draw_changed, layout)))
        self.assertFalse(original.is_equivalent(KeyRegionSnapshot.from_frame(meld_changed, layout)))

    def test_reset_requires_recognition_for_the_same_frame_again(self):
        stabilizer = RecognitionStabilizer(required_observations=3)
        snapshot = _snapshot(1)
        stabilizer.observe_success(snapshot, _result("1m"))
        self.assertFalse(stabilizer.needs_recognition(snapshot))

        stabilizer.reset()

        self.assertTrue(stabilizer.needs_recognition(snapshot))


if __name__ == "__main__":
    unittest.main()
