import unittest

import numpy as np

from recognizer.config import (
    PlayerMeldLayoutConfig, PlayerRiverLayoutConfig, RelativeRegion, default_config,
)
from recognizer.models import (
    MeldRecognition,
    MeldTileRecognition,
    PlayerMeldRecognition,
    ObservedTileRecognition,
    PlayerRiverRecognition,
    RecognitionResult,
)
from recognizer.stability import KeyRegionSnapshot, RecognitionStabilizer, result_key


def _result(hand=("1m",), meld_name="2p"):
    melds = []
    meld_tiles = []
    if meld_name is not None:
        melds = [
            MeldRecognition(
                kind="pon",
                tiles=[MeldTileRecognition(meld_name, None)],
                confidence=0.9,
            )
        ]
        meld_tiles = [meld_name]
    return RecognitionResult(
        hand=list(hand),
        draw=None,
        dora_indicators=["east"],
        meld_tiles=meld_tiles,
        confidence=0.9,
        matches=[],
        melds=melds,
    )


class KeyRegionSnapshotTests(unittest.TestCase):
    def test_snapshot_uses_only_active_regions_and_detects_mad(self):
        config = default_config()
        config.layout.hand_region = RelativeRegion(0, 0, 0.5, 1)
        config.layout.draw_tile_count = 0
        config.layout.dora_tile_count = 0
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        changed_outside = frame.copy()
        changed_outside[:, 20:] = 255

        first = KeyRegionSnapshot.from_frame(frame, config.layout)
        outside = KeyRegionSnapshot.from_frame(changed_outside, config.layout)

        self.assertEqual(first.regions["hand"].shape, (32, 32))
        self.assertEqual(first.regions.keys(), {"hand"})
        self.assertTrue(first.is_equivalent(outside))

        changed_inside = frame.copy()
        changed_inside[:, :20] = 10
        inside = KeyRegionSnapshot.from_frame(changed_inside, config.layout)
        self.assertFalse(first.is_equivalent(inside, max_mad=2.0))

    def test_snapshot_includes_active_meld_region(self):
        config = default_config()
        config.layout.draw_tile_count = 0
        config.layout.dora_tile_count = 0
        config.layout.meld_region = RelativeRegion(0.5, 0, 0.5, 1)
        config.layout.meld_tile_count = 3

        snapshot = KeyRegionSnapshot.from_frame(
            np.zeros((20, 20, 3), dtype=np.uint8), config.layout
        )

        self.assertEqual(snapshot.regions.keys(), {"hand", "meld"})

    def test_snapshot_includes_each_active_opponent_meld_region(self):
        config = default_config()
        config.layout.draw_tile_count = 0
        config.layout.dora_tile_count = 0
        config.layout.opponent_melds = [
            PlayerMeldLayoutConfig("right", RelativeRegion(0.5, 0, 0.5, 0.5), 3),
            PlayerMeldLayoutConfig("left", RelativeRegion(0.5, 0.5, 0.5, 0.5), 3),
        ]
        snapshot = KeyRegionSnapshot.from_frame(np.zeros((20, 20, 3), dtype=np.uint8), config.layout)
        self.assertEqual(snapshot.regions.keys(), {"hand", "meld:right", "meld:left"})

    def test_snapshot_includes_active_rivers(self):
        config = default_config()
        config.layout.draw_tile_count = config.layout.dora_tile_count = 0
        config.layout.rivers = [
            PlayerRiverLayoutConfig("self", RelativeRegion(0, 0, 0.5, 0.5), 1),
            PlayerRiverLayoutConfig("across", RelativeRegion(0.5, 0.5, 0.5, 0.5), 1),
        ]
        snapshot = KeyRegionSnapshot.from_frame(np.zeros((20, 20, 3), dtype=np.uint8), config.layout)
        self.assertEqual(snapshot.regions.keys(), {"hand", "river:self", "river:across"})


class RecognitionStabilizerTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = KeyRegionSnapshot({"hand": np.zeros((32, 32), dtype=np.uint8)})

    def test_result_key_includes_structured_meld_kind_and_slot_names(self):
        first = _result(meld_name="2p")
        second = _result(meld_name="3p")
        third = RecognitionResult(
            **{**first.__dict__, "melds": [MeldRecognition("chi", first.melds[0].tiles, 0.9)]}
        )

        self.assertNotEqual(result_key(first), result_key(second))
        self.assertNotEqual(result_key(first), result_key(third))

    def test_result_key_includes_opponents_in_fixed_seat_order_not_scores(self):
        base = _result(meld_name=None)
        tile = MeldTileRecognition("3p", None)
        right = PlayerMeldRecognition("right", ["3p"], [MeldRecognition("pon", [tile], 0.9)])
        changed = PlayerMeldRecognition("right", ["4p"], [MeldRecognition("pon", [MeldTileRecognition("4p", None)], 0.1)])
        first = RecognitionResult(**{**base.__dict__, "opponent_melds": [right]})
        reordered = RecognitionResult(**{**base.__dict__, "opponent_melds": [
            PlayerMeldRecognition("left", [], [], None), right
        ]})
        second = RecognitionResult(**{**base.__dict__, "opponent_melds": [changed]})
        self.assertEqual(result_key(first), result_key(reordered))
        self.assertNotEqual(result_key(first), result_key(second))

    def test_result_key_includes_river_names_and_riichi_not_scores(self):
        base = _result(meld_name=None)
        first_tile = ObservedTileRecognition("2p", None, is_riichi=False)
        riichi_tile = ObservedTileRecognition("2p", None, is_riichi=True)
        moved_tile = ObservedTileRecognition("2p", None, row=1, column=2)
        first = RecognitionResult(**{**base.__dict__, "rivers": [PlayerRiverRecognition("left", [first_tile])]})
        same = RecognitionResult(**{**base.__dict__, "rivers": [PlayerRiverRecognition(
            "left", [ObservedTileRecognition("2p", None, candidates=[], is_riichi=False)]
        )]})
        riichi = RecognitionResult(**{**base.__dict__, "rivers": [PlayerRiverRecognition("left", [riichi_tile])]})
        moved = RecognitionResult(**{**base.__dict__, "rivers": [PlayerRiverRecognition("left", [moved_tile])]})
        self.assertEqual(result_key(first), result_key(same))
        self.assertNotEqual(result_key(first), result_key(riichi))
        self.assertNotEqual(result_key(first), result_key(moved))

    def test_first_recognition_and_two_reuses_publish_on_third_observation(self):
        stabilizer = RecognitionStabilizer(required_observations=3)
        result = _result()

        self.assertTrue(stabilizer.needs_recognition(self.snapshot))
        first = stabilizer.observe_success(self.snapshot, result)
        self.assertFalse(stabilizer.needs_recognition(self.snapshot))
        second = stabilizer.observe_reused()
        third = stabilizer.observe_reused()

        self.assertEqual(first.pending_count, 1)
        self.assertIsNone(second.published_result)
        self.assertTrue(second.reused)
        self.assertEqual(third.published_result, result)
        self.assertTrue(third.just_published)
        self.assertTrue(third.reused)

    def test_changed_result_restarts_pending_count(self):
        stabilizer = RecognitionStabilizer(required_observations=2)
        stabilizer.observe_success(self.snapshot, _result(hand=("1m",)))

        update = stabilizer.observe_success(self.snapshot, _result(hand=("2m",)))

        self.assertEqual(update.pending_count, 1)
        self.assertIsNone(update.published_result)

    def test_error_does_not_advance_and_forces_retry(self):
        stabilizer = RecognitionStabilizer(required_observations=2)
        result = _result()
        stabilizer.observe_success(self.snapshot, result)

        error = stabilizer.observe_error()

        self.assertEqual(error.pending_count, 1)
        self.assertTrue(stabilizer.needs_recognition(self.snapshot))
        published = stabilizer.observe_success(self.snapshot, result)
        self.assertEqual(published.pending_count, 2)
        self.assertTrue(published.just_published)

    def test_reset_and_reuse_without_success(self):
        stabilizer = RecognitionStabilizer()
        with self.assertRaises(RuntimeError):
            stabilizer.observe_reused()
        stabilizer.observe_success(self.snapshot, _result())
        stabilizer.reset()
        self.assertTrue(stabilizer.needs_recognition(self.snapshot))
        self.assertIsNone(stabilizer.published_result)

    def test_required_observations_must_be_positive(self):
        with self.assertRaises(ValueError):
            RecognitionStabilizer(0)

    def test_manual_publish_seeds_stable_baseline(self):
        stabilizer = RecognitionStabilizer(required_observations=3)
        result = _result()

        update = stabilizer.publish_success(self.snapshot, result)

        self.assertEqual(update.published_result, result)
        self.assertTrue(update.just_published)
        self.assertEqual(update.pending_count, 3)
        self.assertFalse(stabilizer.needs_recognition(self.snapshot))


if __name__ == "__main__":
    unittest.main()
