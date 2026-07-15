import unittest

import numpy as np

from recognizer.config import (
    MeldConfig, MeldTileSlotConfig, PlayerMeldLayoutConfig, PlayerRiverLayoutConfig,
    RelativeRegion, RiverTileSlotConfig, default_config,
)
from recognizer.models import TileMatch
from recognizer.recognizer import (
    RecognitionError, TileRecognizer, crop_meld_slots, crop_river_slots,
)


class _FakeTemplates:
    def __init__(self, matches):
        self.templates = {"placeholder": np.zeros((1, 1, 3), dtype=np.uint8)}
        self._matches = iter(matches)

    def match(self, tile):
        return next(self._matches)

    def match_candidates(self, tile, limit=2):
        first = next(self._matches)
        candidates = [first]
        if first.score < 0.78:
            candidates.append(TileMatch("second", first.score - 0.05))
        return candidates[:limit]


class RecognizerTests(unittest.TestCase):
    def test_low_confidence_hand_still_blocks_result(self):
        recognizer = TileRecognizer.__new__(TileRecognizer)
        recognizer.config = default_config()
        recognizer.templates = _FakeTemplates([TileMatch("1m", 0.60)])
        tile = np.zeros((10, 10, 3), dtype=np.uint8)
        recognizer.extract_tiles = lambda frame: ([tile], [], [], [])

        with self.assertRaises(RecognitionError) as context:
            recognizer.recognize(tile)

        self.assertIn("手牌第 1 张", str(context.exception))
        self.assertIn("1m=0.600", str(context.exception))
        self.assertIn("second=0.550", str(context.exception))

    def test_low_confidence_meld_does_not_block_hand_result(self):
        recognizer = TileRecognizer.__new__(TileRecognizer)
        recognizer.config = default_config()
        recognizer.templates = _FakeTemplates(
            [TileMatch("1m", 0.95), TileMatch("2p", 0.60)]
        )
        tile = np.zeros((10, 10, 3), dtype=np.uint8)
        recognizer.extract_tiles = lambda frame: ([tile], [], [], [tile])

        result = recognizer.recognize(tile)

        self.assertEqual(result.hand, ["1m"])
        self.assertEqual(result.meld_tiles, [])
        self.assertIn("第 1 张副露牌", result.meld_error)
        self.assertEqual([item.name for item in result.melds[0].tiles[0].candidates], ["2p", "second"])
        self.assertEqual(result.confidence, 0.95)

    def test_structured_meld_keeps_successful_slots_and_core_confidence(self):
        recognizer = TileRecognizer.__new__(TileRecognizer)
        recognizer.config = default_config()
        recognizer.config.layout.melds = [
            MeldConfig(
                kind="pon",
                tiles=[
                    MeldTileSlotConfig(RelativeRegion(0, 0, 0.3, 1)),
                    MeldTileSlotConfig(RelativeRegion(0.3, 0, 0.3, 1)),
                    MeldTileSlotConfig(RelativeRegion(0.6, 0, 0.3, 1)),
                ],
            )
        ]
        recognizer.templates = _FakeTemplates(
            [
                TileMatch("1m", 0.95),
                TileMatch("2p", 0.90),
                TileMatch("3p", 0.60),
                TileMatch("4p", 0.80),
            ]
        )
        tile = np.zeros((10, 10, 3), dtype=np.uint8)
        recognizer.extract_tiles = lambda frame: ([tile], [], [], [tile, tile, tile])

        result = recognizer.recognize(tile)

        self.assertEqual(result.meld_tiles, ["2p", "4p"])
        self.assertEqual([tile.name for tile in result.melds[0].tiles], ["2p", None, "4p"])
        self.assertAlmostEqual(result.melds[0].confidence, 0.85)
        self.assertIn("第 1 组第 2 张副露牌", result.meld_error)
        self.assertEqual(
            [item.name for item in result.melds[0].tiles[1].candidates],
            ["3p", "second"],
        )
        self.assertEqual(result.confidence, 0.95)

    def test_crop_meld_slots_respects_gaps_overlap_and_rotation(self):
        region = np.zeros((4, 8, 3), dtype=np.uint8)
        region[:, 0:2] = (10, 20, 30)
        region[:, 3:7] = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        melds = [
            MeldConfig(
                kind="kakan",
                tiles=[
                    MeldTileSlotConfig(RelativeRegion(0, 0, 0.25, 1)),
                    MeldTileSlotConfig(
                        RelativeRegion(0.375, 0, 0.5, 1),
                        orientation="rotated_cw",
                    ),
                    MeldTileSlotConfig(RelativeRegion(0.5, 0, 0.25, 1), stack_level=1),
                ],
            )
        ]

        groups = crop_meld_slots(region, melds)

        self.assertEqual(groups[0][0].shape, (4, 2, 3))
        self.assertTrue(np.all(groups[0][0] == (10, 20, 30)))
        expected_rotated = np.rot90(region[:, 3:7], 1)
        np.testing.assert_array_equal(groups[0][1], expected_rotated)
        np.testing.assert_array_equal(groups[0][2], region[:, 4:6])

    def test_opponent_meld_soft_failure_and_confidence_isolation(self):
        recognizer = TileRecognizer.__new__(TileRecognizer)
        recognizer.config = default_config()
        recognizer.config.layout.opponent_melds = [
            PlayerMeldLayoutConfig("right", RelativeRegion(0, 0, 1, 1), tile_count=2)
        ]
        recognizer.templates = _FakeTemplates([
            TileMatch("1m", 0.95), TileMatch("2p", 0.90), TileMatch("3p", 0.60)
        ])
        tile = np.zeros((10, 10, 3), dtype=np.uint8)
        recognizer.extract_tiles = lambda frame: ([tile], [], [], [])
        recognizer.extract_opponent_meld_tiles = lambda frame: {"right": [tile, tile]}

        result = recognizer.recognize(tile)

        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.opponent_melds[0].meld_tiles, ["2p"])
        self.assertEqual([item.name for item in result.opponent_melds[0].melds[0].tiles], ["2p", None])
        self.assertEqual([item.name for item in result.matches], ["1m", "2p"])

    def test_crop_meld_slots_rotates_180_degrees(self):
        region = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        melds = [MeldConfig("unknown", [
            MeldTileSlotConfig(RelativeRegion(0, 0, 1, 1), "rotated_180")
        ])]
        np.testing.assert_array_equal(crop_meld_slots(region, melds)[0][0], np.rot90(region, 2))

    def test_river_soft_failure_preserves_riichi_and_core_confidence(self):
        recognizer = TileRecognizer.__new__(TileRecognizer)
        recognizer.config = default_config()
        recognizer.config.layout.rivers = [PlayerRiverLayoutConfig(
            "self", RelativeRegion(0, 0, 1, 1), 2,
            [
                RiverTileSlotConfig(RelativeRegion(0, 0, 0.5, 1)),
                RiverTileSlotConfig(RelativeRegion(0.5, 0, 0.5, 1), is_riichi=True),
            ],
        )]
        recognizer.templates = _FakeTemplates([
            TileMatch("1m", 0.95), TileMatch("2p", 0.90), TileMatch("3p", 0.60)
        ])
        tile = np.zeros((10, 10, 3), dtype=np.uint8)
        recognizer.extract_tiles = lambda frame: ([tile], [], [], [])
        recognizer.extract_river_tiles = lambda frame: {"self": [tile, tile]}

        result = recognizer.recognize(tile)
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual([item.name for item in result.rivers[0].tiles], ["2p", None])
        self.assertTrue(result.rivers[0].tiles[1].is_riichi)
        self.assertEqual((result.rivers[0].tiles[1].row, result.rivers[0].tiles[1].column), (0, 0))
        self.assertIn("river:self", result.rivers[0].error)
        self.assertEqual([item.name for item in result.matches], ["1m", "2p"])

    def test_crop_river_slots_uses_explicit_regions_and_rotation(self):
        region = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
        slots = [RiverTileSlotConfig(RelativeRegion(0.5, 0, 0.5, 1), "rotated_180")]
        np.testing.assert_array_equal(crop_river_slots(region, slots)[0], np.rot90(region[:, 2:4], 2))


if __name__ == "__main__":
    unittest.main()
