import unittest

import numpy as np

from recognizer.config import (
    LayoutConfig,
    PlayerRiverLayoutConfig,
    RelativeRegion,
    RiverTileSlotConfig,
    default_config,
)
from recognizer.models import (
    ObservedTileRecognition,
    PlayerRiverRecognition,
    RecognitionResult,
    TileMatch,
)
from recognizer.recognizer import TileRecognizer, crop_river_slots
from recognizer.stability import KeyRegionSnapshot, result_key
from recognizer.visible_tiles import collect_visible_tiles


def _slot(x=0, y=0, width=1, height=1, row=0, column=0, is_riichi=False):
    return RiverTileSlotConfig(
        RelativeRegion(x, y, width, height),
        row=row,
        column=column,
        is_riichi=is_riichi,
    )


def _river_layout(seat, region=None, slots=None):
    slots = slots or [_slot()]
    return PlayerRiverLayoutConfig(
        seat=seat,
        region=region or RelativeRegion(0, 0, 1, 1),
        tile_count=len(slots),
        tiles=slots,
    )


def _result(rivers):
    return RecognitionResult(
        hand=["1m"],
        draw=None,
        dora_indicators=[],
        meld_tiles=[],
        confidence=0.9,
        matches=[],
        rivers=rivers,
    )


class _FakeCandidates:
    def __init__(self, candidate_sets):
        self.templates = {"placeholder": np.zeros((1, 1, 3), dtype=np.uint8)}
        self._candidate_sets = iter(candidate_sets)

    def match_candidates(self, tile, limit=2):
        return list(next(self._candidate_sets))[:limit]


class RiverTests(unittest.TestCase):
    def test_explicit_slots_cross_rows_and_allow_incomplete_last_row(self):
        rows, columns = np.indices((12, 18))
        region = np.stack((rows, columns, rows * 18 + columns), axis=2)
        slots = [
            _slot(0 / 18, 0 / 12, 4 / 18, 5 / 12, row=0, column=0),
            _slot(6 / 18, 0 / 12, 4 / 18, 5 / 12, row=0, column=1),
            _slot(12 / 18, 0 / 12, 6 / 18, 5 / 12, row=0, column=2),
            _slot(1 / 18, 7 / 12, 4 / 18, 5 / 12, row=1, column=0),
            _slot(8 / 18, 7 / 12, 5 / 18, 5 / 12, row=1, column=1),
        ]

        tiles = crop_river_slots(region, slots)

        self.assertEqual(len(tiles), 5)
        np.testing.assert_array_equal(tiles[0], region[0:5, 0:4])
        np.testing.assert_array_equal(tiles[2], region[0:5, 12:18])
        np.testing.assert_array_equal(tiles[3], region[7:12, 1:5])
        np.testing.assert_array_equal(tiles[4], region[7:12, 8:13])

    def test_four_seats_have_fixed_result_order_and_one_failure_is_local(self):
        recognizer = TileRecognizer.__new__(TileRecognizer)
        recognizer.config = default_config()
        recognizer.config.layout.rivers = [
            _river_layout("self", slots=[_slot(), _slot(row=0, column=1)]),
            _river_layout("right"),
            _river_layout("across"),
            _river_layout("left"),
        ]
        recognizer.templates = _FakeCandidates(
            [
                [TileMatch("1m", 0.95), TileMatch("2m", 0.50)],
                [TileMatch("east", 0.92), TileMatch("south", 0.40)],
                [TileMatch("3p", 0.60), TileMatch("4p", 0.58)],
                [TileMatch("5s", 0.91), TileMatch("6s", 0.50)],
                [TileMatch("white", 0.90), TileMatch("green", 0.40)],
                [TileMatch("red", 0.89), TileMatch("north", 0.40)],
            ]
        )
        tile = np.zeros((4, 4, 3), dtype=np.uint8)
        recognizer.extract_tiles = lambda frame: ([tile], [], [], [])
        recognizer.extract_opponent_meld_tiles = lambda frame: {}
        recognizer.extract_river_tiles = lambda frame: {
            "self": [tile, tile], "right": [tile], "across": [tile], "left": [tile]
        }

        result = recognizer.recognize(tile)

        self.assertEqual([river.seat for river in result.rivers], ["self", "right", "across", "left"])
        self.assertEqual([item.name for item in result.rivers[0].tiles], ["east", None])
        self.assertIsNotNone(result.rivers[0].error)
        self.assertEqual(result.rivers[1].tiles[0].name, "5s")
        self.assertEqual(result.rivers[2].tiles[0].name, "white")
        self.assertEqual(result.rivers[3].tiles[0].name, "red")
        self.assertEqual(result.hand, ["1m"])
        self.assertEqual(result.confidence, 0.95)

    def test_riichi_state_is_in_result_key_and_scores_are_not(self):
        def river(is_riichi, score):
            match = TileMatch("east", score)
            tile = ObservedTileRecognition("east", match, candidates=[match], is_riichi=is_riichi)
            return PlayerRiverRecognition("self", [tile])

        plain = result_key(_result([river(False, 0.91)]))
        same_with_other_score = result_key(_result([river(False, 0.51)]))
        riichi = result_key(_result([river(True, 0.91)]))

        self.assertEqual(plain, same_with_other_score)
        self.assertNotEqual(plain, riichi)
        self.assertEqual([seat for seat, _ in plain[-2]], ["self", "right", "across", "left"])

    def test_any_small_river_region_change_changes_snapshot(self):
        layout = LayoutConfig(
            hand_region=RelativeRegion(0, 0, 0.1, 0.1),
            draw_region=RelativeRegion(0, 0, 0.1, 0.1),
            dora_region=RelativeRegion(0, 0, 0.1, 0.1),
            hand_tile_count=0,
            draw_tile_count=0,
            dora_tile_count=0,
            rivers=[
                _river_layout("self", RelativeRegion(0.0, 0.0, 0.1, 0.1)),
                _river_layout("right", RelativeRegion(0.3, 0.0, 0.1, 0.1)),
                _river_layout("across", RelativeRegion(0.6, 0.0, 0.1, 0.1)),
                _river_layout("left", RelativeRegion(0.9, 0.0, 0.1, 0.1)),
            ],
        )
        base = np.zeros((100, 100, 3), dtype=np.uint8)
        original = KeyRegionSnapshot.from_frame(base, layout)

        for seat, x1 in (("self", 0), ("right", 30), ("across", 60), ("left", 90)):
            with self.subTest(seat=seat):
                changed = base.copy()
                changed[0:10, x1:x1 + 10] = 255
                self.assertFalse(
                    original.is_equivalent(KeyRegionSnapshot.from_frame(changed, layout))
                )

    def test_visible_tiles_collects_only_known_river_tiles_once(self):
        known = ObservedTileRecognition("east", TileMatch("east", 0.9))
        unknown = ObservedTileRecognition(None, TileMatch("south", 0.6))
        result = _result([
            PlayerRiverRecognition("self", [known, unknown]),
            PlayerRiverRecognition("self", [known]),
        ])

        self.assertEqual(collect_visible_tiles(result), ["east"])


if __name__ == "__main__":
    unittest.main()
