import tempfile
import unittest
from pathlib import Path

import numpy as np

from recognizer.config import (
    LayoutConfig,
    MeldConfig,
    MeldTileSlotConfig,
    PlayerMeldLayoutConfig,
    RelativeRegion,
    default_config,
    load_config,
    save_config,
)
from recognizer.models import (
    MeldRecognition,
    MeldTileRecognition,
    PlayerMeldRecognition,
    RecognitionResult,
    TileMatch,
)
from recognizer.recognizer import TileRecognizer, crop_meld_slots
from recognizer.stability import KeyRegionSnapshot, result_key


def _slot(orientation="upright"):
    return MeldTileSlotConfig(RelativeRegion(0, 0, 1, 1), orientation=orientation)


def _player_layout(seat, region=None):
    return PlayerMeldLayoutConfig(
        seat=seat,
        region=region or RelativeRegion(0, 0, 1, 1),
        tile_count=1,
        melds=[MeldConfig(kind="unknown", tiles=[_slot()])],
    )


class _FakeCandidates:
    def __init__(self, candidate_sets):
        self.templates = {"placeholder": np.zeros((1, 1, 3), dtype=np.uint8)}
        self._candidate_sets = iter(candidate_sets)

    def match_candidates(self, tile, limit=2):
        return list(next(self._candidate_sets))[:limit]


class OpponentMeldTests(unittest.TestCase):
    def test_right_across_left_config_round_trip_preserves_canonical_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            config = default_config()
            config.templates_dir = str(Path(tmp) / "templates")
            config.layout.opponent_melds = [
                _player_layout("right"),
                _player_layout("across"),
                _player_layout("left"),
            ]

            save_config(config, path)
            loaded = load_config(path)

            self.assertEqual(
                [player.seat for player in loaded.layout.opponent_melds],
                ["right", "across", "left"],
            )

    def test_rotated_180_slot_is_normalized_and_contiguous(self):
        region = np.arange(4 * 6 * 3, dtype=np.int16).reshape(4, 6, 3)
        melds = [MeldConfig(kind="unknown", tiles=[_slot("rotated_180")])]

        tile = crop_meld_slots(region, melds)[0][0]

        np.testing.assert_array_equal(tile, np.rot90(region, 2))
        self.assertTrue(tile.flags.c_contiguous)

    def test_one_opponent_low_slot_keeps_other_seats_and_core_confidence(self):
        recognizer = TileRecognizer.__new__(TileRecognizer)
        recognizer.config = default_config()
        recognizer.config.layout.opponent_melds = [
            _player_layout("right"),
            _player_layout("across"),
            _player_layout("left"),
        ]
        recognizer.templates = _FakeCandidates(
            [
                [TileMatch("1m", 0.95), TileMatch("2m", 0.50)],
                [TileMatch("east", 0.91), TileMatch("south", 0.40)],
                [TileMatch("3p", 0.60), TileMatch("4p", 0.58)],
                [TileMatch("red", 0.92), TileMatch("green", 0.45)],
            ]
        )
        tile = np.zeros((4, 4, 3), dtype=np.uint8)
        recognizer.extract_tiles = lambda frame: ([tile], [], [], [])
        recognizer.extract_opponent_meld_tiles = lambda frame: {
            "right": [tile],
            "across": [tile],
            "left": [tile],
        }

        result = recognizer.recognize(tile)

        self.assertEqual(
            [player.seat for player in result.opponent_melds],
            ["right", "across", "left"],
        )
        self.assertEqual(result.opponent_melds[0].meld_tiles, ["east"])
        self.assertEqual(result.opponent_melds[1].meld_tiles, [])
        self.assertIsNone(result.opponent_melds[1].melds[0].tiles[0].name)
        self.assertEqual(
            [item.name for item in result.opponent_melds[1].melds[0].tiles[0].candidates],
            ["3p", "4p"],
        )
        self.assertEqual(result.opponent_melds[2].meld_tiles, ["red"])
        self.assertEqual(result.hand, ["1m"])
        self.assertEqual(result.confidence, 0.95)

    def test_any_small_opponent_region_change_changes_snapshot(self):
        layout = LayoutConfig(
            hand_region=RelativeRegion(0, 0, 0.5, 0.5),
            draw_region=RelativeRegion(0, 0, 0.1, 0.1),
            dora_region=RelativeRegion(0, 0, 0.1, 0.1),
            hand_tile_count=0,
            draw_tile_count=0,
            dora_tile_count=0,
            opponent_melds=[
                _player_layout("right", RelativeRegion(0.0, 0.0, 0.1, 0.1)),
                _player_layout("across", RelativeRegion(0.45, 0.0, 0.1, 0.1)),
                _player_layout("left", RelativeRegion(0.9, 0.0, 0.1, 0.1)),
            ],
        )
        base = np.zeros((100, 100, 3), dtype=np.uint8)
        original = KeyRegionSnapshot.from_frame(base, layout)

        for seat, x1, x2 in (("right", 0, 10), ("across", 45, 55), ("left", 90, 100)):
            with self.subTest(seat=seat):
                changed = base.copy()
                changed[0:10, x1:x2] = 255
                snapshot = KeyRegionSnapshot.from_frame(changed, layout)
                self.assertFalse(original.is_equivalent(snapshot))

    def test_result_key_uses_fixed_seats_keeps_unknown_and_ignores_scores(self):
        def player(seat, name, score):
            match = TileMatch("candidate", score)
            tile = MeldTileRecognition(name, match, candidates=[match])
            return PlayerMeldRecognition(
                seat=seat,
                meld_tiles=[] if name is None else [name],
                melds=[MeldRecognition("unknown", [tile], score)],
            )

        def result(scores):
            return RecognitionResult(
                hand=["1m"],
                draw=None,
                dora_indicators=[],
                meld_tiles=[],
                confidence=scores[0],
                matches=[],
                opponent_melds=[
                    player("left", "red", scores[1]),
                    player("right", "east", scores[2]),
                    player("across", None, scores[3]),
                ],
            )

        first_key = result_key(result((0.9, 0.8, 0.7, 0.6)))
        second_key = result_key(result((0.1, 0.2, 0.3, 0.4)))

        self.assertEqual(first_key, second_key)
        self.assertEqual(
            [seat for seat, _ in first_key[-1]],
            ["right", "across", "left"],
        )
        self.assertIsNone(first_key[-1][1][1][0][1][0])


if __name__ == "__main__":
    unittest.main()
