import unittest

import numpy as np

from mahjong.analyzer import analyze_hand
from recognizer.config import (
    MeldConfig,
    MeldTileSlotConfig,
    PlayerMeldLayoutConfig,
    PlayerRiverLayoutConfig,
    RelativeRegion,
    RiverTileSlotConfig,
    default_config,
)
from recognizer.models import TileMatch
from recognizer.recognizer import TileRecognizer
from recognizer.visible_tiles import collect_visible_tiles


class _PixelTemplates:
    """Deterministic matcher that lets the production crop pipeline drive results."""

    def __init__(self, encoded):
        self.templates = {"synthetic": np.zeros((1, 1, 3), dtype=np.uint8)}
        self.encoded = encoded

    def match_candidates(self, tile, limit=2):
        key = tuple(int(channel) for channel in tile[0, 0])
        name, score = self.encoded[key]
        return [
            TileMatch(name, score),
            TileMatch("9s" if name != "9s" else "8s", max(0.0, score - 0.05)),
        ][:limit]


def _region(x, y, width, height, frame_width=400, frame_height=200):
    return RelativeRegion(
        x / frame_width,
        y / frame_height,
        width / frame_width,
        height / frame_height,
    )


def _slot(x, width=20, orientation="upright", frame_width=60):
    return MeldTileSlotConfig(
        RelativeRegion(x / frame_width, 0, width / frame_width, 1),
        orientation=orientation,
    )


class FullTablePipelineTests(unittest.TestCase):
    def test_combined_frame_recognition_visible_tiles_and_ukeire(self):
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        encoded = {}
        next_code = 1

        def paint(x, y, width, height, name, score):
            nonlocal next_code
            color = (next_code, next_code + 40, next_code + 80)
            next_code += 1
            frame[y:y + height, x:x + width] = color
            encoded[color] = (name, score)

        hand = ["1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m"]
        core_scores = [0.91 + index * 0.001 for index in range(13)]
        for index, name in enumerate(hand):
            paint(index * 20, 0, 20, 20, name, core_scores[index])
        paint(200, 0, 20, 20, "east", core_scores[10])
        paint(220, 0, 20, 20, "1m", core_scores[11])
        paint(240, 0, 20, 20, "1m", core_scores[12])

        for index, name in enumerate(["south", "west", "north"]):
            paint(index * 20, 30, 20, 20, name, 0.84 + index * 0.01)
        opponent_names = {
            60: ["2p", "3p", "5p"],
            80: ["6p", "7p", "8p"],
            100: ["2s", "3s", "4s"],
        }
        for y, names in opponent_names.items():
            for index, name in enumerate(names):
                paint(index * 20, y, 20, 20, name, 0.82 + index * 0.01)

        river_names = {
            "self": ("green", "white"),
            "right": ("red", "south"),
            "across": ("9s", "8s"),
            "left": ("9p", "4m"),
        }
        river_y = {"self": 125, "right": 140, "across": 155, "left": 170}
        for seat, names in river_names.items():
            y = river_y[seat]
            paint(0, y, 20, 10, names[0], 0.88)
            paint(20, y, 20, 10, names[1], 0.60 if seat == "left" else 0.87)

        config = default_config()
        config.layout.hand_region = _region(0, 0, 200, 20)
        config.layout.draw_region = _region(200, 0, 20, 20)
        config.layout.dora_region = _region(220, 0, 40, 20)
        config.layout.hand_tile_count = 10
        config.layout.draw_tile_count = 1
        config.layout.dora_tile_count = 2
        config.layout.meld_region = _region(0, 30, 60, 20)
        config.layout.melds = [MeldConfig("pon", [_slot(0), _slot(20), _slot(40)])]
        config.layout.meld_tile_count = 3
        config.layout.open_meld_count = 1
        config.layout.opponent_melds = [
            PlayerMeldLayoutConfig(
                seat,
                _region(0, y, 60, 20),
                3,
                [MeldConfig("pon", [_slot(0), _slot(20), _slot(40)])],
            )
            for seat, y in (("right", 60), ("across", 80), ("left", 100))
        ]
        config.layout.rivers = []
        for seat in ("self", "right", "across", "left"):
            slots = [
                RiverTileSlotConfig(RelativeRegion(0, 0, 0.5, 1), row=0, column=0),
                RiverTileSlotConfig(
                    RelativeRegion(0.5, 0, 0.5, 1),
                    orientation="rotated_cw" if seat == "right" else "upright",
                    row=0,
                    column=1,
                    is_riichi=seat == "right",
                ),
            ]
            config.layout.rivers.append(
                PlayerRiverLayoutConfig(seat, _region(0, river_y[seat], 40, 10), 2, slots)
            )

        recognizer = TileRecognizer.__new__(TileRecognizer)
        recognizer.config = config
        recognizer.templates = _PixelTemplates(encoded)

        extracted_rivers = recognizer.extract_river_tiles(frame)
        self.assertEqual(extracted_rivers["self"][1].shape, (10, 20, 3))
        self.assertEqual(extracted_rivers["right"][1].shape, (20, 10, 3))

        result = recognizer.recognize(frame)

        self.assertEqual(result.hand, hand)
        self.assertEqual(result.draw, "east")
        self.assertEqual(result.dora_indicators, ["1m", "1m"])
        self.assertEqual([player.seat for player in result.opponent_melds], ["right", "across", "left"])
        self.assertEqual([river.seat for river in result.rivers], ["self", "right", "across", "left"])
        self.assertEqual([tile.name for tile in result.rivers[1].tiles], ["red", "south"])
        self.assertTrue(result.rivers[1].tiles[1].is_riichi)
        self.assertEqual([tile.name for tile in result.rivers[3].tiles], ["9p", None])
        self.assertEqual(result.rivers[3].tiles[1].match.name, "4m")
        self.assertIsNotNone(result.rivers[3].error)
        self.assertAlmostEqual(result.confidence, sum(core_scores) / len(core_scores))

        visible = collect_visible_tiles(result)
        expected_visible = (
            ["1m", "1m"]
            + ["south", "west", "north"]
            + ["2p", "3p", "5p"]
            + ["6p", "7p", "8p"]
            + ["2s", "3s", "4s"]
            + ["green", "white", "red", "south", "9s", "8s", "9p"]
        )
        self.assertEqual(visible, expected_visible)
        self.assertEqual(visible.count("1m"), 2)
        self.assertNotIn("4m", visible)

        concealed = [*result.hand, result.draw]
        baseline = analyze_hand(concealed, open_meld_count=1)
        corrected = analyze_hand(concealed, open_meld_count=1, visible_tiles=visible)
        baseline_east = next(item for item in baseline.recommendations if item.discard == "east")
        corrected_east = next(item for item in corrected.recommendations if item.discard == "east")
        self.assertEqual(baseline_east.effective_tiles, ["1m", "4m", "7m"])
        self.assertEqual(baseline_east.ukeire_count, 8)
        self.assertEqual(corrected_east.ukeire_count, 6)


if __name__ == "__main__":
    unittest.main()
