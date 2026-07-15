import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from recognizer.config import (
    MeldConfig,
    MeldTileSlotConfig,
    PlayerRiverLayoutConfig,
    RelativeRegion,
    RiverTileSlotConfig,
    default_config,
)
from recognizer.models import TileMatch
from recognizer.recognizer import RecognitionError, TileRecognizer
from recognizer.stability import result_key
from recognizer.templates import TemplateLibrary
from recognizer.visible_tiles import collect_visible_tiles


class _FakeCandidateTemplates:
    def __init__(self, candidate_sets):
        self.templates = {"placeholder": np.zeros((1, 1, 3), dtype=np.uint8)}
        self._candidate_sets = iter(candidate_sets)
        self.limits = []

    def match_candidates(self, tile, limit=2):
        self.limits.append(limit)
        return list(next(self._candidate_sets))[:limit]


class RecognitionDiagnosticTests(unittest.TestCase):
    def test_template_candidates_are_score_sorted_and_limited_to_two(self):
        library = TemplateLibrary.__new__(TemplateLibrary)
        library.templates_dir = None
        library.templates = {
            "3m": np.zeros((2, 2, 3), dtype=np.uint8),
            "1m": np.zeros((2, 2, 3), dtype=np.uint8),
            "2m": np.zeros((2, 2, 3), dtype=np.uint8),
        }
        scores = iter([0.60, 0.91, 0.75])
        tile = np.zeros((2, 2, 3), dtype=np.uint8)

        with patch("recognizer.templates.match_score", side_effect=lambda *_: next(scores)):
            candidates = library.match_candidates(tile, limit=2)

        self.assertEqual(candidates, [TileMatch("1m", 0.91), TileMatch("2m", 0.75)])

    def test_core_low_confidence_error_has_area_index_candidates_and_threshold(self):
        recognizer = TileRecognizer.__new__(TileRecognizer)
        recognizer.config = default_config()
        recognizer.templates = _FakeCandidateTemplates(
            [[TileMatch("1m", 0.70), TileMatch("2m", 0.65)]]
        )
        tile = np.zeros((4, 4, 3), dtype=np.uint8)
        recognizer.extract_tiles = lambda frame: ([tile], [], [], [])

        with self.assertRaises(RecognitionError) as context:
            recognizer.recognize(tile)

        message = str(context.exception)
        self.assertIn("手牌第 1 张", message)
        self.assertIn("1m=0.700", message)
        self.assertIn("2m=0.650", message)
        self.assertIn("阈值 0.780", message)
        self.assertEqual(recognizer.templates.limits, [2])

    def test_low_meld_slot_keeps_candidates_and_other_successful_slot(self):
        recognizer = TileRecognizer.__new__(TileRecognizer)
        recognizer.config = default_config()
        slot = MeldTileSlotConfig(RelativeRegion(0, 0, 0.5, 1))
        recognizer.config.layout.melds = [MeldConfig(kind="pon", tiles=[slot, slot])]
        recognizer.templates = _FakeCandidateTemplates(
            [
                [TileMatch("1m", 0.95), TileMatch("9m", 0.40)],
                [TileMatch("2p", 0.70), TileMatch("3p", 0.68)],
                [TileMatch("east", 0.93), TileMatch("south", 0.50)],
            ]
        )
        tile = np.zeros((4, 4, 3), dtype=np.uint8)
        recognizer.extract_tiles = lambda frame: ([tile], [], [], [tile, tile])

        result = recognizer.recognize(tile)

        failed, succeeded = result.melds[0].tiles
        self.assertIsNone(failed.name)
        self.assertEqual(
            failed.candidates,
            [TileMatch("2p", 0.70), TileMatch("3p", 0.68)],
        )
        self.assertIn("第 1 组第 1 张副露牌", failed.error)
        self.assertEqual(succeeded.name, "east")
        self.assertEqual(
            succeeded.candidates,
            [TileMatch("east", 0.93), TileMatch("south", 0.50)],
        )
        self.assertEqual(result.meld_tiles, ["east"])
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(recognizer.templates.limits, [2, 2, 2])

    def test_low_river_slot_reports_coordinates_without_polluting_results(self):
        recognizer = TileRecognizer.__new__(TileRecognizer)
        recognizer.config = default_config()
        recognizer.config.layout.rivers = [PlayerRiverLayoutConfig(
            seat="left",
            region=RelativeRegion(0, 0, 1, 1),
            tile_count=2,
            tiles=[
                RiverTileSlotConfig(RelativeRegion(0, 0, 0.5, 1), row=2, column=4),
                RiverTileSlotConfig(RelativeRegion(0.5, 0, 0.5, 1), row=2, column=5),
            ],
        )]
        recognizer.templates = _FakeCandidateTemplates([
            [TileMatch("1m", 0.95), TileMatch("9m", 0.40)],
            [TileMatch("2p", 0.70), TileMatch("3p", 0.68)],
            [TileMatch("east", 0.93), TileMatch("south", 0.50)],
        ])
        tile = np.zeros((4, 4, 3), dtype=np.uint8)
        recognizer.extract_tiles = lambda frame: ([tile], [], [], [])
        recognizer.extract_river_tiles = lambda frame: {"left": [tile, tile]}

        result = recognizer.recognize(tile)

        failed, succeeded = result.rivers[0].tiles
        self.assertIsNone(failed.name)
        self.assertEqual(succeeded.name, "east")
        for message in (failed.error, result.rivers[0].error):
            self.assertIn("river:left", message)
            self.assertIn("row=2", message)
            self.assertIn("column=4", message)
            self.assertIn("2p=0.700", message)
            self.assertIn("3p=0.680", message)
            self.assertIn("threshold=0.780", message)
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(collect_visible_tiles(result), ["east"])

        changed_failed = replace(
            failed,
            match=TileMatch("2p", 0.10),
            error="different diagnostic text",
            candidates=[TileMatch("2p", 0.10), TileMatch("3p", 0.09)],
        )
        changed_river = replace(
            result.rivers[0],
            tiles=[changed_failed, succeeded],
            error="different aggregate text",
        )
        changed_result = replace(
            result,
            rivers=[changed_river],
            confidence=0.10,
            matches=[],
        )
        self.assertEqual(result_key(result), result_key(changed_result))


if __name__ == "__main__":
    unittest.main()
