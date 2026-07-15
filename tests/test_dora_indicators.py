import unittest

import numpy as np

from recognizer.config import RelativeRegion, default_config
from recognizer.models import TileMatch
from recognizer.recognizer import RecognitionError, TileRecognizer


class _DoraTemplates:
    def __init__(self, matches: list[TileMatch]) -> None:
        self.templates = {"placeholder": np.zeros((1, 1, 3), dtype=np.uint8)}
        self._matches = iter(matches)

    def match_candidates(self, tile, limit=2):
        first = next(self._matches)
        return [first][:limit]


class DoraIndicatorTests(unittest.TestCase):
    def test_one_revealed_indicator(self):
        self._assert_revealed_indicators(["east"])

    def test_two_revealed_indicators_after_kan(self):
        self._assert_revealed_indicators(["east", "5p"])

    def test_five_revealed_indicators(self):
        self._assert_revealed_indicators(["east", "5p", "red", "9s", "1m"])

    def test_low_confidence_indicator_keeps_existing_hard_failure_semantics(self):
        recognizer, frame = self._recognizer_for(["east", "5p"])
        recognizer.templates = _DoraTemplates([
            TileMatch("east", 0.95),
            TileMatch("5p", 0.60),
        ])

        with self.assertRaises(RecognitionError):
            recognizer.recognize(frame)

    def _assert_revealed_indicators(self, names: list[str]) -> None:
        recognizer, frame = self._recognizer_for(names)

        hand, draw, crops, meld = recognizer.extract_tiles(frame)
        self.assertEqual((hand, draw, meld), ([], [], []))
        self.assertEqual(len(crops), len(names))
        self.assertTrue(all(crop.shape == (4, 6, 3) for crop in crops))
        self.assertEqual(
            [int(crop[0, 0, 0]) for crop in crops],
            list(range(1, len(names) + 1)),
        )

        result = recognizer.recognize(frame)
        self.assertEqual(result.dora_indicators, names)
        self.assertEqual([match.name for match in result.matches], names)
        self.assertAlmostEqual(result.confidence, 0.95)

    @staticmethod
    def _recognizer_for(names: list[str]) -> tuple[TileRecognizer, np.ndarray]:
        config = default_config()
        config.layout.hand_tile_count = 0
        config.layout.draw_tile_count = 0
        config.layout.dora_region = RelativeRegion(0, 0, 1, 1)
        config.layout.dora_tile_count = len(names)

        frame = np.zeros((4, 6 * len(names), 3), dtype=np.uint8)
        for index in range(len(names)):
            frame[:, index * 6 : (index + 1) * 6] = index + 1

        recognizer = TileRecognizer.__new__(TileRecognizer)
        recognizer.config = config
        recognizer.templates = _DoraTemplates(
            [TileMatch(name, 0.95) for name in names]
        )
        return recognizer, frame


if __name__ == "__main__":
    unittest.main()
