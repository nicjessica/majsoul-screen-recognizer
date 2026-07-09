import unittest

from mahjong.analyzer import analyze_hand
from mahjong.shanten import calculate_shanten
from mahjong.tiles import tiles_to_counts


class MahjongAnalysisTests(unittest.TestCase):
    def test_completed_hand_is_minus_one_shanten(self):
        counts = tiles_to_counts(
            ["1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s", "east", "east", "east", "red", "red"]
        )
        self.assertEqual(calculate_shanten(counts), -1)

    def test_tenpai_hand_is_zero_shanten(self):
        counts = tiles_to_counts(
            ["1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s", "east", "east", "east", "red"]
        )
        self.assertEqual(calculate_shanten(counts), 0)

    def test_analyzer_returns_sorted_recommendations_for_14_tiles(self):
        analysis = analyze_hand(
            ["1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s", "east", "east", "red", "red", "9m"]
        )
        self.assertGreaterEqual(len(analysis.recommendations), 1)
        first = analysis.recommendations[0]
        self.assertLessEqual(first.resulting_shanten, analysis.recommendations[-1].resulting_shanten)

    def test_invalid_tile_count(self):
        with self.assertRaises(ValueError):
            analyze_hand(["1m", "2m"])


if __name__ == "__main__":
    unittest.main()

