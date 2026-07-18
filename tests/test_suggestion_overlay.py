import unittest

from app.suggestion_overlay import format_overlay_suggestion, tile_label
from mahjong.analyzer import DiscardRecommendation, HandAnalysis


class SuggestionOverlayFormattingTests(unittest.TestCase):
    def test_tile_label_uses_compact_chinese_names(self):
        self.assertEqual(tile_label("9m"), "9万")
        self.assertEqual(tile_label("5pr"), "赤5筒")
        self.assertEqual(tile_label("red"), "中")

    def test_overlay_shows_best_choice_effective_tiles_and_alternatives(self):
        analysis = HandAnalysis(
            shanten=1,
            recommendations=[
                DiscardRecommendation("9m", 0, 7, ["red", "5p"], "best"),
                DiscardRecommendation("east", 1, 4, ["2s"], "second"),
            ],
        )

        title, detail = format_overlay_suggestion(analysis)

        self.assertEqual(title, "首选切牌  9万")
        self.assertIn("0 向听  ·  有效牌 7 枚", detail)
        self.assertIn("进张：中、5筒", detail)
        self.assertIn("备选：东", detail)

    def test_overlay_handles_no_recommendation(self):
        title, detail = format_overlay_suggestion(HandAnalysis(shanten=-1, recommendations=[]))

        self.assertEqual(title, "已完成和牌形")
        self.assertIn("尚未判断", detail)

    def test_overlay_handles_empty_non_winning_analysis(self):
        title, detail = format_overlay_suggestion(HandAnalysis(shanten=2, recommendations=[]))

        self.assertEqual(title, "暂无切牌建议")
        self.assertEqual(detail, "当前向听 2")


if __name__ == "__main__":
    unittest.main()
