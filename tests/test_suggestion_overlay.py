import unittest

from app.suggestion_overlay import format_overlay_suggestion, tile_label
from mahjong.analyzer import DiscardRecommendation, HandAnalysis
from mahjong.decision import ActionCandidate, ActionEvaluation, DecisionReport, ValueEstimate
from recognizer.river_events import RiverDiscardEvent


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

    def test_overlay_marks_riichi_as_conditional_when_round_data_is_missing(self):
        analysis = HandAnalysis(
            shanten=1,
            recommendations=[DiscardRecommendation("9m", 0, 8, ["5s", "8s"], "best")],
        )
        action = ActionCandidate("riichi", discard_tile="9m")
        decision = DecisionReport((ActionEvaluation(
            action=action,
            legal=True,
            legality="unverified",
            resulting_shanten=0,
            ukeire_count=8,
            effective_tiles=("5s", "8s"),
            relative_win_chance="similar",
            win_chance_uncertainty="high",
            value=ValueEstimate(("riichi",), 1, 0),
            recommendation="consider",
            reasons=("点棒或牌山余量未识别",),
        ),), action)

        _, detail = format_overlay_suggestion(analysis, decision)

        self.assertIn("点棒/余牌满足时可考虑", detail)
        self.assertIn("默听保留改良", detail)

    def test_call_window_lists_skip_calls_and_kan_uncertainty(self):
        analysis = HandAnalysis(shanten=1, recommendations=[])
        value = ValueEstimate((), 0, 0)
        actions = (
            ActionEvaluation(ActionCandidate("skip"), True, "legal", 1, 8, (), "similar", "high", value, "recommended", ()),
            ActionEvaluation(ActionCandidate("pon", "red", ("red", "red"), source="right"), True, "legal", 1, 6, (), "lower", "high", value, "skip_preferred", ()),
            ActionEvaluation(ActionCandidate("minkan", "red", ("red",) * 3, source="right"), True, "legal", 1, 6, (), "unknown", "high", value, "consider", ()),
        )
        event = RiverDiscardEvent(7, "red", "right", 1, 3)

        title, detail = format_overlay_suggestion(
            analysis, DecisionReport(actions, actions[0].action), event
        )

        self.assertEqual(title, "鸣牌窗口  右家切中")
        self.assertIn("跳过", detail)
        self.assertIn("碰", detail)
        self.assertIn("明杠", detail)
        self.assertIn("岭上牌未知", detail)
        self.assertNotIn("胡率", detail)


if __name__ == "__main__":
    unittest.main()
