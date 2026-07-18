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

        self.assertEqual(title, "建议操作")
        self.assertIn("操作：切 9万", detail)
        self.assertIn("0 向听  ·  有效牌 7 枚", detail)
        self.assertIn("进张：中、5筒", detail)
        self.assertIn("备选：东", detail)

    def test_overlay_handles_no_recommendation(self):
        title, detail = format_overlay_suggestion(HandAnalysis(shanten=-1, recommendations=[]))

        self.assertEqual(title, "建议操作")
        self.assertIn("操作：跳过", detail)
        self.assertIn("尚未判断", detail)

    def test_overlay_handles_empty_non_winning_analysis(self):
        title, detail = format_overlay_suggestion(HandAnalysis(shanten=2, recommendations=[]))

        self.assertEqual(title, "建议操作")
        self.assertEqual(detail, "操作：跳过\n当前向听 2")

    def test_waiting_stage_explicitly_recommends_skip(self):
        analysis = HandAnalysis(
            shanten=2,
            recommendations=[DiscardRecommendation("-", 2, 20, ["2m", "3s"], "wait")],
        )

        title, detail = format_overlay_suggestion(analysis)

        self.assertEqual(title, "建议操作")
        self.assertIn("操作：跳过（等待进张）", detail)
        self.assertIn("2 向听  ·  有效牌 20 枚", detail)

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

        title, detail = format_overlay_suggestion(analysis, decision)

        self.assertEqual(title, "建议操作")
        self.assertIn("操作：切 9万 / 立直（条件式）", detail)
        self.assertIn("点棒/余牌满足时可考虑立直", detail)
        self.assertIn("默听保留改良", detail)

    def test_overlay_explicitly_lists_legal_riichi(self):
        analysis = HandAnalysis(
            shanten=1,
            recommendations=[DiscardRecommendation("9m", 0, 8, ["5s"], "best")],
        )
        value = ValueEstimate(("riichi",), 1, 0)
        evaluation = ActionEvaluation(
            ActionCandidate("riichi", discard_tile="9m"), True, "legal", 0, 8,
            ("5s",), "similar", "high", value, "recommended", (),
        )

        title, detail = format_overlay_suggestion(
            analysis, DecisionReport((evaluation,), evaluation.action)
        )

        self.assertEqual(title, "建议操作")
        self.assertIn("操作：切 9万 / 立直", detail)
        self.assertNotIn("条件式", detail)

    def test_call_window_lists_skip_calls_and_kan_uncertainty(self):
        analysis = HandAnalysis(shanten=1, recommendations=[])
        value = ValueEstimate((), 0, 0)
        actions = (
            ActionEvaluation(ActionCandidate("skip"), True, "legal", 1, 8, (), "similar", "high", value, "recommended", ()),
            ActionEvaluation(ActionCandidate("chi", "3m", ("1m", "2m"), source="left"), True, "legal", 1, 6, (), "lower", "high", value, "consider", ()),
            ActionEvaluation(ActionCandidate("pon", "red", ("red", "red"), source="right"), True, "legal", 1, 6, (), "lower", "high", value, "skip_preferred", ()),
            ActionEvaluation(ActionCandidate("minkan", "red", ("red",) * 3, source="right"), True, "legal", 1, 6, (), "unknown", "high", value, "consider", ()),
            ActionEvaluation(ActionCandidate("chi", "3m", ("2m", "4m"), source="right"), False, "illegal", 1, 0, (), "unknown", "high", value, "illegal", ()),
        )
        event = RiverDiscardEvent(7, "red", "right", 1, 3)

        title, detail = format_overlay_suggestion(
            analysis, DecisionReport(actions, actions[0].action), event
        )

        self.assertEqual(title, "建议操作")
        self.assertIn("右家切中", detail)
        self.assertIn("跳过", detail)
        self.assertIn("吃（1万 2万）", detail)
        self.assertNotIn("吃（2万 4万）", detail)
        self.assertIn("碰", detail)
        self.assertIn("杠", detail)
        self.assertNotIn("明杠", detail)
        self.assertLess(detail.index("吃"), detail.index("碰"))
        self.assertLess(detail.index("碰"), detail.index("杠"))
        self.assertLess(detail.index("杠"), detail.index("跳过"))
        self.assertIn("岭上牌未知", detail)
        self.assertNotIn("胡率", detail)


if __name__ == "__main__":
    unittest.main()
