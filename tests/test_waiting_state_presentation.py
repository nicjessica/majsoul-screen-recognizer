import unittest

from app.main_window import MainWindow
from app.suggestion_overlay import format_overlay_suggestion
from mahjong.analyzer import analyze_hand
from mahjong.tiles import TILE_NAMES


class WaitingStatePresentationTests(unittest.TestCase):
    def test_analyzer_contract_distinguishes_waiting_and_discard_counts(self):
        for waiting_count, discard_count, open_meld_count in (
            (13, 14, 0),
            (10, 11, 1),
            (7, 8, 2),
            (4, 5, 3),
            (1, 2, 4),
        ):
            with self.subTest(waiting_count=waiting_count):
                waiting = analyze_hand(
                    list(TILE_NAMES[:waiting_count]),
                    open_meld_count=open_meld_count,
                )
                after_draw = analyze_hand(
                    list(TILE_NAMES[:discard_count]),
                    open_meld_count=open_meld_count,
                )

                self.assertEqual(len(waiting.recommendations), 1)
                self.assertEqual(waiting.recommendations[0].discard, "-")
                self.assertTrue(after_draw.recommendations)
                self.assertTrue(all(item.discard != "-" for item in after_draw.recommendations))

    def test_overlay_labels_waiting_state_without_fake_discard(self):
        analysis = analyze_hand(list(TILE_NAMES[:13]))

        title, detail = format_overlay_suggestion(analysis)

        self.assertEqual(title, "建议操作")
        self.assertIn("操作：跳过（等待进张）", detail)
        self.assertNotIn("切牌", title)
        self.assertNotIn("切 -", f"{title}\n{detail}")
        self.assertIn("有效牌", detail)
        self.assertIn("进张：", detail)

    def test_main_result_text_labels_waiting_state_without_fake_discard(self):
        analysis = analyze_hand(list(TILE_NAMES[:13]))

        text = MainWindow.format_analysis(analysis)

        self.assertIn("等待进张", text)
        self.assertNotIn("切 -", text)


if __name__ == "__main__":
    unittest.main()
