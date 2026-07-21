import unittest
from types import SimpleNamespace

from app.main_window import MainWindow
from mahjong.analyzer import analyze_hand
from mahjong.shape_routes import ShapeRouteContext, match_shape_routes
from recognizer.config import PlayerMeldLayoutConfig, RelativeRegion, default_config
from recognizer.geometry import ScreenRegion


class MainWindowHelperTests(unittest.TestCase):
    def test_parse_meld_structure_with_orientation_and_stack(self):
        melds = MainWindow._parse_meld_structure(
            "pon rotated_cw upright upright\n"
            "kakan rotated_cw upright upright rotated_cw@1"
        )

        self.assertEqual([meld.kind for meld in melds], ["pon", "kakan"])
        self.assertEqual(melds[0].tiles[0].orientation, "rotated_cw")
        self.assertEqual(melds[1].tiles[3].stack_level, 1)

    def test_parse_opponent_meld_accepts_rotated_180(self):
        melds = MainWindow._parse_meld_structure(
            "pon rotated_180 rotated_180 rotated_180"
        )

        self.assertEqual(melds[0].tiles[0].orientation, "rotated_180")

    def test_parse_meld_structure_rejects_wrong_slot_count(self):
        with self.assertRaises(ValueError):
            MainWindow._parse_meld_structure("pon upright upright")

    def test_format_meld_structure_round_trip(self):
        original = MainWindow._parse_meld_structure(
            "chi upright upright rotated_ccw\nunknown upright"
        )

        formatted = MainWindow._format_meld_structure(original)

        self.assertEqual(MainWindow._parse_meld_structure(formatted), original)

    def test_selected_region_converts_relative_to_meld_region(self):
        config = default_config()
        config.game_region = ScreenRegion(left=100, top=200, width=1000, height=500)
        config.layout.meld_region = RelativeRegion(0.5, 0.4, 0.4, 0.4)
        window = SimpleNamespace(config=config)

        relative = MainWindow.to_relative_meld_region(
            window,
            ScreenRegion(left=700, top=425, width=100, height=50),
        )

        self.assertAlmostEqual(relative.x, 0.25)
        self.assertAlmostEqual(relative.y, 0.125)
        self.assertAlmostEqual(relative.width, 0.25)
        self.assertAlmostEqual(relative.height, 0.25)

    def test_selected_region_converts_relative_to_opponent_meld_region(self):
        config = default_config()
        config.game_region = ScreenRegion(left=0, top=0, width=1000, height=500)
        config.layout.opponent_melds = [
            PlayerMeldLayoutConfig(
                seat="across",
                region=RelativeRegion(0.2, 0.1, 0.4, 0.2),
            )
        ]
        window = SimpleNamespace(config=config)
        window._get_opponent_meld_layout = lambda seat: MainWindow._get_opponent_meld_layout(
            window, seat
        )

        relative = MainWindow.to_relative_meld_region(
            window,
            ScreenRegion(left=300, top=75, width=100, height=50),
            "across",
        )

        self.assertAlmostEqual(relative.x, 0.25)
        self.assertAlmostEqual(relative.y, 0.25)
        self.assertAlmostEqual(relative.width, 0.25)
        self.assertAlmostEqual(relative.height, 0.5)

    def test_format_analysis_explains_when_seven_pairs_is_the_best_shape(self):
        analysis = analyze_hand([
            "1m", "1m", "4s", "4s", "7s", "7s", "8s", "8s",
            "east", "east", "north", "white", "red", "9s",
        ])

        text = MainWindow.format_analysis(analysis)

        self.assertIn("一般形 2", text)
        self.assertIn("七对子 1", text)
        self.assertIn("当前最优形：七对子", text)

    def test_table_and_strategy_status_distinguish_missing_inputs(self):
        self.assertEqual(MainWindow._table_field_text("No score templates available.", True), "?（缺模板）")
        self.assertEqual(MainWindow._table_field_text(None, False), "?（未框选）")
        self.assertIn("未启用", MainWindow._strategy_text({"self": 25000}))
        self.assertIn(
            "速度优先",
            MainWindow._strategy_text({"self": 25000, "right": 30000, "across": 25000, "left": 25000}),
        )

    def test_format_shape_routes_labels_it_as_a_non_scoring_direction(self):
        report = match_shape_routes(ShapeRouteContext(
            concealed_tiles=(
                "1m", "1m", "4s", "4s", "7s", "7s", "8s", "8s",
                "east", "east", "north", "white", "red", "9s",
            ),
        ))

        text = MainWindow.format_shape_routes(report)

        self.assertIn("七对子: 1 向听", text)
        self.assertIn("非和牌、役成立或打点判定", text)
        self.assertIn("不会改变向听、有效牌或切牌排序", text)


if __name__ == "__main__":
    unittest.main()
