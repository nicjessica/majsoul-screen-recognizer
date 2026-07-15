import unittest
from types import SimpleNamespace

from app.main_window import MainWindow
from recognizer.config import RelativeRegion, default_config
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


if __name__ == "__main__":
    unittest.main()
