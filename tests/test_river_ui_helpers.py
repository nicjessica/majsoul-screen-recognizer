import unittest

from app.main_window import MainWindow, SEAT_LABELS
from recognizer.config import PlayerRiverLayoutConfig, RelativeRegion, default_config
from recognizer.geometry import ScreenRegion


class RiverUiHelperTests(unittest.TestCase):
    def test_parse_accepts_rotated_180_and_riichi_and_sorts_positions(self):
        slots = MainWindow._parse_river_slots(
            "1 0 upright normal\n0 1 rotated_180 riichi\n0 0 rotated_cw normal"
        )

        self.assertEqual([(slot.row, slot.column) for slot in slots], [(0, 0), (0, 1), (1, 0)])
        self.assertEqual(slots[1].orientation, "rotated_180")
        self.assertTrue(slots[1].is_riichi)

    def test_parse_rejects_duplicate_position(self):
        with self.assertRaises(ValueError):
            MainWindow._parse_river_slots(
                "0 0 upright normal\n0 0 rotated_180 riichi"
            )

    def test_parse_rejects_negative_row_or_column(self):
        for text in ("-1 0 upright normal", "0 -1 upright normal"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                MainWindow._parse_river_slots(text)

    def test_parse_rejects_invalid_state(self):
        with self.assertRaises(ValueError):
            MainWindow._parse_river_slots("0 0 upright discarded")

    def test_format_parse_round_trip(self):
        original = MainWindow._parse_river_slots(
            "2 1 rotated_ccw normal\n0 0 rotated_180 riichi\n1 3 upright normal"
        )

        formatted = MainWindow._format_river_slots(original)

        self.assertEqual(MainWindow._parse_river_slots(formatted), original)

    def test_selected_physical_region_converts_relative_to_requested_seat_river(self):
        window = MainWindow.__new__(MainWindow)
        window.config = default_config()
        window.config.game_region = ScreenRegion(left=100, top=200, width=1000, height=500)
        window.config.layout.rivers = [
            PlayerRiverLayoutConfig(
                "right", RelativeRegion(0.1, 0.2, 0.2, 0.4), tile_count=1
            ),
            PlayerRiverLayoutConfig(
                "left", RelativeRegion(0.6, 0.1, 0.3, 0.6), tile_count=1
            ),
        ]

        relative = window.to_relative_river_region(
            ScreenRegion(left=750, top=275, width=150, height=150), "left"
        )

        self.assertEqual(relative, RelativeRegion(1 / 6, 1 / 12, 0.5, 0.5))
        self.assertIsNone(
            window.to_relative_river_region(
                ScreenRegion(left=200, top=300, width=50, height=50), "left"
            )
        )

    def test_fixed_seat_labels(self):
        self.assertEqual(
            [SEAT_LABELS[seat] for seat in ("self", "right", "across", "left")],
            ["自己", "右家", "对家", "左家"],
        )


if __name__ == "__main__":
    unittest.main()
