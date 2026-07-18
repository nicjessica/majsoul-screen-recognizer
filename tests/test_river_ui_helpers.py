import unittest
from unittest.mock import MagicMock, patch

from app.main_window import (
    MainWindow,
    RIVER_SEAT_LABELS,
    RIVER_SELECT_SPECS,
    SEAT_LABELS,
)
from recognizer.config import (
    PlayerRiverLayoutConfig,
    PlayerScoreLayoutConfig,
    RelativeRegion,
    default_config,
)
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

    def test_river_labels_map_turn_order_to_internal_screen_seats(self):
        self.assertEqual(
            RIVER_SEAT_LABELS,
            {"self": "我的", "left": "上家", "right": "下家", "across": "对家"},
        )

    def test_river_selection_exposes_four_direct_buttons(self):
        self.assertEqual(
            RIVER_SELECT_SPECS,
            (
                ("self", "框选我的牌河"),
                ("left", "框选上家牌河"),
                ("right", "框选下家牌河"),
                ("across", "框选对家牌河"),
            ),
        )

    def test_across_score_selection_replaces_old_region_and_saves_rotated_180(self):
        window = MainWindow.__new__(MainWindow)
        window.config = default_config()
        window.config.game_region = ScreenRegion(left=100, top=200, width=1000, height=500)
        old_region = RelativeRegion(0.1, 0.1, 0.1, 0.1)
        window.config.layout.table_state.scores = [
            PlayerScoreLayoutConfig("self", old_region),
            PlayerScoreLayoutConfig("across", old_region, "upright"),
        ]
        window.pending_region = "table:score:across"
        window.pending_score_orientation = "rotated_180"
        window.selector_completed = False
        window._invalidate_recognition_state = MagicMock()
        window.region_label = MagicMock()
        window.layout_label = MagicMock()
        window.status_label = MagicMock()
        window.region_text = MagicMock(return_value="region")
        window.layout_text = MagicMock(return_value="layout")

        with patch("app.main_window.save_config") as save:
            window.on_region_selected(
                ScreenRegion(left=600, top=300, width=200, height=100)
            )

        across = [
            item
            for item in window.config.layout.table_state.scores
            if item.seat == "across"
        ]
        self.assertEqual(len(across), 1)
        self.assertEqual(across[0].region, RelativeRegion(0.5, 0.2, 0.2, 0.2))
        self.assertEqual(across[0].orientation, "rotated_180")
        self.assertEqual(window.config.layout.table_state.scores[0].seat, "self")
        save.assert_called_once_with(window.config)
        window._invalidate_recognition_state.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
