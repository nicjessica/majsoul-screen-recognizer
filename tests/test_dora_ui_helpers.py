import unittest
from unittest.mock import MagicMock, patch

from app.main_window import MainWindow
from recognizer.config import default_config


def _window(initial_count=3):
    window = MainWindow.__new__(MainWindow)
    window.config = default_config()
    window.config.layout.dora_tile_count = initial_count
    window.layout_label = MagicMock()
    window.status_label = MagicMock()
    window._invalidate_recognition_state = MagicMock()
    return window


class DoraUiHelperTests(unittest.TestCase):
    def test_boundary_counts_one_and_five_save_invalidate_and_update_text(self):
        for value in (1, 5):
            with self.subTest(value=value):
                window = _window(initial_count=3)
                with patch("app.main_window.save_config") as save:
                    window._apply_dora_tile_count(value)

                self.assertEqual(window.config.layout.dora_tile_count, value)
                save.assert_called_once_with(window.config)
                window._invalidate_recognition_state.assert_called_once_with()
                window.layout_label.setText.assert_called_once_with(window.layout_text())
                status_text = window.status_label.setText.call_args.args[0]
                self.assertIn("宝牌指示牌张数", status_text)
                self.assertIn(str(value), status_text)

    def test_out_of_range_counts_do_not_change_save_invalidate_or_update_text(self):
        for value in (0, 6):
            with self.subTest(value=value):
                window = _window(initial_count=3)
                with patch("app.main_window.save_config") as save:
                    with self.assertRaises(ValueError):
                        window._apply_dora_tile_count(value)

                self.assertEqual(window.config.layout.dora_tile_count, 3)
                save.assert_not_called()
                window._invalidate_recognition_state.assert_not_called()
                window.layout_label.setText.assert_not_called()
                window.status_label.setText.assert_not_called()


if __name__ == "__main__":
    unittest.main()
