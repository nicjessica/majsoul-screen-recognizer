import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import run_app


class RunAppConsoleTests(unittest.TestCase):
    def test_visible_console_is_minimized_and_can_be_restored(self):
        user32 = SimpleNamespace(IsIconic=Mock(return_value=False), ShowWindow=Mock())
        kernel32 = SimpleNamespace(GetConsoleWindow=Mock(return_value=123))
        windll = SimpleNamespace(user32=user32, kernel32=kernel32)

        with patch.object(run_app.ctypes, "windll", windll, create=True):
            hwnd = run_app.minimize_windows_console()
            run_app.restore_windows_console(hwnd)

        self.assertEqual(hwnd, 123)
        self.assertEqual(
            user32.ShowWindow.call_args_list,
            [call(123, 6), call(123, 9)],
        )

    def test_already_minimized_console_is_left_unchanged(self):
        user32 = SimpleNamespace(IsIconic=Mock(return_value=True), ShowWindow=Mock())
        kernel32 = SimpleNamespace(GetConsoleWindow=Mock(return_value=456))
        windll = SimpleNamespace(user32=user32, kernel32=kernel32)

        with patch.object(run_app.ctypes, "windll", windll, create=True):
            self.assertIsNone(run_app.minimize_windows_console())

        user32.ShowWindow.assert_not_called()


if __name__ == "__main__":
    unittest.main()
