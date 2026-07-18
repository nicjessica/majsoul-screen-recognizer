from __future__ import annotations

import ctypes


def enable_windows_dpi_awareness() -> None:
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def minimize_windows_console() -> int | None:
    """Minimize a visible launcher console so it cannot contaminate screenshots."""
    try:
        hwnd = int(ctypes.windll.kernel32.GetConsoleWindow())
        if not hwnd or ctypes.windll.user32.IsIconic(hwnd):
            return None
        ctypes.windll.user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE
        return hwnd
    except Exception:
        return None


def restore_windows_console(hwnd: int | None) -> None:
    if hwnd is None:
        return
    try:
        ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    except Exception:
        pass


if __name__ == "__main__":
    enable_windows_dpi_awareness()
    console_window = minimize_windows_console()
    from app.main_window import main

    try:
        main()
    finally:
        restore_windows_console(console_window)
