from __future__ import annotations


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


if __name__ == "__main__":
    enable_windows_dpi_awareness()
    from app.main_window import main

    main()
