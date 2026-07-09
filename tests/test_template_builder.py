import tempfile
import unittest
import importlib.util
from pathlib import Path

from recognizer.config import default_config


class TemplateBuilderTests(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("PIL") is None, "Pillow is not installed")
    def test_build_templates_from_screenshot(self):
        from PIL import Image

        from recognizer.template_builder import build_templates_from_screenshot

        with tempfile.TemporaryDirectory() as tmp:
            screenshot = Path(tmp) / "screen.png"
            Image.new("RGB", (1920, 1080), (255, 255, 255)).save(screenshot)
            config = default_config()
            names = [f"{index + 1}m" for index in range(9)] + [
                "east",
                "south",
                "west",
                "north",
                "white",
            ]

            saved = build_templates_from_screenshot(
                screenshot,
                names,
                Path(tmp) / "templates",
                config.layout,
            )

            self.assertEqual(len(saved), 14)
            self.assertTrue((Path(tmp) / "templates" / "1m.png").exists())
            self.assertTrue((Path(tmp) / "templates" / "white.png").exists())

    @unittest.skipIf(importlib.util.find_spec("PIL") is None, "Pillow is not installed")
    def test_build_templates_without_draw_tile(self):
        from PIL import Image

        from recognizer.template_builder import build_templates_from_screenshot

        with tempfile.TemporaryDirectory() as tmp:
            screenshot = Path(tmp) / "screen.png"
            Image.new("RGB", (1920, 1080), (255, 255, 255)).save(screenshot)
            config = default_config()
            config.layout.hand_tile_count = 10
            config.layout.draw_tile_count = 0
            names = [f"{index + 1}m" for index in range(9)] + ["east"]

            saved = build_templates_from_screenshot(
                screenshot,
                names,
                Path(tmp) / "templates",
                config.layout,
            )

            self.assertEqual(len(saved), 10)


if __name__ == "__main__":
    unittest.main()
