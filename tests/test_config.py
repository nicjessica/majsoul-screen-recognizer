import tempfile
import unittest
from pathlib import Path

from recognizer.config import default_config, load_config, save_config
from recognizer.geometry import ScreenRegion


class ConfigTests(unittest.TestCase):
    def test_round_trip_config_without_ui_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            config = default_config()
            config.game_region = ScreenRegion(left=1, top=2, width=3, height=4)
            config.templates_dir = str(Path(tmp) / "templates")
            save_config(config, path)

            loaded = load_config(path)
            self.assertEqual(loaded.game_region, config.game_region)
            self.assertEqual(loaded.layout.hand_tile_count, 13)
            self.assertEqual(loaded.layout.draw_tile_count, 1)


if __name__ == "__main__":
    unittest.main()
