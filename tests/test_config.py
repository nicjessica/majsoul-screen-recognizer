import json
import tempfile
import unittest
from pathlib import Path

from recognizer.config import (
    MeldConfig,
    MeldTileSlotConfig,
    PlayerMeldLayoutConfig,
    RelativeRegion,
    default_config,
    load_config,
    save_config,
    validate_config,
)
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
            self.assertEqual(loaded.layout.open_meld_count, 0)
            self.assertEqual(loaded.layout.melds, [])

    def test_old_config_infers_open_meld_count_from_concealed_tiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            data = {
                "layout": {
                    "hand_region": {"x": 0, "y": 0, "width": 1, "height": 1},
                    "draw_region": {"x": 0, "y": 0, "width": 1, "height": 1},
                    "dora_region": {"x": 0, "y": 0, "width": 1, "height": 1},
                    "hand_tile_count": 10,
                    "draw_tile_count": 1,
                    "meld_tile_count": 4,
                }
            }
            path.write_text(json.dumps(data), encoding="utf-8")

            loaded = load_config(path)

            self.assertEqual(loaded.layout.open_meld_count, 1)
            self.assertEqual(loaded.layout.melds, [])

    def test_structured_meld_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            config = default_config()
            config.templates_dir = str(Path(tmp) / "templates")
            config.layout.melds = [
                MeldConfig(
                    kind="kakan",
                    tiles=[
                        MeldTileSlotConfig(
                            region=RelativeRegion(0.1, 0.2, 0.3, 0.4),
                            orientation="rotated_cw",
                            stack_level=1,
                        )
                    ],
                )
            ]

            save_config(config, path)
            loaded = load_config(path)

            self.assertEqual(loaded.layout.melds, config.layout.melds)

    def test_opponent_meld_round_trip_and_old_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            config = default_config()
            config.templates_dir = str(Path(tmp) / "templates")
            config.layout.opponent_melds = [PlayerMeldLayoutConfig(
                seat="across",
                region=RelativeRegion(0.2, 0.1, 0.3, 0.2),
                tile_count=3,
                melds=[MeldConfig("pon", [
                    MeldTileSlotConfig(RelativeRegion(0, 0, 0.3, 1), "rotated_180"),
                    MeldTileSlotConfig(RelativeRegion(0.3, 0, 0.3, 1), "rotated_180"),
                    MeldTileSlotConfig(RelativeRegion(0.6, 0, 0.3, 1), "rotated_180"),
                ])],
            )]
            save_config(config, path)
            self.assertEqual(load_config(path).layout.opponent_melds, config.layout.opponent_melds)

            path.write_text(json.dumps({"layout": {}}), encoding="utf-8")
            self.assertEqual(load_config(path).layout.opponent_melds, [])

    def test_validate_opponent_meld_seats_and_layout(self):
        config = default_config()
        config.layout.opponent_melds = [
            PlayerMeldLayoutConfig("self", tile_count=1),
            PlayerMeldLayoutConfig("right", tile_count=3),
            PlayerMeldLayoutConfig("right", region=RelativeRegion(0, 0, 1, 1), tile_count=-1),
        ]
        text = "\n".join(validate_config(config))
        self.assertIn("must be right/across/left", text)
        self.assertIn("is duplicated", text)
        self.assertIn("region is required", text)
        self.assertIn("cannot be negative", text)

    def test_unknown_meld_values_degrade_to_safe_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            data = {
                "layout": {
                    "melds": [
                        {
                            "kind": "future_kind",
                            "tiles": [
                                {
                                    "region": {"x": 0, "y": 0, "width": 1, "height": 1},
                                    "orientation": "diagonal",
                                }
                            ],
                        }
                    ]
                }
            }
            path.write_text(json.dumps(data), encoding="utf-8")

            loaded = load_config(path)

            self.assertEqual(loaded.layout.melds[0].kind, "unknown")
            self.assertEqual(loaded.layout.melds[0].tiles[0].orientation, "upright")

    def test_validate_config_accepts_old_unstructured_layout(self):
        config = default_config()
        config.layout.open_meld_count = 2
        config.layout.melds = []

        self.assertEqual(validate_config(config), [])

    def test_validate_config_accepts_valid_structured_meld(self):
        config = default_config()
        config.layout.meld_region = RelativeRegion(0.8, 0.8, 0.2, 0.2)
        config.layout.open_meld_count = 1
        config.layout.melds = [
            MeldConfig(
                kind="chi",
                tiles=[
                    MeldTileSlotConfig(RelativeRegion(0.0, 0.0, 0.2, 1.0)),
                    MeldTileSlotConfig(
                        RelativeRegion(0.3, 0.0, 0.3, 0.8),
                        orientation="rotated_cw",
                    ),
                    MeldTileSlotConfig(RelativeRegion(0.7, 0.0, 0.2, 1.0)),
                ],
            )
        ]

        self.assertEqual(validate_config(config), [])

    def test_validate_config_reports_structural_and_slot_errors(self):
        config = default_config()
        config.layout.open_meld_count = 5
        config.layout.melds = [
            MeldConfig(
                kind="pon",
                tiles=[
                    MeldTileSlotConfig(
                        RelativeRegion(0.9, -0.1, 0.2, 0),
                        orientation="diagonal",
                        stack_level=-1,
                    )
                ],
            ),
            MeldConfig(kind="unknown"),
        ]
        config.recognition.threshold = 1.1

        errors = validate_config(config)
        text = "\n".join(errors)

        self.assertIn("副露组数必须在 0 至 4", text)
        self.assertIn("置信度阈值必须在 0 至 1", text)
        self.assertIn("必须先框选副露区", text)
        self.assertIn("与副露组数 5 不一致", text)
        self.assertIn("pon 应配置 3 张", text)
        self.assertIn("方向无效", text)
        self.assertIn("叠放层级不能小于 0", text)
        self.assertIn("区域必须位于副露区", text)
        self.assertIn("unknown 副露至少需要 1 张", text)


if __name__ == "__main__":
    unittest.main()
