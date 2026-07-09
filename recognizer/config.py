from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from recognizer.geometry import ScreenRegion


CONFIG_PATH = Path("data/config.json")


@dataclass
class RelativeRegion:
    x: float
    y: float
    width: float
    height: float


@dataclass
class LayoutConfig:
    hand_region: RelativeRegion
    draw_region: RelativeRegion
    dora_region: RelativeRegion
    meld_region: RelativeRegion | None = None
    hand_tile_count: int = 13
    draw_tile_count: int = 1
    dora_tile_count: int = 1
    meld_tile_count: int = 0


@dataclass
class RecognitionConfig:
    threshold: float = 0.78


@dataclass
class AppConfig:
    game_region: ScreenRegion | None
    layout: LayoutConfig
    recognition: RecognitionConfig
    templates_dir: str = "data/templates"
    capture_interval_seconds: float = 0.75


def default_config() -> AppConfig:
    return AppConfig(
        game_region=None,
        layout=LayoutConfig(
            hand_region=RelativeRegion(x=0.115, y=0.852, width=0.650, height=0.126),
            draw_region=RelativeRegion(x=0.773, y=0.852, width=0.055, height=0.126),
            dora_region=RelativeRegion(x=0.506, y=0.088, width=0.052, height=0.064),
        ),
        recognition=RecognitionConfig(),
    )


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    if not path.exists():
        config = default_config()
        save_config(config, path)
        return config

    data = json.loads(path.read_text(encoding="utf-8"))
    return _config_from_dict(data)


def save_config(config: AppConfig, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(config.templates_dir).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_config_to_dict(config), indent=2, ensure_ascii=False), encoding="utf-8")


def _config_to_dict(config: AppConfig) -> dict[str, Any]:
    data = asdict(config)
    return data


def _config_from_dict(data: dict[str, Any]) -> AppConfig:
    region_data = data.get("game_region")
    region = ScreenRegion(**region_data) if region_data else None

    layout_data = data.get("layout", {})
    default = default_config()
    default_layout = default.layout
    layout = LayoutConfig(
        hand_region=RelativeRegion(**layout_data.get("hand_region", asdict(default_layout.hand_region))),
        draw_region=RelativeRegion(**layout_data.get("draw_region", asdict(default_layout.draw_region))),
        dora_region=RelativeRegion(**layout_data.get("dora_region", asdict(default_layout.dora_region))),
        meld_region=(
            RelativeRegion(**layout_data["meld_region"])
            if layout_data.get("meld_region")
            else default_layout.meld_region
        ),
        hand_tile_count=int(layout_data.get("hand_tile_count", default_layout.hand_tile_count)),
        draw_tile_count=int(layout_data.get("draw_tile_count", default_layout.draw_tile_count)),
        dora_tile_count=int(layout_data.get("dora_tile_count", default_layout.dora_tile_count)),
        meld_tile_count=int(layout_data.get("meld_tile_count", default_layout.meld_tile_count)),
    )

    recognition_data = data.get("recognition", {})
    recognition = RecognitionConfig(
        threshold=float(recognition_data.get("threshold", default.recognition.threshold))
    )

    return AppConfig(
        game_region=region,
        layout=layout,
        recognition=recognition,
        templates_dir=str(data.get("templates_dir", default.templates_dir)),
        capture_interval_seconds=float(
            data.get("capture_interval_seconds", default.capture_interval_seconds)
        ),
    )
