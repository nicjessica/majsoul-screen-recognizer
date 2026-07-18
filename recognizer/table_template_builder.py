from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from recognizer.config import LayoutConfig, RelativeRegion
from recognizer.table_state import _normalize_orientation


def build_table_state_templates_from_screenshot(
    screenshot_path: str | Path,
    output_dir: str | Path,
    layout: LayoutConfig,
    *,
    round_value: tuple[str, int] | None = None,
    self_wind: str | None = None,
    scores: dict[str, int] | None = None,
) -> list[Path]:
    """Crop user-labelled table-state templates from one full game screenshot."""
    frame = np.asarray(Image.open(screenshot_path).convert("RGB"))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    table = layout.table_state

    if round_value is not None:
        if table.round_region is None:
            raise ValueError("尚未框选场风和局数区域")
        wind, hand_number = round_value
        saved.append(_save_crop(frame, table.round_region, destination / f"round_{wind}_{hand_number}.png"))

    if self_wind is not None:
        if table.self_wind_region is None:
            raise ValueError("尚未框选我的自风区域")
        saved.append(_save_crop(frame, table.self_wind_region, destination / f"wind_{self_wind}.png"))

    score_values = scores or {}
    score_layouts = {item.seat: item for item in table.scores}
    for seat, value in score_values.items():
        if seat not in score_layouts:
            raise ValueError(f"尚未框选 {seat} 点数区域")
        item = score_layouts[seat]
        crop = _crop(frame, item.region)
        crop = _normalize_orientation(crop, item.orientation)
        path = destination / f"score_{value}.png"
        Image.fromarray(crop).save(path)
        saved.append(path)
    return saved


def _save_crop(frame: np.ndarray, region: RelativeRegion, path: Path) -> Path:
    Image.fromarray(_crop(frame, region)).save(path)
    return path


def _crop(frame: np.ndarray, region: RelativeRegion) -> np.ndarray:
    height, width = frame.shape[:2]
    x1 = round(region.x * width)
    y1 = round(region.y * height)
    x2 = round((region.x + region.width) * width)
    y2 = round((region.y + region.height) * height)
    return frame[y1:y2, x1:x2]
