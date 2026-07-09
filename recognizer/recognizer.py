from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from recognizer.config import AppConfig, RelativeRegion
from recognizer.models import RecognitionResult, TileMatch
from recognizer.templates import TemplateLibrary


class RecognitionError(RuntimeError):
    pass


class TileRecognizer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        try:
            self.templates = TemplateLibrary(config.templates_dir)
        except RuntimeError as exc:
            raise RecognitionError(str(exc)) from exc

    def recognize(self, frame_rgb: np.ndarray) -> RecognitionResult:
        if not self.templates.templates:
            raise RecognitionError(
                f"没有牌模板。请把模板图片放入 {self.config.templates_dir}"
            )

        hand_tiles, draw_tiles, dora_tiles, meld_tiles = self.extract_tiles(frame_rgb)

        matches: list[TileMatch] = []
        hand: list[str] = []
        for tile in hand_tiles:
            match = self._match_checked(tile)
            matches.append(match)
            hand.append(match.name)

        draw: str | None = None
        if draw_tiles:
            draw_match = self._match_checked(draw_tiles[0])
            matches.append(draw_match)
            draw = draw_match.name

        dora: list[str] = []
        for tile in dora_tiles:
            match = self._match_checked(tile)
            matches.append(match)
            dora.append(match.name)

        meld: list[str] = []
        for tile in meld_tiles:
            match = self._match_checked(tile)
            matches.append(match)
            meld.append(match.name)

        confidence = sum(match.score for match in matches) / max(len(matches), 1)
        return RecognitionResult(
            hand=hand,
            draw=draw,
            dora_indicators=dora,
            meld_tiles=meld,
            confidence=confidence,
            matches=matches,
        )

    def extract_tiles(
        self, frame_rgb: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        layout = self.config.layout
        hand_tiles = split_region(
            crop_relative(frame_rgb, layout.hand_region),
            layout.hand_tile_count,
        )
        draw_tiles = split_region(
            crop_relative(frame_rgb, layout.draw_region),
            layout.draw_tile_count,
        )
        dora_tiles = split_region(
            crop_relative(frame_rgb, layout.dora_region),
            layout.dora_tile_count,
        )
        meld_tiles: list[np.ndarray] = []
        if layout.meld_region is not None and layout.meld_tile_count > 0:
            meld_tiles = split_region(
                crop_relative(frame_rgb, layout.meld_region),
                layout.meld_tile_count,
            )
        return hand_tiles, draw_tiles, dora_tiles, meld_tiles

    def save_debug_tiles(self, frame_rgb: np.ndarray, output_dir: str | Path = "data/debug/last_failed") -> Path:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        self.save_region_overlay(frame_rgb, output / "full_capture_regions.png")

        hand_tiles, draw_tiles, dora_tiles, meld_tiles = self.extract_tiles(frame_rgb)
        for index, tile in enumerate(hand_tiles, start=1):
            Image.fromarray(tile).save(output / f"hand_{index:02d}.png")
        for index, tile in enumerate(draw_tiles, start=1):
            Image.fromarray(tile).save(output / f"draw_{index:02d}.png")
        for index, tile in enumerate(dora_tiles, start=1):
            Image.fromarray(tile).save(output / f"dora_{index:02d}.png")
        for index, tile in enumerate(meld_tiles, start=1):
            Image.fromarray(tile).save(output / f"meld_{index:02d}.png")
        return output

    def save_region_overlay(self, frame_rgb: np.ndarray, output_path: str | Path) -> None:
        image = Image.fromarray(frame_rgb).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        regions = [
            ("hand", self.config.layout.hand_region, (255, 60, 60)),
            ("draw", self.config.layout.draw_region, (60, 180, 255)),
            ("dora", self.config.layout.dora_region, (70, 220, 100)),
        ]
        if self.config.layout.meld_region is not None and self.config.layout.meld_tile_count > 0:
            regions.append(("meld", self.config.layout.meld_region, (255, 190, 60)))

        for label, region, color in regions:
            left = round(region.x * width)
            top = round(region.y * height)
            right = round((region.x + region.width) * width)
            bottom = round((region.y + region.height) * height)
            draw.rectangle((left, top, right, bottom), outline=color, width=4)
            draw.text((left + 6, max(0, top - 18)), label, fill=color)

        image.save(output_path)

    def _match_checked(self, tile_rgb: np.ndarray) -> TileMatch:
        match = self.templates.match(tile_rgb)
        if match.score < self.config.recognition.threshold:
            raise RecognitionError(
                f"识别置信度过低: {match.name}={match.score:.3f}，"
                f"阈值 {self.config.recognition.threshold:.3f}。请检查模板或区域配置。"
            )
        return match


def crop_relative(frame: np.ndarray, region: RelativeRegion) -> np.ndarray:
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, round(region.x * width)))
    y1 = max(0, min(height - 1, round(region.y * height)))
    x2 = max(x1 + 1, min(width, round((region.x + region.width) * width)))
    y2 = max(y1 + 1, min(height, round((region.y + region.height) * height)))
    return frame[y1:y2, x1:x2]


def split_region(region: np.ndarray, count: int) -> list[np.ndarray]:
    if count <= 0:
        return []

    height, width = region.shape[:2]
    tile_width = width / count
    tiles = []
    for index in range(count):
        x1 = round(index * tile_width)
        x2 = round((index + 1) * tile_width)
        tile = region[:, x1:x2]
        if tile.size:
            tiles.append(tile)

    if len(tiles) != count:
        raise RecognitionError(f"牌区切分失败: expected={count}, actual={len(tiles)}")
    return tiles
