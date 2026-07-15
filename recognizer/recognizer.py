from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from recognizer.config import AppConfig, MeldConfig, RelativeRegion, RiverTileSlotConfig
from recognizer.models import (
    MeldRecognition,
    MeldTileRecognition,
    PlayerMeldRecognition,
    ObservedTileRecognition,
    PlayerRiverRecognition,
    RecognitionResult,
    TileMatch,
)
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

        core_matches: list[TileMatch] = []
        hand: list[str] = []
        for index, tile in enumerate(hand_tiles, start=1):
            match = self._match_checked(tile, "手牌", index)
            core_matches.append(match)
            hand.append(match.name)

        draw: str | None = None
        if draw_tiles:
            draw_match = self._match_checked(draw_tiles[0], "摸牌", 1)
            core_matches.append(draw_match)
            draw = draw_match.name

        dora: list[str] = []
        for index, tile in enumerate(dora_tiles, start=1):
            match = self._match_checked(tile, "宝牌指示牌", index)
            core_matches.append(match)
            dora.append(match.name)

        meld_results, meld_matches = self._recognize_melds(meld_tiles)
        meld = [
            tile.name
            for group in meld_results
            for tile in group.tiles
            if tile.name is not None
        ]
        meld_errors = [group.error for group in meld_results if group.error]
        meld_error = "；".join(meld_errors) or None
        opponent_results: list[PlayerMeldRecognition] = []
        opponent_matches: list[TileMatch] = []
        opponent_tiles = self.extract_opponent_meld_tiles(frame_rgb)
        for player in self.config.layout.opponent_melds:
            groups, successful = self._recognize_melds(
                opponent_tiles.get(player.seat, []),
                configs=player.melds,
                area=player.seat,
            )
            names = [tile.name for group in groups for tile in group.tiles if tile.name is not None]
            errors = [group.error for group in groups if group.error]
            opponent_results.append(PlayerMeldRecognition(
                seat=player.seat,
                meld_tiles=names,
                melds=groups,
                error="; ".join(errors) or None,
            ))
            opponent_matches.extend(successful)
        river_results: list[PlayerRiverRecognition] = []
        river_matches: list[TileMatch] = []
        river_tiles = self.extract_river_tiles(frame_rgb)
        for river in self.config.layout.rivers:
            recognized: list[ObservedTileRecognition] = []
            errors: list[str] = []
            slots = river.tiles
            for index, tile in enumerate(river_tiles.get(river.seat, []), start=1):
                candidates = self.templates.match_candidates(tile, limit=2)
                match = candidates[0]
                is_riichi = slots[index - 1].is_riichi if slots else False
                row = slots[index - 1].row if slots else 0
                column = slots[index - 1].column if slots else index - 1
                if match.score < self.config.recognition.threshold:
                    error = (
                        f"river:{river.seat} tile {index} "
                        f"row={row} column={column} confidence too low: "
                        f"{_format_candidates(candidates)}; "
                        f"threshold={self.config.recognition.threshold:.3f}"
                    )
                    recognized.append(ObservedTileRecognition(
                        None, match, error, candidates, is_riichi, row, column
                    ))
                    errors.append(error)
                else:
                    recognized.append(ObservedTileRecognition(
                        match.name,
                        match,
                        candidates=candidates,
                        is_riichi=is_riichi,
                        row=row,
                        column=column,
                    ))
                    river_matches.append(match)
            river_results.append(PlayerRiverRecognition(
                river.seat, recognized, "; ".join(errors) or None
            ))
        matches = [*core_matches, *meld_matches, *opponent_matches, *river_matches]
        confidence = sum(match.score for match in core_matches) / max(len(core_matches), 1)
        return RecognitionResult(
            hand=hand,
            draw=draw,
            dora_indicators=dora,
            meld_tiles=meld,
            confidence=confidence,
            matches=matches,
            meld_error=meld_error,
            melds=meld_results,
            opponent_melds=opponent_results,
            rivers=river_results,
        )

    def _recognize_melds(
        self,
        meld_tiles: list[np.ndarray],
        configs: list[MeldConfig] | None = None,
        area: str = "self",
    ) -> tuple[list[MeldRecognition], list[TileMatch]]:
        configs = self.config.layout.melds if configs is None else configs
        if not configs and not meld_tiles:
            return [], []
        group_sizes = [len(meld.tiles) for meld in configs] if configs else [len(meld_tiles)]
        group_kinds = [meld.kind for meld in configs] if configs else ["unknown"]
        results: list[MeldRecognition] = []
        successful_matches: list[TileMatch] = []
        offset = 0
        for group_index, (kind, size) in enumerate(zip(group_kinds, group_sizes), start=1):
            tile_results: list[MeldTileRecognition] = []
            group_matches: list[TileMatch] = []
            errors: list[str] = []
            for tile_index, tile in enumerate(meld_tiles[offset : offset + size], start=1):
                candidates = self.templates.match_candidates(tile, limit=2)
                match = candidates[0]
                if match.score < self.config.recognition.threshold:
                    error = (
                        f"第 {group_index} 组第 {tile_index} 张副露牌置信度过低: "
                        f"候选 {_format_candidates(candidates)}，"
                        f"阈值 {self.config.recognition.threshold:.3f}"
                    )
                    error = f"{area} meld: {error}"
                    tile_results.append(
                        MeldTileRecognition(
                            name=None,
                            match=match,
                            error=error,
                            candidates=candidates,
                        )
                    )
                    errors.append(error)
                else:
                    tile_results.append(
                        MeldTileRecognition(
                            name=match.name,
                            match=match,
                            candidates=candidates,
                        )
                    )
                    group_matches.append(match)
                    successful_matches.append(match)
            offset += size
            group_confidence = (
                sum(match.score for match in group_matches) / len(group_matches)
                if group_matches
                else None
            )
            results.append(
                MeldRecognition(
                    kind=kind,
                    tiles=tile_results,
                    confidence=group_confidence,
                    error="；".join(errors) or None,
                )
            )
        return results, successful_matches

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
        if layout.meld_region is not None:
            meld_region = crop_relative(frame_rgb, layout.meld_region)
            if layout.melds:
                meld_tiles = [tile for group in crop_meld_slots(meld_region, layout.melds) for tile in group]
            elif layout.meld_tile_count > 0:
                meld_tiles = split_region(meld_region, layout.meld_tile_count)
        return hand_tiles, draw_tiles, dora_tiles, meld_tiles

    def extract_opponent_meld_tiles(self, frame_rgb: np.ndarray) -> dict[str, list[np.ndarray]]:
        extracted: dict[str, list[np.ndarray]] = {}
        for player in self.config.layout.opponent_melds:
            if player.region is None:
                extracted[player.seat] = []
                continue
            region = crop_relative(frame_rgb, player.region)
            if player.melds:
                extracted[player.seat] = [
                    tile for group in crop_meld_slots(region, player.melds) for tile in group
                ]
            else:
                extracted[player.seat] = split_region(region, player.tile_count)
        return extracted

    def extract_river_tiles(self, frame_rgb: np.ndarray) -> dict[str, list[np.ndarray]]:
        extracted: dict[str, list[np.ndarray]] = {}
        for river in self.config.layout.rivers:
            if river.region is None:
                extracted[river.seat] = []
                continue
            region = crop_relative(frame_rgb, river.region)
            if river.tiles:
                extracted[river.seat] = crop_river_slots(region, river.tiles)
            else:
                extracted[river.seat] = split_region(region, river.tile_count)
        return extracted

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
        if self.config.layout.melds:
            offset = 0
            for group_index, meld in enumerate(self.config.layout.melds, start=1):
                for tile_index in range(1, len(meld.tiles) + 1):
                    Image.fromarray(meld_tiles[offset]).save(
                        output / f"meld_{group_index:02d}_{tile_index:02d}.png"
                    )
                    offset += 1
        else:
            for index, tile in enumerate(meld_tiles, start=1):
                Image.fromarray(tile).save(output / f"meld_{index:02d}.png")
        opponent_tiles = self.extract_opponent_meld_tiles(frame_rgb)
        for player in self.config.layout.opponent_melds:
            tiles = opponent_tiles.get(player.seat, [])
            if player.melds:
                offset = 0
                for group_index, meld in enumerate(player.melds, start=1):
                    for tile_index in range(1, len(meld.tiles) + 1):
                        Image.fromarray(tiles[offset]).save(
                            output / f"meld_{player.seat}_{group_index:02d}_{tile_index:02d}.png"
                        )
                        offset += 1
            else:
                for index, tile in enumerate(tiles, start=1):
                    Image.fromarray(tile).save(output / f"meld_{player.seat}_{index:02d}.png")
        river_tiles = self.extract_river_tiles(frame_rgb)
        for river in self.config.layout.rivers:
            tiles = river_tiles.get(river.seat, [])
            for index, tile in enumerate(tiles, start=1):
                if river.tiles:
                    slot = river.tiles[index - 1]
                    name = f"river_{river.seat}_r{slot.row:02d}_c{slot.column:02d}_{index:02d}.png"
                else:
                    name = f"river_{river.seat}_{index:02d}.png"
                Image.fromarray(tile).save(output / name)
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
        if self.config.layout.meld_region is not None and (
            self.config.layout.meld_tile_count > 0 or self.config.layout.melds
        ):
            regions.append(("meld", self.config.layout.meld_region, (255, 190, 60)))
        for player in self.config.layout.opponent_melds:
            if player.region is not None and (player.tile_count > 0 or player.melds):
                regions.append((f"meld:{player.seat}", player.region, (210, 100, 255)))
        for river in self.config.layout.rivers:
            if river.region is not None and (river.tile_count > 0 or river.tiles):
                regions.append((f"river:{river.seat}", river.region, (80, 220, 220)))

        for label, region, color in regions:
            left = round(region.x * width)
            top = round(region.y * height)
            right = round((region.x + region.width) * width)
            bottom = round((region.y + region.height) * height)
            draw.rectangle((left, top, right, bottom), outline=color, width=4)
            draw.text((left + 6, max(0, top - 18)), label, fill=color)

        image.save(output_path)

    def _match_checked(self, tile_rgb: np.ndarray, area: str, index: int) -> TileMatch:
        candidates = self.templates.match_candidates(tile_rgb, limit=2)
        match = candidates[0]
        if match.score < self.config.recognition.threshold:
            raise RecognitionError(
                f"{area}第 {index} 张识别置信度过低: "
                f"候选 {_format_candidates(candidates)}，"
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


def crop_meld_slots(region: np.ndarray, melds: list[MeldConfig]) -> list[list[np.ndarray]]:
    groups: list[list[np.ndarray]] = []
    for meld in melds:
        tiles: list[np.ndarray] = []
        for slot in meld.tiles:
            tile = crop_relative(region, slot.region)
            if slot.orientation == "rotated_cw":
                tile = np.rot90(tile, 1)
            elif slot.orientation == "rotated_ccw":
                tile = np.rot90(tile, -1)
            elif slot.orientation == "rotated_180":
                tile = np.rot90(tile, 2)
            tiles.append(np.ascontiguousarray(tile))
        groups.append(tiles)
    return groups


def crop_river_slots(region: np.ndarray, slots: list[RiverTileSlotConfig]) -> list[np.ndarray]:
    tiles: list[np.ndarray] = []
    for slot in slots:
        tile = crop_relative(region, slot.region)
        if slot.orientation == "rotated_cw":
            tile = np.rot90(tile, 1)
        elif slot.orientation == "rotated_ccw":
            tile = np.rot90(tile, -1)
        elif slot.orientation == "rotated_180":
            tile = np.rot90(tile, 2)
        tiles.append(np.ascontiguousarray(tile))
    return tiles


def _format_candidates(candidates: list[TileMatch]) -> str:
    return "、".join(f"{match.name}={match.score:.3f}" for match in candidates)
