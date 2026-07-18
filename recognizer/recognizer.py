from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class _DetectedTileState:
    hand_matches: list[TileMatch]
    draw_match: TileMatch | None
    open_meld_count: int


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

        self._auto_state_error: str | None = None
        self._auto_state_geometry_valid = False
        detected_state = self._try_detect_tile_state(frame_rgb)
        if detected_state is None:
            if self._auto_state_error is not None and self._auto_state_geometry_valid:
                raise RecognitionError(f"自动牌数识别不确定：{self._auto_state_error}")
            try:
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
            except RecognitionError as exc:
                if self._auto_state_error is not None:
                    raise RecognitionError(
                        f"自动牌数识别不确定（{self._auto_state_error}）；"
                        f"手动牌数回退也失败：{exc}"
                    ) from exc
                raise
            open_meld_count = self.config.layout.open_meld_count
        else:
            core_matches = list(detected_state.hand_matches)
            hand = [match.name for match in detected_state.hand_matches]
            draw = detected_state.draw_match.name if detected_state.draw_match else None
            if detected_state.draw_match is not None:
                core_matches.append(detected_state.draw_match)
            open_meld_count = detected_state.open_meld_count

        dora: list[str] = []
        for index, tile in enumerate(dora_tiles, start=1):
            match = self._match_checked(tile, "宝牌指示牌", index)
            core_matches.append(match)
            dora.append(match.name)

        if detected_state is not None and detected_state.open_meld_count == 0:
            # Ignore stale configured meld slots only after this frame was
            # positively detected as a closed hand.
            meld_results, meld_matches = [], []
        else:
            meld_results, meld_matches = self._recognize_melds(
                meld_tiles,
                expected_group_count=open_meld_count,
            )
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
            open_meld_count=open_meld_count,
        )

    def _try_detect_tile_state(self, frame_rgb: np.ndarray) -> _DetectedTileState | None:
        """Detect the left-aligned concealed-tile state, or defer to manual config.

        Automatic detection deliberately has an all-or-nothing contract.  It only
        accepts a maximum-width 13-slot hand region with a legal occupied prefix;
        ambiguous scores and legacy regions framed for the current tile count use
        the existing manual counts instead.
        """
        recognition = self.config.recognition
        self._auto_state_geometry_valid = False
        if not recognition.auto_detect_tile_state:
            return None

        hand_region = crop_relative(frame_rgb, self.config.layout.hand_region)
        draw_region = crop_relative(frame_rgb, self.config.layout.draw_region)
        if not self._has_auto_tile_geometry(hand_region, draw_region):
            return self._defer_auto_state("手牌区不是与摸牌区等宽的 13 槽区域")
        self._auto_state_geometry_valid = True

        hand_tiles = split_region(hand_region, 13)
        occupancy = [_tile_occupancy_features(tile) for tile in hand_tiles]
        occupancy_states = [_classify_occupancy(*features) for features in occupancy]
        if any(state is None for state in occupancy_states):
            return self._defer_auto_state("手牌槽占位介于有牌与空槽阈值之间")

        draw_occupied = _classify_occupancy(*_tile_occupancy_features(draw_region))
        if draw_occupied is None:
            return self._defer_auto_state("摸牌区占位介于有牌与空槽阈值之间")

        detected_count: int | None = None
        continuous_extra_tile = False
        separated_draw_index: int | None = None
        for base_count in (13, 10, 7, 4, 1):
            if not all(occupancy_states[:base_count]):
                continue
            tail = occupancy_states[base_count:]
            if base_count == 13:
                detected_count = base_count
                break
            if not any(tail):
                detected_count = base_count
                break
            if tail[0] and not any(tail[1:]):
                # After chi/pon, 11/8/5/2 tiles form one continuous hand row.
                # RecognitionResult.draw is reserved for visually separated
                # draw tiles, so the extra tile remains part of ``hand``.
                detected_count = base_count
                continuous_extra_tile = True
                break
            if not any(tail[:-1]) and tail[-1]:
                detected_count = base_count
                separated_draw_index = 12
                break

        if detected_count is None:
            return self._defer_auto_state("手牌有牌槽不符合 13/10/7/4/1 张及末槽摸牌布局")
        if (continuous_extra_tile or separated_draw_index is not None) and draw_occupied:
            return self._defer_auto_state("手牌区待切牌与独立摸牌区同时有牌（同時有牌）")

        recognized_hand_count = detected_count + (1 if continuous_extra_tile else 0)
        hand_matches = [
            self._match_checked(tile, "手牌", index)
            for index, tile in enumerate(hand_tiles[:recognized_hand_count], start=1)
        ]
        if separated_draw_index is not None:
            detected_draw = self._match_checked(hand_tiles[separated_draw_index], "摸牌", 1)
        else:
            detected_draw = self._match_checked(draw_region, "摸牌", 1) if draw_occupied else None

        return _DetectedTileState(
            hand_matches=hand_matches,
            draw_match=detected_draw,
            open_meld_count=(13 - detected_count) // 3,
        )

    def _defer_auto_state(self, reason: str) -> None:
        self._auto_state_error = reason
        return None

    @staticmethod
    def _has_auto_tile_geometry(hand_region: np.ndarray, draw_region: np.ndarray) -> bool:
        """Reject legacy hand regions framed for fewer than thirteen tiles."""
        hand_height, hand_width = hand_region.shape[:2]
        draw_height, draw_width = draw_region.shape[:2]
        slot_width = hand_width / 13
        if min(hand_height, draw_height, slot_width, draw_width) < 8:
            return False
        width_ratio = slot_width / draw_width
        height_ratio = hand_height / draw_height
        return 0.82 <= width_ratio <= 1.22 and 0.80 <= height_ratio <= 1.25

    def _recognize_melds(
        self,
        meld_tiles: list[np.ndarray],
        configs: list[MeldConfig] | None = None,
        area: str = "self",
        expected_group_count: int | None = None,
    ) -> tuple[list[MeldRecognition], list[TileMatch]]:
        configs = self.config.layout.melds if configs is None else configs
        if not configs and not meld_tiles:
            return [], []
        if configs:
            group_sizes = [len(meld.tiles) for meld in configs]
        elif expected_group_count and len(meld_tiles) == expected_group_count * 3:
            group_sizes = [3] * expected_group_count
        elif expected_group_count and len(meld_tiles) == expected_group_count * 4:
            group_sizes = [4] * expected_group_count
        else:
            group_sizes = [len(meld_tiles)]
        group_kinds = [meld.kind for meld in configs] if configs else ["unknown"]
        if len(group_kinds) != len(group_sizes):
            group_kinds = ["unknown"] * len(group_sizes)
        results: list[MeldRecognition] = []
        successful_matches: list[TileMatch] = []
        offset = 0
        for group_index, (kind, size) in enumerate(zip(group_kinds, group_sizes), start=1):
            tile_results: list[MeldTileRecognition] = []
            group_matches: list[TileMatch] = []
            errors: list[str] = []
            for tile_index, tile in enumerate(meld_tiles[offset : offset + size], start=1):
                if _is_tile_back(tile):
                    # Concealed-kan backs are intentionally not sent through
                    # the front-face template library and never enter visible
                    # tile accounting.
                    tile_results.append(MeldTileRecognition(name=None, match=None))
                    continue
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
            config = configs[group_index - 1] if configs else None
            inferred_kind = infer_meld_kind(tile_results, config)
            resolved_kind = inferred_kind if inferred_kind != "unknown" else kind
            results.append(
                MeldRecognition(
                    kind=resolved_kind,
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


def _tile_occupancy_features(tile_rgb: np.ndarray) -> tuple[float, float]:
    """Return light-face and glyph fractions for a fixed hand slot.

    Mahjong Soul keeps bright white placeholders in vacated open-hand slots.
    They look even more like a tile face than a real tile, so lightness alone
    cannot distinguish them.  A real tile also has a sufficiently large dark or
    saturated glyph/pip region inside the face.
    """
    height, width = tile_rgb.shape[:2]
    y1, y2 = round(height * 0.08), round(height * 0.92)
    x1, x2 = round(width * 0.08), round(width * 0.92)
    core = tile_rgb[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)].astype(np.int16)
    channel_min = core.min(axis=2)
    channel_max = core.max(axis=2)
    tile_face = (channel_min >= 165) & ((channel_max - channel_min) <= 55)
    glyph = (channel_min <= 145) | ((channel_max - channel_min) >= 65)
    return float(np.mean(tile_face)), float(np.mean(glyph))


def _tile_occupancy_score(tile_rgb: np.ndarray) -> float:
    """Compatibility helper retained for diagnostics and older tests."""
    return _tile_occupancy_features(tile_rgb)[0]


def _classify_occupancy(face_score: float, glyph_score: float = 1.0) -> bool | None:
    if face_score >= 0.25 and glyph_score >= 0.035:
        return True
    if face_score <= 0.12 or (face_score >= 0.55 and glyph_score <= 0.018):
        return False
    return None


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


def infer_meld_kind(
    tiles: list[MeldTileRecognition],
    config: MeldConfig | None = None,
) -> str:
    """Infer a configured meld's semantic kind without guessing ambiguous kans."""
    names = [_normalize_red_five(tile.name) for tile in tiles if tile.name is not None]
    hidden_count = sum(tile.name is None and tile.error is None for tile in tiles)
    if len(tiles) == 3 and len(names) == 3:
        if len(set(names)) == 1:
            return "pon"
        if _is_chi_sequence(names):
            return "chi"
        return "unknown"

    if len(tiles) != 4:
        return "unknown"
    if hidden_count >= 2:
        # A group with at least two positively detected backs is necessarily a
        # concealed kan; the hidden identities remain unknown by design.
        return "ankan"
    if len(names) != 4 or len(set(names)) != 1:
        return "unknown"
    if config is not None and _has_stacked_kan_geometry(config):
        return "kakan"
    return "minkan"


def _normalize_red_five(name: str) -> str:
    return {"5mr": "5m", "5pr": "5p", "5sr": "5s"}.get(name, name)


def _is_chi_sequence(names: list[str]) -> bool:
    if len(names) != 3 or any(len(name) != 2 for name in names):
        return False
    suits = {name[1] for name in names}
    if len(suits) != 1 or next(iter(suits)) not in {"m", "p", "s"}:
        return False
    try:
        ranks = sorted(int(name[0]) for name in names)
    except ValueError:
        return False
    return ranks[1] == ranks[0] + 1 and ranks[2] == ranks[1] + 1


def _has_stacked_kan_geometry(config: MeldConfig) -> bool:
    if any(slot.stack_level > 0 for slot in config.tiles):
        return True
    for index, first in enumerate(config.tiles):
        for second in config.tiles[index + 1 :]:
            left = max(first.region.x, second.region.x)
            top = max(first.region.y, second.region.y)
            right = min(
                first.region.x + first.region.width,
                second.region.x + second.region.width,
            )
            bottom = min(
                first.region.y + first.region.height,
                second.region.y + second.region.height,
            )
            overlap = max(0.0, right - left) * max(0.0, bottom - top)
            smaller = min(
                first.region.width * first.region.height,
                second.region.width * second.region.height,
            )
            if smaller > 0 and overlap / smaller >= 0.35:
                return True
    return False


def _is_tile_back(tile_rgb: np.ndarray) -> bool:
    """Conservatively detect Mahjong Soul's saturated green tile back."""
    if tile_rgb.size == 0:
        return False
    pixels = tile_rgb.reshape(-1, 3).astype(np.int16)
    red, green, blue = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    green_pixels = (
        (green >= 55)
        & (green - red >= 18)
        & (green - blue >= 8)
        & ((pixels.max(axis=1) - pixels.min(axis=1)) >= 28)
    )
    light_face, _ = _tile_occupancy_features(tile_rgb)
    return float(np.mean(green_pixels)) >= 0.38 and light_face <= 0.18
