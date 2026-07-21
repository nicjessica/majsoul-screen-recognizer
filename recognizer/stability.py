from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from recognizer.config import LayoutConfig, RelativeRegion
from recognizer.models import RecognitionResult


# A full-region mean can conceal a meaningful change confined to a few of the
# 32x32 snapshot pixels.  Four-by-four blocks keep the check cheap while
# catching a tile face or score digit changing inside one small part of a
# region.  The block threshold is deliberately higher than ordinary antialias
# noise, and remains independent from the global MAD threshold.
_LOCAL_BLOCK_SIZE = 4
_LOCAL_BLOCK_MAD_THRESHOLD = 12.0


@dataclass(frozen=True)
class KeyRegionSnapshot:
    regions: dict[str, np.ndarray]

    @classmethod
    def from_frame(
        cls,
        frame: np.ndarray,
        layout: LayoutConfig,
        *,
        auto_detect_tile_state: bool = False,
    ) -> KeyRegionSnapshot:
        active_regions: list[tuple[str, RelativeRegion]] = []
        if auto_detect_tile_state or layout.hand_tile_count > 0:
            active_regions.append(("hand", layout.hand_region))
        if auto_detect_tile_state or layout.draw_tile_count > 0:
            active_regions.append(("draw", layout.draw_region))
        if layout.dora_tile_count > 0:
            active_regions.append(("dora", layout.dora_region))
        if layout.meld_region is not None and (
            auto_detect_tile_state or layout.meld_tile_count > 0 or layout.melds
        ):
            active_regions.append(("meld", layout.meld_region))
        for player in layout.opponent_melds:
            if player.region is not None and (player.tile_count > 0 or player.melds):
                active_regions.append((f"meld:{player.seat}", player.region))
        for river in layout.rivers:
            if river.region is not None and (river.tile_count > 0 or river.tiles):
                active_regions.append((f"river:{river.seat}", river.region))
        table_state = layout.table_state
        if table_state.round_region is not None:
            active_regions.append(("table:round", table_state.round_region))
        if table_state.self_wind_region is not None:
            active_regions.append(("table:self_wind", table_state.self_wind_region))
        for score in table_state.scores:
            active_regions.append((f"table:score:{score.seat}", score.region))

        return cls(
            regions={
                name: _normalized_grayscale(_crop_relative(frame, region))
                for name, region in active_regions
            }
        )

    def is_equivalent(self, other: KeyRegionSnapshot, max_mad: float = 2.0) -> bool:
        if self.regions.keys() != other.regions.keys():
            return False
        return all(
            _regions_are_equivalent(self.regions[name], other.regions[name], max_mad)
            for name in self.regions
        )


def result_key(result: RecognitionResult) -> tuple[object, ...]:
    if result.melds:
        meld_key = tuple(
            (meld.kind, tuple(tile.name for tile in meld.tiles))
            for meld in result.melds
        )
    else:
        meld_key = (("legacy", tuple(result.meld_tiles)),)
    opponents_by_seat = {player.seat: player for player in result.opponent_melds}
    opponent_key = []
    for seat in ("right", "across", "left"):
        player = opponents_by_seat.get(seat)
        if player is None or (not player.melds and not player.meld_tiles):
            opponent_key.append((seat, ()))
        elif player.melds:
            opponent_key.append(
                (seat, tuple((meld.kind, tuple(tile.name for tile in meld.tiles)) for meld in player.melds))
            )
        else:
            opponent_key.append((seat, (("legacy", tuple(player.meld_tiles)),)))
    rivers_by_seat = {river.seat: river for river in result.rivers}
    river_key = tuple(
        (
            seat,
            tuple(
                (tile.name, tile.is_riichi, tile.row, tile.column)
                for tile in rivers_by_seat[seat].tiles
            )
            if seat in rivers_by_seat
            else (),
        )
        for seat in ("self", "right", "across", "left")
    )
    scores_by_seat = {item.seat: item.score for item in result.table_state.scores}
    table_key = (
        result.table_state.round.round_wind,
        result.table_state.round.hand_number,
        result.table_state.self_wind.wind,
        tuple((seat, scores_by_seat.get(seat)) for seat in ("self", "right", "across", "left")),
    )
    return (
        tuple(result.hand),
        result.draw,
        result.open_meld_count,
        tuple(result.dora_indicators),
        meld_key,
        table_key,
        river_key,
        tuple(opponent_key),
    )


@dataclass(frozen=True)
class StabilityUpdate:
    published_result: RecognitionResult | None
    pending_count: int
    just_published: bool = False
    reused: bool = False


class RecognitionStabilizer:
    def __init__(
        self,
        required_observations: int = 3,
        max_reuses_before_recognition: int = 4,
    ) -> None:
        if required_observations <= 0:
            raise ValueError("required_observations 必须大于 0")
        if max_reuses_before_recognition < 0:
            raise ValueError("最大复用次数不能小于 0")
        self.required_observations = required_observations
        self.max_reuses_before_recognition = max_reuses_before_recognition
        self.reset()

    def needs_recognition(self, snapshot: KeyRegionSnapshot) -> bool:
        return (
            self._retry_required
            or self.reuse_count >= self.max_reuses_before_recognition
            or self.last_success_snapshot is None
            or not self.last_success_snapshot.is_equivalent(snapshot)
        )

    def observe_success(
        self,
        snapshot: KeyRegionSnapshot,
        result: RecognitionResult,
    ) -> StabilityUpdate:
        self.last_success_snapshot = snapshot
        self.raw_result = result
        self._retry_required = False
        self.reuse_count = 0
        return self._observe(result, reused=False)

    def observe_reused(self) -> StabilityUpdate:
        if self.raw_result is None:
            raise RuntimeError("尚无成功识别结果，不能复用")
        self.reuse_count += 1
        return self._observe(self.raw_result, reused=True)

    def publish_success(
        self,
        snapshot: KeyRegionSnapshot,
        result: RecognitionResult,
    ) -> StabilityUpdate:
        """立即发布一次手动识别，并将其设为后续连续识别的稳定基线。"""
        self.last_success_snapshot = snapshot
        self.raw_result = result
        self.pending_key = result_key(result)
        self.pending_count = self.required_observations
        self.pending_result = result
        self.published_result = result
        self._retry_required = False
        self.reuse_count = 0
        return self._update(just_published=True, reused=False)

    def observe_error(self) -> StabilityUpdate:
        self._retry_required = True
        return self._update(just_published=False, reused=False)

    def reset(self) -> None:
        self.last_success_snapshot: KeyRegionSnapshot | None = None
        self.raw_result: RecognitionResult | None = None
        self.pending_key: tuple[object, ...] | None = None
        self.pending_count = 0
        self.pending_result: RecognitionResult | None = None
        self.published_result: RecognitionResult | None = None
        self._retry_required = False
        self.reuse_count = 0

    def _observe(self, result: RecognitionResult, reused: bool) -> StabilityUpdate:
        key = result_key(result)
        if key == self.pending_key:
            self.pending_count += 1
        else:
            self.pending_key = key
            self.pending_count = 1
            self.pending_result = result

        just_published = False
        if self.pending_count >= self.required_observations:
            published_key = (
                result_key(self.published_result)
                if self.published_result is not None
                else None
            )
            if published_key != key:
                self.published_result = self.pending_result
                just_published = True
        return self._update(just_published=just_published, reused=reused)

    def _update(self, just_published: bool, reused: bool) -> StabilityUpdate:
        return StabilityUpdate(
            published_result=self.published_result,
            pending_count=self.pending_count,
            just_published=just_published,
            reused=reused,
        )


def _crop_relative(frame: np.ndarray, region: RelativeRegion) -> np.ndarray:
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, round(region.x * width)))
    y1 = max(0, min(height - 1, round(region.y * height)))
    x2 = max(x1 + 1, min(width, round((region.x + region.width) * width)))
    y2 = max(y1 + 1, min(height, round((region.y + region.height) * height)))
    return frame[y1:y2, x1:x2]


def _normalized_grayscale(region: np.ndarray) -> np.ndarray:
    image = Image.fromarray(region.astype(np.uint8)).convert("L")
    return np.asarray(image.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.uint8)


def _regions_are_equivalent(
    first: np.ndarray,
    second: np.ndarray,
    max_mad: float,
) -> bool:
    difference = np.abs(first.astype(np.float32) - second.astype(np.float32))
    if float(np.mean(difference)) > max_mad:
        return False
    return _max_local_mad(difference) <= _LOCAL_BLOCK_MAD_THRESHOLD


def _max_local_mad(difference: np.ndarray) -> float:
    """Return the highest average difference in a small local snapshot block."""
    height, width = difference.shape
    block = _LOCAL_BLOCK_SIZE
    maxima = 0.0
    for top in range(0, height, block):
        for left in range(0, width, block):
            maxima = max(maxima, float(np.mean(difference[top : top + block, left : left + block])))
    return maxima
