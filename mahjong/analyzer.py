from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from mahjong.shanten import calculate_shanten
from mahjong.tiles import INDEX_TO_TILE, TILE_NAMES, normalize_tile, tiles_to_counts


@dataclass(frozen=True)
class DiscardRecommendation:
    discard: str
    resulting_shanten: int
    ukeire_count: int
    effective_tiles: list[str]
    reason: str


@dataclass(frozen=True)
class HandAnalysis:
    shanten: int
    recommendations: list[DiscardRecommendation]


def analyze_hand(
    tiles: list[str],
    open_meld_count: int = 0,
    visible_tiles: Iterable[str] = (),
) -> HandAnalysis:
    if not 0 <= open_meld_count <= 4:
        raise ValueError("副露组数必须在 0 到 4 之间")
    normalized = [normalize_tile(tile) for tile in tiles]
    counts = tiles_to_counts(normalized)
    visible_counts = tiles_to_counts([normalize_tile(tile) for tile in visible_tiles])
    _validate_known_tile_counts(counts, visible_counts)
    tile_count = sum(counts)
    expected_counts = (13 - 3 * open_meld_count, 14 - 3 * open_meld_count)
    if tile_count not in expected_counts:
        raise ValueError(
            f"{open_meld_count} 组副露需要 {expected_counts[0]} 或 {expected_counts[1]} 张暗牌，"
            f"当前 {tile_count} 张"
        )

    current_shanten = calculate_shanten(counts, open_meld_count)
    recommendations: list[DiscardRecommendation] = []

    if tile_count == expected_counts[0]:
        effective, count = effective_draws(
            counts, current_shanten, open_meld_count, visible_counts
        )
        recommendations.append(
            DiscardRecommendation(
                discard="-",
                resulting_shanten=current_shanten,
                ukeire_count=count,
                effective_tiles=effective,
                reason="当前为摸牌前状态，列出能降低向听的有效牌",
            )
        )
        return HandAnalysis(shanten=current_shanten, recommendations=recommendations)

    for discard in sorted(set(normalized), key=TILE_NAMES.index):
        discard_index = TILE_NAMES.index(discard)
        if counts[discard_index] == 0:
            continue
        after_discard = counts.copy()
        after_discard[discard_index] -= 1
        visible_after_discard = visible_counts.copy()
        visible_after_discard[discard_index] += 1
        resulting = calculate_shanten(after_discard, open_meld_count)
        effective, count = effective_draws(
            after_discard, resulting, open_meld_count, visible_after_discard
        )
        recommendations.append(
            DiscardRecommendation(
                discard=discard,
                resulting_shanten=resulting,
                ukeire_count=count,
                effective_tiles=effective,
                reason=_reason(current_shanten, resulting, count),
            )
        )

    recommendations.sort(
        key=lambda item: (item.resulting_shanten, -item.ukeire_count, TILE_NAMES.index(item.discard))
    )
    return HandAnalysis(shanten=current_shanten, recommendations=recommendations)


def effective_draws(
    counts: list[int],
    base_shanten: int,
    open_meld_count: int = 0,
    visible_counts: Sequence[int] | None = None,
) -> tuple[list[str], int]:
    external_counts = list(visible_counts) if visible_counts is not None else [0] * 34
    if len(external_counts) != 34:
        raise ValueError("可见牌计数必须包含 34 种牌")
    _validate_known_tile_counts(counts, external_counts)
    effective: list[str] = []
    total_remaining = 0

    for index in range(34):
        if counts[index] >= 4:
            continue
        test_counts = counts.copy()
        test_counts[index] += 1
        if calculate_shanten(test_counts, open_meld_count) < base_shanten:
            effective.append(INDEX_TO_TILE[index])
            total_remaining += 4 - counts[index] - external_counts[index]

    return effective, total_remaining


def _validate_known_tile_counts(counts: Sequence[int], visible_counts: Sequence[int]) -> None:
    for index, (concealed, visible) in enumerate(zip(counts, visible_counts, strict=True)):
        if concealed + visible > 4:
            raise ValueError(f"已知同一种牌超过 4 张: {INDEX_TO_TILE[index]}")


def _reason(current_shanten: int, resulting_shanten: int, ukeire_count: int) -> str:
    if resulting_shanten < current_shanten:
        return f"降低到 {resulting_shanten} 向听，有效牌 {ukeire_count} 枚"
    if resulting_shanten == current_shanten:
        return f"保持 {resulting_shanten} 向听，有效牌 {ukeire_count} 枚"
    return f"向听后退到 {resulting_shanten}，通常不优先"
