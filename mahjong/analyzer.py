from __future__ import annotations

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


def analyze_hand(tiles: list[str]) -> HandAnalysis:
    normalized = [normalize_tile(tile) for tile in tiles]
    counts = tiles_to_counts(normalized)
    tile_count = sum(counts)
    if tile_count not in (13, 14):
        raise ValueError(f"需要 13 或 14 张牌，当前 {tile_count} 张")

    current_shanten = calculate_shanten(counts)
    recommendations: list[DiscardRecommendation] = []

    if tile_count == 13:
        effective, count = effective_draws(counts, current_shanten)
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
        resulting = calculate_shanten(after_discard)
        effective, count = effective_draws(after_discard, resulting)
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


def effective_draws(counts: list[int], base_shanten: int) -> tuple[list[str], int]:
    effective: list[str] = []
    total_remaining = 0

    for index in range(34):
        if counts[index] >= 4:
            continue
        test_counts = counts.copy()
        test_counts[index] += 1
        if calculate_shanten(test_counts) < base_shanten:
            effective.append(INDEX_TO_TILE[index])
            total_remaining += 4 - counts[index]

    return effective, total_remaining


def _reason(current_shanten: int, resulting_shanten: int, ukeire_count: int) -> str:
    if resulting_shanten < current_shanten:
        return f"降低到 {resulting_shanten} 向听，有效牌 {ukeire_count} 枚"
    if resulting_shanten == current_shanten:
        return f"保持 {resulting_shanten} 向听，有效牌 {ukeire_count} 枚"
    return f"向听后退到 {resulting_shanten}，通常不优先"

