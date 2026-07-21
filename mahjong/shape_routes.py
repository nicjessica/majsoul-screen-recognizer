"""Conservative, current-hand routes toward a small set of riichi yaku.

This module deliberately answers a narrower question than a yaku evaluator:
given only reliable tiles in the current frame, which *routes* are compatible
with the hand?  A route is never a declaration that the hand can win, has a
yaku, or has a score.  In particular, static constraints such as honitsu are
reported as cards that conflict with the constraint, not as ``yaku shanten``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from mahjong.shanten import ShantenBreakdown, calculate_shanten_breakdown
from mahjong.tiles import HONORS, TILE_NAMES, normalize_tile, tiles_to_counts


RouteStatus = Literal["candidate", "blocked", "insufficient_data"]
RouteMetricKind = Literal["shanten", "discard_conflicts", "tile_copies"]

_WINDS = {"east", "south", "west", "north"}
_DRAGONS = ("white", "green", "red")
_SUITS = ("m", "p", "s")
_SUIT_LABELS = {"m": "万", "p": "筒", "s": "索"}
_GENERAL_WARNING = "路线只基于当前可靠牌面，不判定和牌、役成立、番符或点数"


@dataclass(frozen=True)
class KnownMeld:
    """A self meld whose kind and every visible tile are reliable."""

    kind: str
    tiles: tuple[str, ...]
    is_open: bool = True


@dataclass(frozen=True)
class ShapeRouteContext:
    concealed_tiles: tuple[str, ...]
    open_meld_count: int = 0
    melds: tuple[KnownMeld, ...] = ()
    unknown_meld_count: int = 0
    seat_wind: str | None = None
    round_wind: str | None = None
    open_tanyao: bool | None = None


@dataclass(frozen=True)
class ShapeRouteCandidate:
    id: str
    name: str
    status: RouteStatus
    metric_kind: RouteMetricKind
    metric: int | None
    evidence: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShapeRouteReport:
    base_shapes: ShantenBreakdown
    candidates: tuple[ShapeRouteCandidate, ...]
    primary_candidate: ShapeRouteCandidate | None
    warnings: tuple[str, ...] = ()


def match_shape_routes(context: ShapeRouteContext) -> ShapeRouteReport:
    """Return deterministic, evidence-backed route candidates.

    ``open_meld_count`` is the shanten-layer count of all fixed melds.  A
    caller may supply only the reliable subset in ``melds``; any unrepresented
    group is conservatively treated as unknown, so it can never support a
    whole-hand claim such as tanyao or honitsu.
    """

    normalized_concealed, normalized_melds, effective_unknown = _validate_context(context)
    all_known = [
        *normalized_concealed,
        *(tile for meld in normalized_melds for tile in meld.tiles),
    ]
    breakdown = calculate_shanten_breakdown(
        tiles_to_counts(normalized_concealed), context.open_meld_count
    )

    candidates: list[ShapeRouteCandidate] = [
        _normal_route(breakdown),
        _special_shape_route(
            "chiitoitsu",
            "七对子",
            breakdown.chiitoitsu,
            context.open_meld_count,
            "七对子不能有任何已完成副露",
        ),
        _special_shape_route(
            "kokushi_musou",
            "国士无双",
            breakdown.kokushi,
            context.open_meld_count,
            "国士无双不能有任何已完成副露",
        ),
        _tanyao_route(context, normalized_melds, all_known, effective_unknown),
        _best_suit_route("honitsu", "混一色", all_known, effective_unknown),
        _best_suit_route("chinitsu", "清一色", all_known, effective_unknown),
        _honroutou_route(all_known, effective_unknown),
        *_yakuhai_routes(
            all_known,
            effective_unknown,
            context.seat_wind,
            context.round_wind,
        ),
    ]
    primary = min(
        (item for item in candidates if item.status == "candidate" and item.metric_kind == "shanten"),
        key=lambda item: (item.metric if item.metric is not None else 99, item.id),
        default=None,
    )
    warnings = [_GENERAL_WARNING]
    if effective_unknown:
        warnings.append(f"有 {effective_unknown} 组副露未可靠识别，所有全手约束路线已降级")
    return ShapeRouteReport(breakdown, tuple(candidates), primary, tuple(warnings))


def _validate_context(
    context: ShapeRouteContext,
) -> tuple[list[str], tuple[KnownMeld, ...], int]:
    if not 0 <= context.open_meld_count <= 4:
        raise ValueError("副露组数必须在 0 到 4 之间")
    if context.unknown_meld_count < 0:
        raise ValueError("未知副露组数不能为负数")
    if len(context.melds) + context.unknown_meld_count > context.open_meld_count:
        raise ValueError("可靠副露与未知副露组数不能超过总副露组数")

    normalized_concealed = _normalize_tiles(context.concealed_tiles)
    expected_counts = (13 - 3 * context.open_meld_count, 14 - 3 * context.open_meld_count)
    if len(normalized_concealed) not in expected_counts:
        raise ValueError(
            f"{context.open_meld_count} 组副露需要 {expected_counts[0]} 或 {expected_counts[1]} 张暗牌，"
            f"当前 {len(normalized_concealed)} 张"
        )

    normalized_melds: list[KnownMeld] = []
    for meld in context.melds:
        if not isinstance(meld.is_open, bool):
            raise ValueError("副露开放状态必须是布尔值")
        normalized_melds.append(
            KnownMeld(meld.kind, tuple(_normalize_tiles(meld.tiles)), meld.is_open)
        )
    # This validates both names and the four-copy limit across reliable tiles.
    tiles_to_counts([
        *normalized_concealed,
        *(tile for meld in normalized_melds for tile in meld.tiles),
    ])
    for label, wind in (("自风", context.seat_wind), ("场风", context.round_wind)):
        if wind is not None and normalize_tile(wind) not in _WINDS:
            raise ValueError(f"{label}必须是东南西北之一")
    if context.open_tanyao is not None and not isinstance(context.open_tanyao, bool):
        raise ValueError("开放断幺规则必须为布尔值或未知")

    # Missing known groups are unknown too; callers cannot accidentally turn a
    # partially mapped set of melds into an all-hand assertion.
    effective_unknown = context.open_meld_count - len(context.melds)
    return normalized_concealed, tuple(normalized_melds), effective_unknown


def _normalize_tiles(tiles: tuple[str, ...]) -> list[str]:
    normalized = [normalize_tile(tile) for tile in tiles]
    for raw, tile in zip(tiles, normalized, strict=True):
        if tile not in TILE_NAMES:
            raise ValueError(f"未知牌名: {raw}")
    return normalized


def _normal_route(breakdown: ShantenBreakdown) -> ShapeRouteCandidate:
    return ShapeRouteCandidate(
        "normal",
        "一般形",
        "candidate",
        "shanten",
        breakdown.normal,
        evidence=(f"一般形 {breakdown.normal} 向听",),
        warnings=(_GENERAL_WARNING,),
    )


def _special_shape_route(
    route_id: str,
    name: str,
    shanten: int | None,
    open_meld_count: int,
    blocker: str,
) -> ShapeRouteCandidate:
    if open_meld_count:
        return ShapeRouteCandidate(
            route_id, name, "blocked", "shanten", None, blockers=(blocker,), warnings=(_GENERAL_WARNING,)
        )
    assert shanten is not None
    return ShapeRouteCandidate(
        route_id,
        name,
        "candidate",
        "shanten",
        shanten,
        evidence=(f"{name} {shanten} 向听",),
        warnings=(_GENERAL_WARNING,),
    )


def _tanyao_route(
    context: ShapeRouteContext,
    melds: tuple[KnownMeld, ...],
    all_known: list[str],
    unknown_meld_count: int,
) -> ShapeRouteCandidate:
    has_known_open_meld = any(meld.is_open for meld in melds)
    if has_known_open_meld and context.open_tanyao is False:
        return ShapeRouteCandidate(
            "tanyao", "断幺九", "blocked", "discard_conflicts", None,
            blockers=("规则集禁止副露断幺九",), warnings=(_GENERAL_WARNING,)
        )
    if has_known_open_meld and context.open_tanyao is None:
        return ShapeRouteCandidate(
            "tanyao", "断幺九", "insufficient_data", "discard_conflicts", None,
            blockers=("副露断幺九规则未知",), warnings=(_GENERAL_WARNING,)
        )
    if unknown_meld_count:
        return _unknown_whole_hand_route("tanyao", "断幺九")
    conflicts = [tile for tile in all_known if not _is_simple(tile)]
    return _static_route("tanyao", "断幺九", conflicts, "已知牌均为 2～8 数牌")


def _best_suit_route(
    route_id: str,
    name: str,
    all_known: list[str],
    unknown_meld_count: int,
) -> ShapeRouteCandidate:
    if unknown_meld_count:
        return _unknown_whole_hand_route(route_id, name)
    best_suit, conflicts = min(
        (
            (suit, [tile for tile in all_known if not _matches_suit_route(tile, suit, route_id)])
            for suit in _SUITS
        ),
        key=lambda item: (len(item[1]), item[0]),
    )
    return _static_route(
        f"{route_id}:{best_suit}",
        f"{name}（{_SUIT_LABELS[best_suit]}）",
        conflicts,
        f"目标花色：{_SUIT_LABELS[best_suit]}",
    )


def _honroutou_route(all_known: list[str], unknown_meld_count: int) -> ShapeRouteCandidate:
    if unknown_meld_count:
        return _unknown_whole_hand_route("honroutou", "混老头")
    conflicts = [tile for tile in all_known if not _is_terminal_or_honor(tile)]
    return _static_route("honroutou", "混老头", conflicts, "已知牌均为幺九或字牌")


def _yakuhai_routes(
    all_known: list[str],
    unknown_meld_count: int,
    seat_wind: str | None,
    round_wind: str | None,
) -> tuple[ShapeRouteCandidate, ...]:
    routes: list[tuple[str, str, str]] = [
        ("yakuhai:white", "役牌·白", "white"),
        ("yakuhai:green", "役牌·发", "green"),
        ("yakuhai:red", "役牌·中", "red"),
    ]
    if seat_wind is not None:
        routes.append(("yakuhai:seat", "自风牌", normalize_tile(seat_wind)))
    if round_wind is not None:
        routes.append(("yakuhai:round", "场风牌", normalize_tile(round_wind)))

    counts = Counter(all_known)
    warning = (_GENERAL_WARNING,)
    if unknown_meld_count:
        warning += ("未知副露未计入役牌张数，实际缺口可能更小",)
    candidates = []
    for route_id, name, tile in routes:
        known = counts[tile]
        evidence = (f"已知 {tile} {known} 张",)
        if known >= 3:
            evidence += ("已知牌中已具备役牌刻子张数",)
        candidates.append(
            ShapeRouteCandidate(
                route_id,
                name,
                "candidate",
                "tile_copies",
                max(0, 3 - known),
                evidence=evidence,
                warnings=warning,
            )
        )
    return tuple(candidates)


def _static_route(
    route_id: str,
    name: str,
    conflicts: list[str],
    clean_evidence: str,
) -> ShapeRouteCandidate:
    evidence = (clean_evidence,) if not conflicts else (f"冲突牌：{_format_tiles(conflicts)}",)
    return ShapeRouteCandidate(
        route_id,
        name,
        "candidate",
        "discard_conflicts",
        len(conflicts),
        evidence=evidence,
        warnings=(_GENERAL_WARNING,),
    )


def _unknown_whole_hand_route(route_id: str, name: str) -> ShapeRouteCandidate:
    return ShapeRouteCandidate(
        route_id,
        name,
        "insufficient_data",
        "discard_conflicts",
        None,
        blockers=("存在未可靠识别的副露，无法确认全手牌约束",),
        warnings=(_GENERAL_WARNING,),
    )


def _is_simple(tile: str) -> bool:
    return tile not in HONORS and tile[0] not in {"1", "9"}


def _is_terminal_or_honor(tile: str) -> bool:
    return tile in HONORS or tile[0] in {"1", "9"}


def _matches_suit_route(tile: str, suit: str, route_id: str) -> bool:
    if tile in HONORS:
        return route_id == "honitsu"
    return tile[-1] == suit


def _format_tiles(tiles: list[str]) -> str:
    counts = Counter(tiles)
    return " ".join(
        f"{tile}×{counts[tile]}" if counts[tile] > 1 else tile
        for tile in TILE_NAMES
        if tile in counts
    )
