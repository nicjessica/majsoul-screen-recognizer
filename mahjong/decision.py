from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

from mahjong.analyzer import analyze_hand
from mahjong.tiles import TILE_NAMES, normalize_tile


ActionKind = Literal[
    "skip",
    "discard",
    "damaten",
    "riichi",
    "chi",
    "pon",
    "minkan",
    "ankan",
    "kakan",
]
RelativeWinChance = Literal["higher", "similar", "lower", "unknown"]
Recommendation = Literal["recommended", "consider", "skip_preferred", "illegal"]
Legality = Literal["legal", "illegal", "unverified"]

CALL_KINDS = {"chi", "pon", "minkan"}
KAN_KINDS = {"minkan", "ankan", "kakan"}
MELD_KINDS = {"chi", "pon", "minkan", "ankan", "kakan", "unknown"}


@dataclass(frozen=True)
class MeldState:
    """A declared/fixed meld. ``is_open`` is false only for a concealed kan."""

    kind: str
    tiles: tuple[str, ...]
    is_open: bool = True


@dataclass(frozen=True)
class RoundContext:
    seat_wind: str = "east"
    round_wind: str = "east"
    points: int | None = None
    remaining_draws: int | None = None
    already_riichi: bool = False
    dora_indicators: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionCandidate:
    kind: ActionKind
    called_tile: str | None = None
    consumed_tiles: tuple[str, ...] = ()
    discard_tile: str | None = None
    source: Literal["self", "left", "across", "right"] = "self"


@dataclass(frozen=True)
class ValueEstimate:
    """Only facts guaranteed by currently known tiles/actions.

    ``known_dora`` is a conservative lower bound after the candidate action.
    This is deliberately not a complete Japanese-mahjong scorer. ``points_range``
    and ``fu`` remain unknown until a complete winning shape and all round facts
    are available.
    """

    guaranteed_yaku: tuple[str, ...]
    guaranteed_han: int
    known_dora: int
    fu: int | None = None
    points_range: tuple[int, int] | None = None
    unknown: tuple[str, ...] = (
        "winning tile and wait",
        "complete yaku composition",
        "fu and exact points",
    )


@dataclass(frozen=True)
class ActionEvaluation:
    action: ActionCandidate
    legal: bool
    legality: Legality
    resulting_shanten: int | None
    ukeire_count: int | None
    effective_tiles: tuple[str, ...]
    relative_win_chance: RelativeWinChance
    win_chance_uncertainty: Literal["high"]
    value: ValueEstimate
    recommendation: Recommendation
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class DecisionReport:
    evaluations: tuple[ActionEvaluation, ...]
    preferred_action: ActionCandidate | None
    limitations: tuple[str, ...] = (
        "和率仅按向听数和有效牌作相对比较，不是概率预测",
        "未识别巡目、点棒差、危险度和完整役种时，不给出精确番符与点数",
    )


@dataclass
class _ActionShape:
    legal: bool
    legality: Legality = "legal"
    concealed: list[str] = field(default_factory=list)
    melds: list[MeldState] = field(default_factory=list)
    shanten: int | None = None
    ukeire: int | None = None
    effective: tuple[str, ...] = ()
    reasons: list[str] = field(default_factory=list)


def evaluate_actions(
    concealed_tiles: Sequence[str],
    melds: Sequence[MeldState] = (),
    candidates: Sequence[ActionCandidate] = (),
    visible_tiles: Iterable[str] = (),
    context: RoundContext = RoundContext(),
) -> DecisionReport:
    """Evaluate explicit calls or generate draw/discard decisions.

    ``visible_tiles`` means other known visible tiles and must not repeat the
    player's own ``melds``; declared meld tiles are deducted automatically.
    Calls must include their exact consumed tiles. This keeps legality checks
    deterministic and prevents the domain layer from guessing what the UI meant.
    When ``candidates`` is empty, discard, damaten and riichi candidates are
    generated from the current hand state.
    """

    raw_hand = list(concealed_tiles)
    raw_melds = list(melds)
    normalized_hand = [normalize_tile(tile) for tile in raw_hand]
    _validate_tile_names(normalized_hand)
    normalized_melds = [_normalize_meld(meld) for meld in melds]
    _validate_melds(normalized_melds)
    _validate_known_copies(normalized_hand, normalized_melds)
    _validate_winds(context)
    open_meld_count = len(normalized_melds)
    if open_meld_count > 4:
        raise ValueError("固定面子不能超过 4 组")

    known_visible = tuple(visible_tiles)
    action_candidates = list(candidates) or generate_state_candidates(
        normalized_hand, normalized_melds, known_visible, context
    )
    if not action_candidates:
        action_candidates = [ActionCandidate("skip")]

    baseline_shanten, baseline_ukeire = _baseline_shape(
        normalized_hand, normalized_melds, known_visible
    )
    shapes = [
        _evaluate_shape(
            normalized_hand,
            normalized_melds,
            candidate,
            known_visible,
            context,
        )
        for candidate in action_candidates
    ]

    evaluations: list[ActionEvaluation] = []
    for candidate, shape in zip(action_candidates, shapes, strict=True):
        relative = _relative_win_chance(
            shape.shanten,
            shape.ukeire,
            baseline_shanten,
            baseline_ukeire,
        ) if shape.legal else "unknown"
        if candidate.kind in KAN_KINDS:
            relative = "unknown"
        value = _value_estimate(
            shape.concealed,
            shape.melds,
            candidate,
            context,
            _red_dora_floor_after_action(raw_hand, raw_melds, candidate),
        )
        recommendation = _recommendation(candidate, shape, relative, value)
        evaluations.append(
            ActionEvaluation(
                action=candidate,
                legal=shape.legal,
                legality=shape.legality,
                resulting_shanten=shape.shanten,
                ukeire_count=shape.ukeire,
                effective_tiles=shape.effective,
                relative_win_chance=relative,
                win_chance_uncertainty="high",
                value=value,
                recommendation=recommendation,
                reasons=tuple(shape.reasons),
            )
        )

    preferred = _pick_preferred(evaluations)
    return DecisionReport(tuple(evaluations), preferred)


def generate_state_candidates(
    concealed_tiles: Sequence[str],
    melds: Sequence[MeldState] = (),
    visible_tiles: Iterable[str] = (),
    context: RoundContext = RoundContext(),
) -> list[ActionCandidate]:
    """Generate passive wait or discard/riichi/damaten choices for own turn."""

    normalized = [normalize_tile(tile) for tile in concealed_tiles]
    open_meld_count = len(melds)
    base_count = 13 - 3 * open_meld_count
    if len(normalized) == base_count:
        return [ActionCandidate("skip")]
    if len(normalized) != base_count + 1:
        return []

    analysis = analyze_hand(
        normalized,
        open_meld_count,
        [*visible_tiles, *(tile for meld in melds for tile in meld.tiles)],
    )
    candidates: list[ActionCandidate] = []
    closed = _is_closed(melds)
    for item in analysis.recommendations:
        discard = item.discard
        candidates.append(ActionCandidate("discard", discard_tile=discard))
        if item.resulting_shanten == 0 and closed:
            candidates.append(ActionCandidate("damaten", discard_tile=discard))
            candidates.append(ActionCandidate("riichi", discard_tile=discard))
    return candidates


def generate_call_candidates(
    concealed_tiles: Sequence[str],
    called_tile: str,
    source: Literal["left", "across", "right"],
) -> list[ActionCandidate]:
    """Generate every strictly legal call shape for one opponent discard.

    The first candidate is always ``skip``. Red fives are normalized only for
    shape checks; ``consumed_tiles`` retains the actual tile names so callers
    can distinguish consuming a red five from consuming a normal five.
    """

    if source not in {"left", "across", "right"}:
        raise ValueError(f"鸣牌来源必须是对手方位: {source}")
    normalized_called = normalize_tile(called_tile)
    if normalized_called not in TILE_NAMES:
        raise ValueError(f"未知牌名: {called_tile}")

    raw_hand = list(concealed_tiles)
    normalized_hand = [normalize_tile(tile) for tile in raw_hand]
    _validate_tile_names(normalized_hand)
    if Counter(normalized_hand)[normalized_called] >= 4:
        raise ValueError(f"被鸣牌与暗牌合计超过 4 张: {normalized_called}")

    candidates = [ActionCandidate("skip")]

    if source == "left" and normalized_called in TILE_NAMES[:27]:
        number = int(normalized_called[0])
        suit = normalized_called[-1]
        for start in range(max(1, number - 2), min(number, 7) + 1):
            sequence = [f"{value}{suit}" for value in range(start, start + 3)]
            needed = sequence.copy()
            needed.remove(normalized_called)
            for consumed in _actual_tile_selections(raw_hand, needed):
                candidates.append(
                    ActionCandidate("chi", called_tile, consumed, source=source)
                )

    matching = [tile for tile in raw_hand if normalize_tile(tile) == normalized_called]
    for consumed in _unique_combinations(matching, 2):
        candidates.append(ActionCandidate("pon", called_tile, consumed, source=source))
    for consumed in _unique_combinations(matching, 3):
        candidates.append(ActionCandidate("minkan", called_tile, consumed, source=source))
    return candidates


def _actual_tile_selections(
    raw_hand: Sequence[str], normalized_needed: Sequence[str]
) -> list[tuple[str, ...]]:
    selections: list[tuple[str, ...]] = [()]
    for needed in normalized_needed:
        choices = sorted({tile for tile in raw_hand if normalize_tile(tile) == needed})
        selections = [(*selection, choice) for selection in selections for choice in choices]
    return list(dict.fromkeys(selections))


def _unique_combinations(tiles: Sequence[str], count: int) -> list[tuple[str, ...]]:
    canonical = (tuple(sorted(selection)) for selection in combinations(tiles, count))
    return list(dict.fromkeys(canonical))


def _evaluate_shape(
    concealed: list[str],
    melds: list[MeldState],
    candidate: ActionCandidate,
    visible_tiles: Iterable[str],
    context: RoundContext,
) -> _ActionShape:
    kind = candidate.kind
    if kind not in {"skip", "discard", "damaten", "riichi", *CALL_KINDS, "ankan", "kakan"}:
        return _illegal(concealed, melds, f"未知动作: {kind}")

    base_count = 13 - 3 * len(melds)
    if kind == "skip":
        if len(concealed) != base_count:
            return _illegal(concealed, melds, "待切状态不能跳过出牌")
        return _analyze(concealed, melds, visible_tiles, "跳过鸣牌，保留门前牌形")

    if kind in {"discard", "damaten", "riichi"}:
        if len(concealed) != base_count + 1:
            return _illegal(concealed, melds, "只有待切状态可以选择出牌")
        discard = normalize_tile(candidate.discard_tile or "")
        if discard not in concealed:
            return _illegal(concealed, melds, "指定切牌不在暗牌中")
        after = concealed.copy()
        after.remove(discard)
        shape = _analyze(after, melds, [*visible_tiles, discard], f"切出 {discard}")
        if kind in {"damaten", "riichi"} and shape.shanten != 0:
            return _illegal(concealed, melds, "默听或立直必须在切牌后听牌")
        if kind == "damaten" and not _is_closed(melds):
            return _illegal(concealed, melds, "副露手不能选择门清默听")
        if kind == "riichi":
            reason = _riichi_illegal_reason(melds, context)
            if reason:
                return _illegal(concealed, melds, reason)
            # Missing point/remaining-wall data does not prove illegality.
            if context.points is None or context.remaining_draws is None:
                shape.legality = "unverified"
                shape.reasons.append("点棒或牌山余量未识别，立直合法性仍需画面确认")
            shape.reasons.append("立直增加确定 1 番，但锁定手牌并失去后续优化自由")
        elif kind == "damaten":
            shape.reasons.append("默听保留手牌调整自由，但当前信息不足以确认必有役")
        return shape

    if kind in CALL_KINDS:
        return _evaluate_call(concealed, melds, candidate, visible_tiles)
    if kind == "ankan":
        return _evaluate_ankan(concealed, melds, candidate, visible_tiles)
    return _evaluate_kakan(concealed, melds, candidate, visible_tiles)


def _evaluate_call(
    concealed: list[str],
    melds: list[MeldState],
    candidate: ActionCandidate,
    visible_tiles: Iterable[str],
) -> _ActionShape:
    if len(melds) >= 4:
        return _illegal(concealed, melds, "已有 4 组固定面子，不能继续鸣牌")
    if len(concealed) != 13 - 3 * len(melds):
        return _illegal(concealed, melds, "吃碰杠只在他家打牌后的待摸状态评估")
    if candidate.source == "self":
        return _illegal(concealed, melds, "不能鸣自己的牌")
    called = normalize_tile(candidate.called_tile or "")
    consumed = tuple(normalize_tile(tile) for tile in candidate.consumed_tiles)
    if called not in TILE_NAMES:
        return _illegal(concealed, melds, "缺少合法的被鸣牌")
    if not _contains_tiles(concealed, consumed):
        return _illegal(concealed, melds, "暗牌中没有动作所需的全部牌")

    tiles = (*consumed, called)
    if candidate.kind == "chi":
        if candidate.source != "left":
            return _illegal(concealed, melds, "只能吃上家的牌")
        if len(consumed) != 2 or not _is_sequence(tiles):
            return _illegal(concealed, melds, "吃必须由同花色连续三张组成")
    elif candidate.kind == "pon":
        if len(consumed) != 2 or len(set(tiles)) != 1:
            return _illegal(concealed, melds, "碰必须使用两张同牌")
    elif len(consumed) != 3 or len(set(tiles)) != 1:
        return _illegal(concealed, melds, "明杠必须使用三张同牌")

    after = concealed.copy()
    for tile in consumed:
        after.remove(tile)
    next_melds = [*melds, MeldState(candidate.kind, tuple(tiles), True)]
    shape = _analyze(after, next_melds, visible_tiles, f"{candidate.kind} 后重新评估牌效")
    if candidate.kind == "minkan":
        shape.reasons.append("杠会改变宝牌与牌山信息；未识别完整局况时只建议谨慎考虑")
    return shape


def _evaluate_ankan(
    concealed: list[str],
    melds: list[MeldState],
    candidate: ActionCandidate,
    visible_tiles: Iterable[str],
) -> _ActionShape:
    if len(melds) >= 4 or len(concealed) != 14 - 3 * len(melds):
        return _illegal(concealed, melds, "暗杠只在自己的待切状态评估")
    consumed = tuple(normalize_tile(tile) for tile in candidate.consumed_tiles)
    if len(consumed) != 4 or len(set(consumed)) != 1 or not _contains_tiles(concealed, consumed):
        return _illegal(concealed, melds, "暗杠必须使用暗牌中的四张同牌")
    after = concealed.copy()
    for tile in consumed:
        after.remove(tile)
    next_melds = [*melds, MeldState("ankan", consumed, False)]
    shape = _analyze(after, next_melds, visible_tiles, "暗杠后等待岭上补牌")
    shape.reasons.append("暗杠不会破坏门清，但会改变宝牌与牌山信息")
    return shape


def _evaluate_kakan(
    concealed: list[str],
    melds: list[MeldState],
    candidate: ActionCandidate,
    visible_tiles: Iterable[str],
) -> _ActionShape:
    if len(concealed) != 14 - 3 * len(melds):
        return _illegal(concealed, melds, "加杠只在自己的待切状态评估")
    consumed = tuple(normalize_tile(tile) for tile in candidate.consumed_tiles)
    if len(consumed) != 1 or not _contains_tiles(concealed, consumed):
        return _illegal(concealed, melds, "加杠必须使用暗牌中的一张牌")
    tile = consumed[0]
    pon_index = next(
        (index for index, meld in enumerate(melds) if meld.kind == "pon" and set(meld.tiles) == {tile}),
        None,
    )
    if pon_index is None:
        return _illegal(concealed, melds, "没有可加杠的同牌碰")
    after = concealed.copy()
    after.remove(tile)
    next_melds = melds.copy()
    pon = next_melds[pon_index]
    next_melds[pon_index] = MeldState("kakan", (*pon.tiles, tile), True)
    shape = _analyze(after, next_melds, visible_tiles, "加杠后等待岭上补牌")
    shape.reasons.append("加杠存在抢杠和新增宝牌的局况风险，当前仅提供牌效比较")
    return shape


def _analyze(
    concealed: list[str],
    melds: list[MeldState],
    visible_tiles: Iterable[str],
    reason: str,
) -> _ActionShape:
    try:
        analysis = analyze_hand(
            concealed,
            len(melds),
            [*visible_tiles, *(tile for meld in melds for tile in meld.tiles)],
        )
    except ValueError as exc:
        return _illegal(concealed, melds, str(exc))
    best = analysis.recommendations[0]
    return _ActionShape(
        legal=True,
        concealed=concealed,
        melds=melds,
        shanten=analysis.shanten if best.discard == "-" else best.resulting_shanten,
        ukeire=best.ukeire_count,
        effective=tuple(best.effective_tiles),
        reasons=[reason],
    )


def _baseline_shape(
    concealed: list[str],
    melds: Sequence[MeldState],
    visible_tiles: Iterable[str],
) -> tuple[int | None, int | None]:
    try:
        analysis = analyze_hand(
            concealed,
            len(melds),
            [*visible_tiles, *(tile for meld in melds for tile in meld.tiles)],
        )
    except ValueError:
        return None, None
    best = analysis.recommendations[0]
    shanten = analysis.shanten if best.discard == "-" else best.resulting_shanten
    return shanten, best.ukeire_count


def _relative_win_chance(
    shanten: int | None,
    ukeire: int | None,
    baseline_shanten: int | None,
    baseline_ukeire: int | None,
) -> RelativeWinChance:
    if None in (shanten, ukeire, baseline_shanten, baseline_ukeire):
        return "unknown"
    if shanten < baseline_shanten:
        return "higher"
    if shanten > baseline_shanten:
        return "lower"
    difference = ukeire - baseline_ukeire
    if difference >= 4:
        return "higher"
    if difference <= -4:
        return "lower"
    return "similar"


def _value_estimate(
    concealed: list[str],
    melds: list[MeldState],
    action: ActionCandidate,
    context: RoundContext,
    known_red_dora: int,
) -> ValueEstimate:
    yaku: list[str] = []
    han = 0
    if action.kind == "riichi":
        yaku.append("riichi")
        han += 1

    for meld in melds:
        if meld.kind not in {"pon", "minkan", "ankan", "kakan"} or not meld.tiles:
            continue
        tile = meld.tiles[0]
        if tile in {"white", "green", "red"}:
            yaku.append(f"yakuhai:{tile}")
            han += 1
        if tile == normalize_tile(context.seat_wind):
            yaku.append("yakuhai:seat")
            han += 1
        if tile == normalize_tile(context.round_wind):
            yaku.append("yakuhai:round")
            han += 1

    all_known = [*concealed, *(tile for meld in melds for tile in meld.tiles)]
    dora_tiles = [_dora_from_indicator(tile) for tile in context.dora_indicators]
    known_dora = sum(all_known.count(tile) for tile in dora_tiles)
    known_dora += known_red_dora
    return ValueEstimate(tuple(yaku), han, known_dora)


def _recommendation(
    candidate: ActionCandidate,
    shape: _ActionShape,
    relative: RelativeWinChance,
    value: ValueEstimate,
) -> Recommendation:
    if not shape.legal:
        return "illegal"
    if candidate.kind == "skip":
        return "recommended"
    if candidate.kind == "discard":
        return "recommended" if relative in {"higher", "similar"} else "skip_preferred"
    if candidate.kind in {"riichi", "damaten"}:
        return "consider"
    if candidate.kind in KAN_KINDS:
        return "consider"
    if relative == "higher":
        return "recommended"
    if relative == "similar" and value.guaranteed_han > 0:
        return "consider"
    return "skip_preferred"


def _pick_preferred(evaluations: Sequence[ActionEvaluation]) -> ActionCandidate | None:
    legal = [item for item in evaluations if item.legal]
    if not legal:
        return None
    rank = {"recommended": 0, "consider": 1, "skip_preferred": 2, "illegal": 3}
    kind_rank = {"discard": 0, "skip": 1, "riichi": 2, "damaten": 3}
    best = min(
        legal,
        key=lambda item: (
            rank[item.recommendation],
            item.resulting_shanten if item.resulting_shanten is not None else 99,
            -(item.ukeire_count or 0),
            kind_rank.get(item.action.kind, 4),
        ),
    )
    return best.action


def _riichi_illegal_reason(melds: Sequence[MeldState], context: RoundContext) -> str | None:
    if not _is_closed(melds):
        return "有明副露时不能立直"
    if context.already_riichi:
        return "已经立直，不能重复立直"
    if context.points is not None and context.points < 1000:
        return "点棒不足 1000 点，不能立直"
    if context.remaining_draws is not None and context.remaining_draws < 4:
        return "牌山剩余不足 4 张，不能立直"
    return None


def _normalize_meld(meld: MeldState) -> MeldState:
    return MeldState(meld.kind, tuple(normalize_tile(tile) for tile in meld.tiles), meld.is_open)


def _validate_melds(melds: Sequence[MeldState]) -> None:
    for meld in melds:
        if meld.kind not in MELD_KINDS:
            raise ValueError(f"未知副露类型: {meld.kind}")
        _validate_tile_names(meld.tiles)
        expected = 4 if meld.kind in KAN_KINDS else 3
        if meld.kind != "unknown" and len(meld.tiles) != expected:
            raise ValueError(f"{meld.kind} 必须包含 {expected} 张牌")
        if meld.kind == "chi" and not _is_sequence(meld.tiles):
            raise ValueError("chi 必须由同花色连续三张组成")
        if meld.kind == "pon" and len(set(meld.tiles)) != 1:
            raise ValueError("pon 必须由三张同牌组成")
        if meld.kind in KAN_KINDS and len(set(meld.tiles)) != 1:
            raise ValueError(f"{meld.kind} 必须由四张同牌组成")
        if meld.kind == "ankan" and meld.is_open:
            raise ValueError("暗杠必须标记为非明副露")
        if meld.kind != "ankan" and meld.kind != "unknown" and not meld.is_open:
            raise ValueError(f"{meld.kind} 必须标记为明副露")


def _validate_known_copies(concealed: Sequence[str], melds: Sequence[MeldState]) -> None:
    counts = Counter([*concealed, *(tile for meld in melds for tile in meld.tiles)])
    for tile, count in counts.items():
        if count > 4:
            raise ValueError(f"已知同一种牌超过 4 张: {tile}")


def _validate_tile_names(tiles: Iterable[str]) -> None:
    for tile in tiles:
        if tile not in TILE_NAMES:
            raise ValueError(f"未知牌名: {tile}")


def _validate_winds(context: RoundContext) -> None:
    winds = {"east", "south", "west", "north"}
    if normalize_tile(context.seat_wind) not in winds or normalize_tile(context.round_wind) not in winds:
        raise ValueError("场风和自风必须是东南西北之一")


def _contains_tiles(hand: Sequence[str], needed: Sequence[str]) -> bool:
    hand_counts = Counter(hand)
    return all(hand_counts[tile] >= count for tile, count in Counter(needed).items())


def _is_sequence(tiles: Sequence[str]) -> bool:
    if len(tiles) != 3 or any(tile not in TILE_NAMES[:27] for tile in tiles):
        return False
    suits = {tile[-1] for tile in tiles}
    numbers = sorted(int(tile[0]) for tile in tiles)
    return len(suits) == 1 and numbers[1] == numbers[0] + 1 and numbers[2] == numbers[1] + 1


def _is_closed(melds: Sequence[MeldState]) -> bool:
    return all(meld.kind == "ankan" and not meld.is_open for meld in melds)


def _dora_from_indicator(raw_indicator: str) -> str:
    indicator = normalize_tile(raw_indicator)
    if indicator in TILE_NAMES[:27]:
        number = int(indicator[0])
        return f"{1 if number == 9 else number + 1}{indicator[-1]}"
    wind_cycle = ("east", "south", "west", "north")
    dragon_cycle = ("white", "green", "red")
    if indicator in wind_cycle:
        return wind_cycle[(wind_cycle.index(indicator) + 1) % len(wind_cycle)]
    if indicator in dragon_cycle:
        return dragon_cycle[(dragon_cycle.index(indicator) + 1) % len(dragon_cycle)]
    raise ValueError(f"未知宝牌指示牌: {raw_indicator}")


def _red_dora_floor_after_action(
    raw_hand: Sequence[str],
    raw_melds: Sequence[MeldState],
    action: ActionCandidate,
) -> int:
    count = sum(
        1
        for tile in [*raw_hand, *(tile for meld in raw_melds for tile in meld.tiles)]
        if tile in {"5mr", "5pr", "5sr"}
    )
    if action.kind in {"discard", "damaten", "riichi"} and action.discard_tile:
        normalized_discard = normalize_tile(action.discard_tile)
        if normalized_discard in {"5m", "5p", "5s"}:
            red_name = f"{normalized_discard}r"
            if red_name in raw_hand:
                # The current analyzer normalizes red fives. If a normal and red
                # five coexist, assuming the red one is discarded is the only
                # safe lower bound for value.
                count -= 1
    if action.kind in CALL_KINDS and action.called_tile in {"5mr", "5pr", "5sr"}:
        count += 1
    return count


def _illegal(concealed: list[str], melds: list[MeldState], reason: str) -> _ActionShape:
    return _ActionShape(False, "illegal", concealed.copy(), melds.copy(), reasons=[reason])
