from __future__ import annotations

def calculate_shanten(counts: list[int], open_meld_count: int = 0) -> int:
    if not 0 <= open_meld_count <= 4:
        raise ValueError("副露组数必须在 0 到 4 之间")
    total = sum(counts)
    if total == 0:
        raise ValueError("没有可分析的牌")
    normal = normal_shanten(tuple(counts), open_meld_count)
    if open_meld_count:
        return normal
    return min(normal, chiitoitsu_shanten(counts), kokushi_shanten(counts))


def normal_shanten(counts: tuple[int, ...], open_meld_count: int = 0) -> int:
    min_shanten = 8

    def dfs(current: list[int], index: int, melds: int, pairs: int, taatsu: int) -> None:
        nonlocal min_shanten

        while index < 34 and current[index] == 0:
            index += 1

        if index >= 34:
            usable_taatsu = min(taatsu, 4 - melds)
            shanten = 8 - 2 * melds - usable_taatsu - pairs
            if pairs > 1:
                shanten += pairs - 1
            min_shanten = min(min_shanten, shanten)
            return

        if melds < 4 and current[index] >= 3:
            current[index] -= 3
            dfs(current, index, melds + 1, pairs, taatsu)
            current[index] += 3

        if melds < 4 and _can_sequence(current, index):
            current[index] -= 1
            current[index + 1] -= 1
            current[index + 2] -= 1
            dfs(current, index, melds + 1, pairs, taatsu)
            current[index] += 1
            current[index + 1] += 1
            current[index + 2] += 1

        if pairs < 7 and current[index] >= 2:
            current[index] -= 2
            dfs(current, index, melds, pairs + 1, taatsu)
            current[index] += 2

        if taatsu < 4 and current[index] >= 2:
            current[index] -= 2
            dfs(current, index, melds, pairs, taatsu + 1)
            current[index] += 2

        if taatsu < 4 and _can_ryanmen(current, index):
            current[index] -= 1
            current[index + 1] -= 1
            dfs(current, index, melds, pairs, taatsu + 1)
            current[index] += 1
            current[index + 1] += 1

        if taatsu < 4 and _can_kanchan(current, index):
            current[index] -= 1
            current[index + 2] -= 1
            dfs(current, index, melds, pairs, taatsu + 1)
            current[index] += 1
            current[index + 2] += 1

        current[index] -= 1
        dfs(current, index, melds, pairs, taatsu)
        current[index] += 1

    dfs(list(counts), 0, open_meld_count, 0, 0)
    return min_shanten


def chiitoitsu_shanten(counts: list[int]) -> int:
    pairs = sum(1 for count in counts if count >= 2)
    unique = sum(1 for count in counts if count > 0)
    return 6 - pairs + max(0, 7 - unique)


def kokushi_shanten(counts: list[int]) -> int:
    terminals_and_honors = [0, 8, 9, 17, 18, 26, *range(27, 34)]
    unique = sum(1 for index in terminals_and_honors if counts[index] > 0)
    has_pair = any(counts[index] >= 2 for index in terminals_and_honors)
    return 13 - unique - (1 if has_pair else 0)


def _can_sequence(counts: list[int], index: int) -> bool:
    return index < 27 and index % 9 <= 6 and counts[index + 1] > 0 and counts[index + 2] > 0


def _can_ryanmen(counts: list[int], index: int) -> bool:
    return index < 27 and index % 9 <= 7 and counts[index + 1] > 0


def _can_kanchan(counts: list[int], index: int) -> bool:
    return index < 27 and index % 9 <= 6 and counts[index + 2] > 0
