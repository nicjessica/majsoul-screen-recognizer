from __future__ import annotations

SUITS = ("m", "p", "s")
HONORS = ("east", "south", "west", "north", "white", "green", "red")
TILE_NAMES = tuple(
    [f"{number}{suit}" for suit in SUITS for number in range(1, 10)] + list(HONORS)
)
TILE_TO_INDEX = {name: index for index, name in enumerate(TILE_NAMES)}
INDEX_TO_TILE = {index: name for name, index in TILE_TO_INDEX.items()}


def normalize_tile(tile: str) -> str:
    aliases = {
        "E": "east",
        "S": "south",
        "W": "west",
        "N": "north",
        "P": "white",
        "F": "green",
        "C": "red",
        "ton": "east",
        "nan": "south",
        "sha": "west",
        "pei": "north",
        "haku": "white",
        "hatsu": "green",
        "chun": "red",
        "5mr": "5m",
        "5pr": "5p",
        "5sr": "5s",
    }
    return aliases.get(tile, tile)


def tiles_to_counts(tiles: list[str]) -> list[int]:
    counts = [0] * 34
    for raw in tiles:
        tile = normalize_tile(raw)
        if tile not in TILE_TO_INDEX:
            raise ValueError(f"未知牌名: {raw}")
        index = TILE_TO_INDEX[tile]
        counts[index] += 1
        if counts[index] > 4:
            raise ValueError(f"同一种牌超过4张: {tile}")
    return counts


def counts_to_tiles(counts: list[int]) -> list[str]:
    tiles: list[str] = []
    for index, count in enumerate(counts):
        tiles.extend([INDEX_TO_TILE[index]] * count)
    return tiles

