from __future__ import annotations

from recognizer.config import PLAYER_SEATS
from recognizer.models import MeldRecognition, RecognitionResult


def collect_visible_tiles(result: RecognitionResult) -> list[str]:
    """Collect reliably recognized public tiles without inferring hidden tiles."""
    visible = list(result.dora_indicators)
    visible.extend(_meld_tile_names(result.melds, result.meld_tiles))

    opponents = {player.seat: player for player in result.opponent_melds}
    for seat in PLAYER_SEATS[1:]:
        player = opponents.get(seat)
        if player is not None:
            visible.extend(_meld_tile_names(player.melds, player.meld_tiles))
    return visible


def _meld_tile_names(melds: list[MeldRecognition], fallback: list[str]) -> list[str]:
    if not melds:
        return list(fallback)
    return [
        tile.name
        for meld in melds
        for tile in meld.tiles
        if tile.name is not None
    ]
