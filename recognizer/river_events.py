from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from recognizer.models import RecognitionResult


OpponentSeat = Literal["right", "across", "left"]
_OPPONENT_SEATS: frozenset[str] = frozenset(("right", "across", "left"))
_Coordinate = tuple[OpponentSeat, int, int]


@dataclass(frozen=True)
class RiverDiscardEvent:
    event_id: int
    tile: str
    source: OpponentSeat
    row: int
    column: int


class RiverEventTracker:
    """Detect a newly visible opponent discard between stable recognition results."""

    def __init__(self) -> None:
        self._baseline: dict[_Coordinate, str | None] | None = None
        self._next_event_id = 1

    def reset(self) -> None:
        """Forget the comparison baseline while keeping event IDs monotonic."""
        self._baseline = None

    def observe(self, result: RecognitionResult) -> RiverDiscardEvent | None:
        current = self._river_state(result)
        previous = self._baseline
        # Every observation advances the lifecycle, including ambiguous frames.
        self._baseline = current

        if previous is None:
            return None

        changed_coordinates = [
            coordinate
            for coordinate in previous.keys() | current.keys()
            if previous.get(coordinate) != current.get(coordinate)
        ]
        if len(changed_coordinates) != 1:
            return None

        coordinate = changed_coordinates[0]
        tile = current.get(coordinate)
        if coordinate in previous or tile is None:
            return None

        source, row, column = coordinate
        event = RiverDiscardEvent(
            event_id=self._next_event_id,
            tile=tile,
            source=source,
            row=row,
            column=column,
        )
        self._next_event_id += 1
        return event

    @staticmethod
    def _river_state(result: RecognitionResult) -> dict[_Coordinate, str | None]:
        state: dict[_Coordinate, str | None] = {}
        for river in result.rivers:
            if river.seat not in _OPPONENT_SEATS:
                continue
            source = cast(OpponentSeat, river.seat)
            for tile in river.tiles:
                state[(source, tile.row, tile.column)] = tile.name
        return state
