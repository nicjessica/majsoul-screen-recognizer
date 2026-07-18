from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TileMatch:
    name: str
    score: float


@dataclass(frozen=True)
class MeldTileRecognition:
    name: str | None
    match: TileMatch | None
    error: str | None = None
    candidates: list[TileMatch] = field(default_factory=list)


@dataclass(frozen=True)
class MeldRecognition:
    kind: str
    tiles: list[MeldTileRecognition]
    confidence: float | None
    error: str | None = None


@dataclass(frozen=True)
class PlayerMeldRecognition:
    seat: str
    meld_tiles: list[str]
    melds: list[MeldRecognition]
    error: str | None = None


@dataclass(frozen=True)
class ObservedTileRecognition:
    name: str | None
    match: TileMatch | None
    error: str | None = None
    candidates: list[TileMatch] = field(default_factory=list)
    is_riichi: bool = False
    row: int = 0
    column: int = 0


@dataclass(frozen=True)
class PlayerRiverRecognition:
    seat: str
    tiles: list[ObservedTileRecognition]
    error: str | None = None


@dataclass(frozen=True)
class RoundRecognition:
    round_wind: str | None = None
    hand_number: int | None = None
    confidence: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class WindRecognition:
    wind: str | None = None
    confidence: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class PlayerScoreRecognition:
    seat: str
    score: int | None = None
    confidence: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class TableStateRecognition:
    round: RoundRecognition = field(default_factory=RoundRecognition)
    self_wind: WindRecognition = field(default_factory=WindRecognition)
    scores: list[PlayerScoreRecognition] = field(default_factory=list)


@dataclass(frozen=True)
class RecognitionResult:
    hand: list[str]
    draw: str | None
    dora_indicators: list[str]
    meld_tiles: list[str]
    confidence: float
    matches: list[TileMatch]
    meld_error: str | None = None
    melds: list[MeldRecognition] = field(default_factory=list)
    opponent_melds: list[PlayerMeldRecognition] = field(default_factory=list)
    rivers: list[PlayerRiverRecognition] = field(default_factory=list)
    open_meld_count: int | None = None
    table_state: TableStateRecognition = field(default_factory=TableStateRecognition)
