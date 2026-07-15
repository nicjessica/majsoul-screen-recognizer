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
class RecognitionResult:
    hand: list[str]
    draw: str | None
    dora_indicators: list[str]
    meld_tiles: list[str]
    confidence: float
    matches: list[TileMatch]
    meld_error: str | None = None
    melds: list[MeldRecognition] = field(default_factory=list)
