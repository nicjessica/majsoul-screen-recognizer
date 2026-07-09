from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TileMatch:
    name: str
    score: float


@dataclass(frozen=True)
class RecognitionResult:
    hand: list[str]
    draw: str | None
    dora_indicators: list[str]
    meld_tiles: list[str]
    confidence: float
    matches: list[TileMatch]
