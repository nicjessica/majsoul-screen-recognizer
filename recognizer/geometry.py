from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenRegion:
    left: int
    top: int
    width: int
    height: int

