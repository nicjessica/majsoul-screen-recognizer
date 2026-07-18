from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from recognizer.config import AppConfig, RelativeRegion
from recognizer.models import (
    PlayerScoreRecognition,
    RoundRecognition,
    TableStateRecognition,
    WindRecognition,
)
from recognizer.templates import SUPPORTED_EXTENSIONS, match_score


_ROUND_PATTERN = re.compile(r"round_(east|south|west|north)_([1-4])$")
_WIND_PATTERN = re.compile(r"wind_(east|south|west|north)$")
_SCORE_PATTERN = re.compile(r"score_(-?\d+)$")


class TableStateRecognizer:
    """Recognize a deliberately finite set of table-state whole-value templates.

    This first version is template classification, not OCR.  Every configured
    field degrades independently so it cannot invalidate hand recognition.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.round_templates: dict[tuple[str, int], np.ndarray] = {}
        self.wind_templates: dict[str, np.ndarray] = {}
        self.score_templates: dict[int, np.ndarray] = {}
        self.load()

    def load(self) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("缺少 opencv-python，请先安装 requirements.txt") from exc

        self.round_templates.clear()
        self.wind_templates.clear()
        self.score_templates.clear()
        directory = Path(self.config.table_templates_dir)
        if not directory.exists():
            return
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            if match := _ROUND_PATTERN.fullmatch(path.stem):
                self.round_templates[(match.group(1), int(match.group(2)))] = image
            elif match := _WIND_PATTERN.fullmatch(path.stem):
                self.wind_templates[match.group(1)] = image
            elif match := _SCORE_PATTERN.fullmatch(path.stem):
                self.score_templates[int(match.group(1))] = image

    def recognize(self, frame: np.ndarray) -> TableStateRecognition:
        layout = self.config.layout.table_state
        round_result = self._recognize_round(frame, layout.round_region)
        wind_result = self._recognize_wind(frame, layout.self_wind_region)
        scores = [
            self._recognize_score(frame, item.seat, item.region, item.orientation)
            for item in layout.scores
        ]
        return TableStateRecognition(round=round_result, self_wind=wind_result, scores=scores)

    def _recognize_round(
        self, frame: np.ndarray, region: RelativeRegion | None
    ) -> RoundRecognition:
        if region is None:
            return RoundRecognition()
        if not self.round_templates:
            return RoundRecognition(error="No round templates available.")
        crop = _crop_relative(frame, region)
        key, confidence = _best_match(crop, self.round_templates)
        if confidence < self.config.recognition.threshold:
            return RoundRecognition(confidence=confidence, error="Round confidence below threshold.")
        return RoundRecognition(key[0], key[1], confidence)

    def _recognize_wind(
        self, frame: np.ndarray, region: RelativeRegion | None
    ) -> WindRecognition:
        if region is None:
            return WindRecognition()
        if not self.wind_templates:
            return WindRecognition(error="No wind templates available.")
        crop = _crop_relative(frame, region)
        wind, confidence = _best_match(crop, self.wind_templates)
        if confidence < self.config.recognition.threshold:
            return WindRecognition(confidence=confidence, error="Wind confidence below threshold.")
        return WindRecognition(wind, confidence)

    def _recognize_score(
        self,
        frame: np.ndarray,
        seat: str,
        region: RelativeRegion,
        orientation: str,
    ) -> PlayerScoreRecognition:
        if not self.score_templates:
            return PlayerScoreRecognition(seat, error="No score templates available.")
        crop = _normalize_orientation(_crop_relative(frame, region), orientation)
        score, confidence = _best_match(crop, self.score_templates)
        if confidence < self.config.recognition.threshold:
            return PlayerScoreRecognition(
                seat, confidence=confidence, error="Score confidence below threshold."
            )
        return PlayerScoreRecognition(seat, score, confidence)


def _crop_relative(frame: np.ndarray, region: RelativeRegion) -> np.ndarray:
    height, width = frame.shape[:2]
    x1 = round(region.x * width)
    y1 = round(region.y * height)
    x2 = round((region.x + region.width) * width)
    y2 = round((region.y + region.height) * height)
    return frame[y1:y2, x1:x2]


def _normalize_orientation(image: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "rotated_cw":
        return np.ascontiguousarray(np.rot90(image, 1))
    if orientation == "rotated_ccw":
        return np.ascontiguousarray(np.rot90(image, -1))
    if orientation == "rotated_180":
        return np.ascontiguousarray(np.rot90(image, 2))
    return image


def _best_match(image_rgb: np.ndarray, templates: dict) -> tuple[object, float]:
    import cv2

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    candidates = [
        (key, match_score(image_bgr, template)) for key, template in templates.items()
    ]
    return max(candidates, key=lambda item: (item[1], str(item[0])))
