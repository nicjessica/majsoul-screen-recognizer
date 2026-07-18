from __future__ import annotations

from pathlib import Path

import numpy as np

from recognizer.models import TileMatch


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
EDGE_TRIM_FRACTION = 0.10


class TemplateLibrary:
    def __init__(self, templates_dir: str | Path) -> None:
        self.templates_dir = Path(templates_dir)
        self.templates: dict[str, np.ndarray] = {}
        self.load()

    def load(self) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("缺少 opencv-python，请先安装 requirements.txt") from exc

        self.templates.clear()
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        for path in sorted(self.templates_dir.iterdir()):
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            self.templates[path.stem] = image

    def match(self, tile_rgb: np.ndarray) -> TileMatch:
        return self.match_candidates(tile_rgb, limit=1)[0]

    def match_candidates(self, tile_rgb: np.ndarray, limit: int = 2) -> list[TileMatch]:
        if not self.templates:
            raise ValueError(f"模板目录为空: {self.templates_dir}")
        if limit <= 0:
            return []

        import cv2

        tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
        candidates: list[TileMatch] = []
        for name, template in self.templates.items():
            score = match_score(tile_bgr, template)
            candidates.append(TileMatch(name=name, score=score))

        candidates.sort(key=lambda match: (-match.score, match.name))
        return candidates[:limit]


def match_score(tile_bgr: np.ndarray, template_bgr: np.ndarray) -> float:
    scores = [_base_match_score(tile_bgr, template_bgr)]

    # Templates are normally built from the larger hand region.  Other regions
    # can frame the same tile face more tightly on one side, so try removing a
    # small amount from one template edge at a time.  Keeping 90% of the full
    # template prevents this fallback from turning into arbitrary patch matching.
    tile_height, tile_width = tile_bgr.shape[:2]
    height, width = template_bgr.shape[:2]
    trim_y = max(1, round(height * EDGE_TRIM_FRACTION))
    trim_x = max(1, round(width * EDGE_TRIM_FRACTION))
    tile_aspect = tile_width / tile_height
    template_aspect = width / height
    if template_aspect < tile_aspect and trim_y < height:
        scores.append(_base_match_score(tile_bgr, template_bgr[trim_y:]))
        scores.append(_base_match_score(tile_bgr, template_bgr[:-trim_y]))
    elif template_aspect > tile_aspect and trim_x < width:
        scores.append(_base_match_score(tile_bgr, template_bgr[:, trim_x:]))
        scores.append(_base_match_score(tile_bgr, template_bgr[:, :-trim_x]))

    return max(scores)


def _base_match_score(tile_bgr: np.ndarray, template_bgr: np.ndarray) -> float:
    import cv2

    resized = cv2.resize(tile_bgr, (template_bgr.shape[1], template_bgr.shape[0]))
    direct = float(cv2.matchTemplate(resized, template_bgr, cv2.TM_CCOEFF_NORMED).max())

    fixed_size = (128, 192)
    tile_fixed = cv2.resize(tile_bgr, fixed_size)
    template_fixed = cv2.resize(template_bgr, fixed_size)
    height, width = template_fixed.shape[:2]
    margin_x = int(width * 0.10)
    margin_y = int(height * 0.10)
    template_core = template_fixed[margin_y : height - margin_y, margin_x : width - margin_x]
    template_core_gray = cv2.cvtColor(template_core, cv2.COLOR_BGR2GRAY)
    if float(np.std(template_core_gray)) < 1.0:
        return direct
    sliding = float(cv2.matchTemplate(tile_fixed, template_core, cv2.TM_CCOEFF_NORMED).max())

    return max(direct, sliding)
