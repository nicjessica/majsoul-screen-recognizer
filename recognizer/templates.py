from __future__ import annotations

from pathlib import Path

import numpy as np

from recognizer.models import TileMatch


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


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
        if not self.templates:
            raise ValueError(f"模板目录为空: {self.templates_dir}")

        import cv2

        tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
        best = TileMatch(name="", score=-1.0)

        for name, template in self.templates.items():
            score = match_score(tile_bgr, template)
            if score > best.score:
                best = TileMatch(name=name, score=score)

        return best


def match_score(tile_bgr: np.ndarray, template_bgr: np.ndarray) -> float:
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
    sliding = float(cv2.matchTemplate(tile_fixed, template_core, cv2.TM_CCOEFF_NORMED).max())

    return max(direct, sliding)
