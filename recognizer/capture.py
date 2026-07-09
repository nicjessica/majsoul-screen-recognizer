from __future__ import annotations

import numpy as np
from PIL import Image

from recognizer.geometry import ScreenRegion


class ScreenCapture:
    def __init__(self) -> None:
        try:
            import mss
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("缺少 mss，请先安装 requirements.txt") from exc
        self._mss = mss.mss()

    def capture_region(self, region: ScreenRegion) -> np.ndarray:
        monitor = {
            "left": region.left,
            "top": region.top,
            "width": region.width,
            "height": region.height,
        }
        shot = self._mss.grab(monitor)
        image = Image.frombytes("RGB", shot.size, shot.rgb)
        return np.array(image)
