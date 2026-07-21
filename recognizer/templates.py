from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recognizer.models import TileMatch


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
EDGE_TRIM_FRACTION = 0.10
_FIXED_SIZE = (128, 192)


@dataclass(frozen=True)
class _PreparedVariant:
    template_bgr: np.ndarray
    size: tuple[int, int]
    core: np.ndarray | None


@dataclass(frozen=True)
class _PreparedTemplate:
    aspect: float
    base: _PreparedVariant
    trim_top: _PreparedVariant | None
    trim_bottom: _PreparedVariant | None
    trim_left: _PreparedVariant | None
    trim_right: _PreparedVariant | None


@dataclass
class _BatchMatchCache:
    tile_bgr: np.ndarray
    templates: dict[int, _PreparedTemplate]
    fixed_tile: np.ndarray | None = None
    resized_tiles: dict[tuple[int, int], np.ndarray] | None = None


# ``match_score`` remains a standalone public function.  TemplateLibrary sets
# this short-lived, context-local cache while scoring one tile against its
# library, allowing the exact same scoring operations to reuse tile resizes and
# precomputed template cores without affecting callers of ``match_score``.
_BATCH_MATCH_CACHE: ContextVar[_BatchMatchCache | None] = ContextVar(
    "template_batch_match_cache", default=None
)


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
        self._prepared_templates: dict[str, _PreparedTemplate] = {}
        self._prepared_sources: tuple[tuple[str, int, tuple[int, ...]], ...] = ()
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
        prepared = self._prepared_templates_for_current_library()
        cache = _BatchMatchCache(
            tile_bgr=tile_bgr,
            templates={id(self.templates[name]): item for name, item in prepared.items()},
        )
        token = _BATCH_MATCH_CACHE.set(cache)
        try:
            candidates = [
                TileMatch(name=name, score=match_score(tile_bgr, template))
                for name, template in self.templates.items()
            ]
        finally:
            _BATCH_MATCH_CACHE.reset(token)

        candidates.sort(key=lambda match: (-match.score, match.name))
        return candidates[:limit]

    def _prepared_templates_for_current_library(self) -> dict[str, _PreparedTemplate]:
        sources = tuple(
            (name, id(template), template.shape)
            for name, template in self.templates.items()
        )
        if getattr(self, "_prepared_sources", ()) != sources:
            self._prepared_templates = {
                name: _prepare_template(template)
                for name, template in self.templates.items()
            }
            self._prepared_sources = sources
        return self._prepared_templates


def match_score(tile_bgr: np.ndarray, template_bgr: np.ndarray) -> float:
    cache = _BATCH_MATCH_CACHE.get()
    if cache is not None and cache.tile_bgr is tile_bgr:
        prepared = cache.templates.get(id(template_bgr))
        if prepared is not None:
            return _batch_match_score(cache, prepared)

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

    tile_fixed = cv2.resize(tile_bgr, _FIXED_SIZE)
    template_fixed = cv2.resize(template_bgr, _FIXED_SIZE)
    height, width = template_fixed.shape[:2]
    margin_x = int(width * 0.10)
    margin_y = int(height * 0.10)
    template_core = template_fixed[margin_y : height - margin_y, margin_x : width - margin_x]
    template_core_gray = cv2.cvtColor(template_core, cv2.COLOR_BGR2GRAY)
    if float(np.std(template_core_gray)) < 1.0:
        return direct
    sliding = float(cv2.matchTemplate(tile_fixed, template_core, cv2.TM_CCOEFF_NORMED).max())

    return max(direct, sliding)


def _prepare_template(template_bgr: np.ndarray) -> _PreparedTemplate:
    height, width = template_bgr.shape[:2]
    trim_y = max(1, round(height * EDGE_TRIM_FRACTION))
    trim_x = max(1, round(width * EDGE_TRIM_FRACTION))
    return _PreparedTemplate(
        aspect=width / height,
        base=_prepare_variant(template_bgr),
        trim_top=_prepare_variant(template_bgr[trim_y:]) if trim_y < height else None,
        trim_bottom=_prepare_variant(template_bgr[:-trim_y]) if trim_y < height else None,
        trim_left=_prepare_variant(template_bgr[:, trim_x:]) if trim_x < width else None,
        trim_right=_prepare_variant(template_bgr[:, :-trim_x]) if trim_x < width else None,
    )


def _prepare_variant(template_bgr: np.ndarray) -> _PreparedVariant:
    import cv2

    height, width = template_bgr.shape[:2]
    template_fixed = cv2.resize(template_bgr, _FIXED_SIZE)
    fixed_height, fixed_width = template_fixed.shape[:2]
    margin_x = int(fixed_width * 0.10)
    margin_y = int(fixed_height * 0.10)
    core = template_fixed[
        margin_y : fixed_height - margin_y,
        margin_x : fixed_width - margin_x,
    ]
    core_gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
    return _PreparedVariant(
        template_bgr=template_bgr,
        size=(width, height),
        core=core if float(np.std(core_gray)) >= 1.0 else None,
    )


def _batch_match_score(cache: _BatchMatchCache, template: _PreparedTemplate) -> float:
    tile_height, tile_width = cache.tile_bgr.shape[:2]
    tile_aspect = tile_width / tile_height
    variants = [template.base]
    if template.aspect < tile_aspect:
        variants.extend(item for item in (template.trim_top, template.trim_bottom) if item is not None)
    elif template.aspect > tile_aspect:
        variants.extend(item for item in (template.trim_left, template.trim_right) if item is not None)
    return max(_batch_base_match_score(cache, variant) for variant in variants)


def _batch_base_match_score(cache: _BatchMatchCache, variant: _PreparedVariant) -> float:
    import cv2

    if cache.resized_tiles is None:
        cache.resized_tiles = {}
    resized = cache.resized_tiles.get(variant.size)
    if resized is None:
        resized = cv2.resize(cache.tile_bgr, variant.size)
        cache.resized_tiles[variant.size] = resized
    direct = float(cv2.matchTemplate(resized, variant.template_bgr, cv2.TM_CCOEFF_NORMED).max())
    if variant.core is None:
        return direct
    if cache.fixed_tile is None:
        cache.fixed_tile = cv2.resize(cache.tile_bgr, _FIXED_SIZE)
    sliding = float(cv2.matchTemplate(cache.fixed_tile, variant.core, cv2.TM_CCOEFF_NORMED).max())
    return max(direct, sliding)
