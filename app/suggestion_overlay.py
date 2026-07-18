from __future__ import annotations

import ctypes
import sys

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QCursor, QGuiApplication, QMouseEvent
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from mahjong.analyzer import HandAnalysis
from recognizer.geometry import ScreenRegion


HONOR_LABELS = {
    "east": "东",
    "south": "南",
    "west": "西",
    "north": "北",
    "white": "白",
    "green": "发",
    "red": "中",
}
SUIT_LABELS = {"m": "万", "p": "筒", "s": "索"}


def tile_label(name: str) -> str:
    if name in HONOR_LABELS:
        return HONOR_LABELS[name]
    normalized = name[:-1] if name.endswith("r") else name
    if len(normalized) == 2 and normalized[0].isdigit() and normalized[1] in SUIT_LABELS:
        prefix = "赤" if name.endswith("r") else ""
        return f"{prefix}{normalized[0]}{SUIT_LABELS[normalized[1]]}"
    return name


def format_overlay_suggestion(analysis: HandAnalysis) -> tuple[str, str]:
    if not analysis.recommendations:
        return "暂无切牌建议", f"当前向听 {analysis.shanten}"

    best = analysis.recommendations[0]
    effective = "、".join(tile_label(name) for name in best.effective_tiles[:8]) or "无"
    if len(best.effective_tiles) > 8:
        effective += "…"
    alternatives = ""
    if len(analysis.recommendations) > 1:
        alternatives = "备选：" + " / ".join(
            tile_label(item.discard) for item in analysis.recommendations[1:3]
        )
    detail = (
        f"{best.resulting_shanten} 向听  ·  有效牌 {best.ukeire_count} 枚\n"
        f"进张：{effective}"
    )
    if alternatives:
        detail += f"\n{alternatives}"
    return f"首选切牌  {tile_label(best.discard)}", detail


class SuggestionOverlay(QWidget):
    position_changed = Signal(float, float)

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowTitle("切牌建议浮层")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self.card = QFrame()
        self.card.setObjectName("overlayCard")
        layout = QVBoxLayout(self.card)
        layout.setContentsMargins(18, 13, 18, 14)
        layout.setSpacing(4)
        self.title_label = QLabel("等待识别")
        self.title_label.setObjectName("overlayTitle")
        self.detail_label = QLabel("稳定结果发布后显示建议")
        self.detail_label.setObjectName("overlayDetail")
        layout.addWidget(self.title_label)
        layout.addWidget(self.detail_label)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.card)
        self._normal_style = """
            QFrame#overlayCard {
                background: rgba(8, 16, 24, 225); border: 2px solid #55d6a8;
                border-radius: 12px;
            }
            QLabel#overlayTitle {
                color: #f6fffb; font-family: "Microsoft YaHei UI";
                font-size: 20px; font-weight: 800;
            }
            QLabel#overlayDetail {
                color: #c9d6df; font-family: "Microsoft YaHei UI";
                font-size: 12px; font-weight: 600;
            }
        """
        self.setStyleSheet(self._normal_style)
        self.capture_excluded = False
        self.adjustment_mode = False
        self._drag_offset: QPointF | None = None
        self._game_region: ScreenRegion | None = None
        self._screen_scale = 1.0

    def show_analysis(
        self,
        analysis: HandAnalysis,
        game_region: ScreenRegion,
        position_x: float = 0.016,
        position_y: float = 0.022,
    ) -> None:
        title, detail = format_overlay_suggestion(analysis)
        self.title_label.setText(title)
        self.detail_label.setText(detail)
        self.adjustSize()
        self._move_into_game_region(game_region, position_x, position_y)
        self.show()
        self.raise_()
        self.capture_excluded = self._exclude_from_capture()

    def mark_stale(self) -> None:
        if self.isVisible() and not self.title_label.text().startswith("⚠"):
            self.title_label.setText("⚠ 旧建议 · " + self.title_label.text())
            self.adjustSize()

    def set_adjustment_mode(self, enabled: bool) -> None:
        self.adjustment_mode = enabled
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not enabled)
        self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor if enabled else Qt.CursorShape.ArrowCursor))
        if enabled:
            self.setStyleSheet(self._normal_style.replace("#55d6a8", "#f4c95d"))
            self.detail_label.setToolTip("拖动卡片调整位置，松开后自动保存")
        else:
            self.setStyleSheet(self._normal_style)
            self.detail_label.setToolTip("")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.adjustment_mode and event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition() - QPointF(self.pos())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.adjustment_mode and self._drag_offset is not None:
            target = event.globalPosition() - self._drag_offset
            self.move(*self._clamped_position(round(target.x()), round(target.y())))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.adjustment_mode and self._drag_offset is not None:
            self._drag_offset = None
            position = self._relative_position()
            if position is not None:
                self.position_changed.emit(*position)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _move_into_game_region(
        self, region: ScreenRegion, position_x: float, position_y: float
    ) -> None:
        screen = self._screen_for_physical_region(region)
        scale = screen.devicePixelRatio() if screen is not None else 1.0
        self._game_region = region
        self._screen_scale = scale
        left = round((region.left + position_x * region.width) / scale)
        top = round((region.top + position_y * region.height) / scale)
        self.move(*self._clamped_position(left, top))

    def _clamped_position(self, left: int, top: int):
        if self._game_region is None:
            return left, top
        region = self._game_region
        scale = self._screen_scale
        minimum_left = round(region.left / scale)
        minimum_top = round(region.top / scale)
        maximum_left = max(minimum_left, round((region.left + region.width) / scale) - self.width())
        maximum_top = max(minimum_top, round((region.top + region.height) / scale) - self.height())
        return (
            min(max(left, minimum_left), maximum_left),
            min(max(top, minimum_top), maximum_top),
        )

    def _relative_position(self) -> tuple[float, float] | None:
        if self._game_region is None:
            return None
        region = self._game_region
        physical_left = self.x() * self._screen_scale
        physical_top = self.y() * self._screen_scale
        return (
            min(1.0, max(0.0, (physical_left - region.left) / region.width)),
            min(1.0, max(0.0, (physical_top - region.top) / region.height)),
        )

    @staticmethod
    def _screen_for_physical_region(region: ScreenRegion):
        for screen in QGuiApplication.screens():
            geometry = screen.geometry()
            scale = screen.devicePixelRatio()
            left = round(geometry.left() * scale)
            top = round(geometry.top() * scale)
            right = left + round(geometry.width() * scale)
            bottom = top + round(geometry.height() * scale)
            if left <= region.left < right and top <= region.top < bottom:
                return screen
        return QGuiApplication.primaryScreen()

    def _exclude_from_capture(self) -> bool:
        if sys.platform != "win32":
            return False
        try:
            hwnd = int(self.winId())
            # WDA_EXCLUDEFROMCAPTURE: Windows 10 2004+. Older systems reject it.
            return bool(ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x11))
        except (AttributeError, OSError, ValueError):
            return False
