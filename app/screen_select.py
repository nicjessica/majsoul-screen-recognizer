from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from recognizer.geometry import ScreenRegion


def screen_region_from_qrect(rect: QRect) -> ScreenRegion:
    normalized = rect.normalized()
    screen = QApplication.screenAt(normalized.center()) or QApplication.primaryScreen()
    scale = screen.devicePixelRatio() if screen is not None else 1.0
    return ScreenRegion(
        left=round(normalized.left() * scale),
        top=round(normalized.top() * scale),
        width=round(normalized.width() * scale),
        height=round(normalized.height() * scale),
    )


class ScreenRegionSelector(QWidget):
    region_selected = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.start: QPoint | None = None
        self.current: QPoint | None = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setCursor(Qt.CursorShape.CrossCursor)

        geometry = QApplication.primaryScreen().virtualGeometry()
        self.setGeometry(geometry)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.start = event.globalPosition().toPoint()
            self.current = self.start
            self.update()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self.start is not None:
            self.current = event.globalPosition().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton and self.start is not None:
            end = event.globalPosition().toPoint()
            rect = QRect(self.start, end).normalized()
            if rect.width() > 20 and rect.height() > 20:
                self.region_selected.emit(screen_region_from_qrect(rect))
            self.close()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Escape:
            self.close()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 90))

        if self.start is None or self.current is None:
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "拖拽框选目标区域，按 Esc 取消",
            )
            return

        selected = QRect(self.mapFromGlobal(self.start), self.mapFromGlobal(self.current)).normalized()
        painter.fillRect(selected, QColor(60, 150, 255, 45))
        pen = QPen(QColor(60, 150, 255), 2)
        painter.setPen(pen)
        painter.drawRect(selected)
