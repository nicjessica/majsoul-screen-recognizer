from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.screen_select import ScreenRegionSelector
from mahjong.analyzer import analyze_hand
from recognizer.capture import ScreenCapture
from recognizer.config import RelativeRegion, load_config, save_config
from recognizer.geometry import ScreenRegion
from recognizer.recognizer import RecognitionError, TileRecognizer
from recognizer.template_builder import build_templates_from_screenshot


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("雀魂画面识别与牌效分析")
        self.resize(960, 680)

        self.config = load_config()
        self.capture: ScreenCapture | None = None
        self.recognizer: TileRecognizer | None = None
        self.selector: ScreenRegionSelector | None = None
        self.pending_region: str | None = None
        self.capture_in_progress = False
        self.capture_hidden = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.recognize_once)

        self.region_label = QLabel(self.region_text())
        self.layout_label = QLabel(self.layout_text())
        self.status_label = QLabel("未开始")
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)

        select_game_button = QPushButton("框选画面")
        select_game_button.clicked.connect(self.select_game_region)

        select_hand_button = QPushButton("框选手牌区")
        select_hand_button.clicked.connect(lambda: self.select_layout_region("hand"))

        select_draw_button = QPushButton("框选摸牌")
        select_draw_button.clicked.connect(lambda: self.select_layout_region("draw"))

        select_dora_button = QPushButton("框选宝牌")
        select_dora_button.clicked.connect(lambda: self.select_layout_region("dora"))

        select_meld_button = QPushButton("框选副露区")
        select_meld_button.clicked.connect(lambda: self.select_layout_region("meld"))

        counts_button = QPushButton("设置牌数")
        counts_button.clicked.connect(self.set_tile_counts)

        self.start_button = QPushButton("开始识别")
        self.start_button.clicked.connect(self.toggle_recognition)

        once_button = QPushButton("识别一次")
        once_button.clicked.connect(self.recognize_once)

        reload_button = QPushButton("重载模板")
        reload_button.clicked.connect(self.reload_templates)

        build_templates_button = QPushButton("从截图裁模板")
        build_templates_button.clicked.connect(self.build_templates)

        region_row = QHBoxLayout()
        region_row.addWidget(select_game_button)
        region_row.addWidget(select_hand_button)
        region_row.addWidget(select_draw_button)
        region_row.addWidget(select_dora_button)
        region_row.addWidget(select_meld_button)

        action_row = QHBoxLayout()
        action_row.addWidget(counts_button)
        action_row.addWidget(self.start_button)
        action_row.addWidget(once_button)
        action_row.addWidget(reload_button)
        action_row.addWidget(build_templates_button)

        layout = QVBoxLayout()
        layout.addLayout(region_row)
        layout.addLayout(action_row)
        layout.addWidget(self.region_label)
        layout.addWidget(self.layout_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.output, 1)

        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)
        self.print_intro()

    def print_intro(self) -> None:
        template_dir = Path(self.config.templates_dir)
        self.output.setPlainText(
            "使用步骤:\n"
            "1. 点击“框选画面”，选择整个雀魂游戏画面。\n"
            "2. 框选手牌区、摸牌、宝牌；有副露时再框选副露区。\n"
            "3. 点击“设置牌数”，设置手牌区、摸牌区和副露区牌张数。\n"
            "4. 点击“识别一次”或“开始识别”。\n\n"
            f"当前模板目录: {template_dir.resolve()}\n"
            "副露第一版只按从左到右识别牌，不判断吃、碰、杠和横置来源。\n"
        )

    def region_text(self) -> str:
        region = self.config.game_region
        if region is None:
            return "画面区域: 未设置"
        return (
            "画面区域: "
            f"left={region.left}, top={region.top}, width={region.width}, height={region.height}"
        )

    def layout_text(self) -> str:
        layout = self.config.layout
        meld = layout.meld_region
        meld_text = "未设置"
        if meld is not None:
            meld_text = f"x={meld.x:.3f}, y={meld.y:.3f}, w={meld.width:.3f}, h={meld.height:.3f}"
        return (
            "牌区比例: "
            f"手牌 x={layout.hand_region.x:.3f}, y={layout.hand_region.y:.3f}, "
            f"w={layout.hand_region.width:.3f}, h={layout.hand_region.height:.3f}, "
            f"张数={layout.hand_tile_count}; "
            f"摸牌 x={layout.draw_region.x:.3f}, y={layout.draw_region.y:.3f}, "
            f"w={layout.draw_region.width:.3f}, h={layout.draw_region.height:.3f}, "
            f"张数={layout.draw_tile_count}; "
            f"宝牌 x={layout.dora_region.x:.3f}, y={layout.dora_region.y:.3f}, "
            f"w={layout.dora_region.width:.3f}, h={layout.dora_region.height:.3f}; "
            f"副露 {meld_text}, 张数={layout.meld_tile_count}"
        )

    def select_game_region(self) -> None:
        self.pending_region = "game"
        self.open_region_selector()

    def select_layout_region(self, region_kind: str) -> None:
        if self.config.game_region is None:
            QMessageBox.warning(self, "缺少画面区域", "请先点击“框选画面”。")
            return
        self.pending_region = region_kind
        self.open_region_selector()

    def open_region_selector(self) -> None:
        self.hide()
        self.selector = ScreenRegionSelector()
        self.selector.region_selected.connect(self.on_region_selected)
        self.selector.destroyed.connect(self.show)
        QTimer.singleShot(150, self.selector.show)

    def on_region_selected(self, region: ScreenRegion) -> None:
        if self.pending_region == "game":
            self.config.game_region = region
            message = "画面区域已保存"
        elif self.pending_region in {"hand", "draw", "dora", "meld"}:
            relative = self.to_relative_region(region)
            if relative is None:
                QMessageBox.warning(self, "区域无效", "请选择游戏画面内部区域。")
                return
            if self.pending_region == "hand":
                self.config.layout.hand_region = relative
                message = "手牌区已保存"
            elif self.pending_region == "draw":
                self.config.layout.draw_region = relative
                message = "摸牌区已保存"
            elif self.pending_region == "dora":
                self.config.layout.dora_region = relative
                message = "宝牌区已保存"
            else:
                self.config.layout.meld_region = relative
                message = "副露区已保存"
        else:
            return

        save_config(self.config)
        self.recognizer = None
        self.region_label.setText(self.region_text())
        self.layout_label.setText(self.layout_text())
        self.status_label.setText(message)

    def to_relative_region(self, selected: ScreenRegion) -> RelativeRegion | None:
        game = self.config.game_region
        if game is None:
            return None

        left = max(selected.left, game.left)
        top = max(selected.top, game.top)
        right = min(selected.left + selected.width, game.left + game.width)
        bottom = min(selected.top + selected.height, game.top + game.height)
        if right <= left or bottom <= top:
            return None

        return RelativeRegion(
            x=(left - game.left) / game.width,
            y=(top - game.top) / game.height,
            width=(right - left) / game.width,
            height=(bottom - top) / game.height,
        )

    def set_tile_counts(self) -> None:
        layout = self.config.layout
        text, ok = QInputDialog.getText(
            self,
            "设置牌数",
            "输入“手牌区张数 摸牌张数 副露区可见张数”。\n"
            "闭门摸牌: 13 1 0；副露后待切: 10 0 3；杠后补牌: 10 1 3",
            text=f"{layout.hand_tile_count} {layout.draw_tile_count} {layout.meld_tile_count}",
        )
        if not ok:
            return
        parts = text.split()
        if len(parts) != 3:
            QMessageBox.warning(self, "格式错误", "请按格式输入三个数字，例如 10 0 3。")
            return
        try:
            hand_count = int(parts[0])
            draw_count = int(parts[1])
            meld_count = int(parts[2])
        except ValueError:
            QMessageBox.warning(self, "格式错误", "牌数必须是数字。")
            return
        if not (1 <= hand_count <= 13 and 0 <= draw_count <= 1 and 0 <= meld_count <= 16):
            QMessageBox.warning(self, "牌数无效", "手牌区张数应为 1-13，摸牌张数应为 0 或 1，副露区可见张数应为 0-16。")
            return

        self.config.layout.hand_tile_count = hand_count
        self.config.layout.draw_tile_count = draw_count
        self.config.layout.meld_tile_count = meld_count
        save_config(self.config)
        self.recognizer = None
        self.layout_label.setText(self.layout_text())
        self.status_label.setText("牌数已保存")

    def toggle_recognition(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.start_button.setText("开始识别")
            self.status_label.setText("已停止")
            return

        if self.config.game_region is None:
            QMessageBox.warning(self, "缺少画面区域", "请先点击“框选画面”。")
            return

        self.timer.start(int(self.config.capture_interval_seconds * 1000))
        self.start_button.setText("停止识别")
        self.status_label.setText("识别中")

    def reload_templates(self) -> None:
        try:
            self.config = load_config()
            self.recognizer = TileRecognizer(self.config)
        except RecognitionError as exc:
            self.status_label.setText("模板加载失败")
            self.output.setPlainText(str(exc))
            return
        self.layout_label.setText(self.layout_text())
        self.status_label.setText(f"已加载 {len(self.recognizer.templates.templates)} 个模板")

    def build_templates(self) -> None:
        screenshot_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择一张清晰雀魂截图",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not screenshot_path:
            return

        text, ok = QInputDialog.getMultiLineText(
            self,
            "输入牌名",
            "按截图底部从左到右输入手牌区所有立牌；如果摸牌张数为 1，再追加摸牌。\n"
            "闭门摸牌通常是 13 张手牌 + 1 张摸牌；副露后待切通常是 10 张手牌、0 张摸牌。",
        )
        if not ok:
            return

        names = text.split()
        try:
            saved = build_templates_from_screenshot(
                screenshot_path=screenshot_path,
                tile_names=names,
                output_dir=self.config.templates_dir,
                layout=self.config.layout,
            )
            self.recognizer = TileRecognizer(self.config)
        except Exception as exc:
            self.status_label.setText("裁剪模板失败")
            self.output.setPlainText(f"{type(exc).__name__}: {exc}")
            return

        self.status_label.setText(f"已生成 {len(saved)} 个模板")
        self.output.setPlainText("已生成模板:\n" + "\n".join(str(path) for path in saved))

    def recognize_once(self) -> None:
        if self.config.game_region is None:
            self.status_label.setText("请先框选画面")
            return
        if self.capture_in_progress:
            return

        self.capture_in_progress = True
        self.capture_hidden = self.should_hide_for_capture()
        if self.capture_hidden:
            self.hide()
            QTimer.singleShot(180, self._recognize_once_after_hide)
        else:
            self._recognize_once_after_hide()

    def should_hide_for_capture(self) -> bool:
        game = self.config.game_region
        if game is None:
            return False
        return regions_overlap(self.current_window_region(), game)

    def current_window_region(self) -> ScreenRegion:
        geometry = self.frameGeometry()
        screen = QApplication.screenAt(geometry.center()) or QApplication.primaryScreen()
        scale = screen.devicePixelRatio() if screen is not None else 1.0
        return ScreenRegion(
            left=round(geometry.left() * scale),
            top=round(geometry.top() * scale),
            width=round(geometry.width() * scale),
            height=round(geometry.height() * scale),
        )

    def _recognize_once_after_hide(self) -> None:
        frame = None
        try:
            if self.capture is None:
                self.capture = ScreenCapture()
            if self.recognizer is None:
                self.recognizer = TileRecognizer(self.config)
            frame = self.capture.capture_region(self.config.game_region)
            result = self.recognizer.recognize(frame)
        except RecognitionError as exc:
            self.status_label.setText("识别失败")
            debug_text = ""
            if frame is not None and self.recognizer is not None:
                try:
                    debug_dir = self.recognizer.save_debug_tiles(frame)
                    debug_text = f"\n\n已保存本次裁剪结果到:\n{debug_dir.resolve()}"
                except Exception as debug_exc:
                    debug_text = f"\n\n保存诊断图片失败: {type(debug_exc).__name__}: {debug_exc}"
            self.output.setPlainText(str(exc) + debug_text)
            self.finish_capture_display()
            return
        except Exception as exc:  # pragma: no cover - UI safety net
            self.status_label.setText("运行错误")
            self.output.setPlainText(f"{type(exc).__name__}: {exc}")
            self.finish_capture_display()
            return

        tiles = list(result.hand)
        if result.draw:
            tiles.append(result.draw)

        try:
            analysis = analyze_hand(tiles)
            analysis_text = self.format_analysis(analysis)
        except ValueError as exc:
            analysis_text = f"牌效分析不可用: {exc}"

        self.status_label.setText(
            f"识别完成，平均置信度 {result.confidence:.3f}，模板 {len(self.recognizer.templates.templates)} 个"
        )
        self.output.setPlainText(
            "识别结果\n"
            f"手牌: {' '.join(result.hand) if result.hand else '-'}\n"
            f"摸牌: {result.draw or '-'}\n"
            f"宝牌: {' '.join(result.dora_indicators) if result.dora_indicators else '-'}\n"
            f"副露: {' '.join(result.meld_tiles) if result.meld_tiles else '-'}\n"
            f"置信度: {result.confidence:.3f}\n\n"
            "牌效分析\n"
            f"{analysis_text}"
        )
        self.finish_capture_display()

    def finish_capture_display(self) -> None:
        self.capture_in_progress = False
        if self.capture_hidden:
            self.show()
        self.capture_hidden = False

    @staticmethod
    def format_analysis(analysis) -> str:
        lines = [f"当前向听: {analysis.shanten}"]
        if not analysis.recommendations:
            lines.append("没有可用推荐。")
            return "\n".join(lines)

        for index, rec in enumerate(analysis.recommendations[:8], start=1):
            effective = " ".join(rec.effective_tiles) if rec.effective_tiles else "-"
            lines.append(
                f"{index}. 切 {rec.discard}: 向听 {rec.resulting_shanten}, "
                f"有效牌 {rec.ukeire_count} 枚 [{effective}]，{rec.reason}"
            )
        return "\n".join(lines)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


def regions_overlap(a: ScreenRegion, b: ScreenRegion) -> bool:
    return not (
        a.left + a.width <= b.left
        or b.left + b.width <= a.left
        or a.top + a.height <= b.top
        or b.top + b.height <= a.top
    )
