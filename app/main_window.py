from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.screen_select import ScreenRegionSelector
from mahjong.analyzer import analyze_hand
from recognizer.capture import ScreenCapture
from recognizer.config import (
    MELD_KINDS,
    MELD_ORIENTATIONS,
    MeldConfig,
    MeldTileSlotConfig,
    PlayerMeldLayoutConfig,
    PlayerRiverLayoutConfig,
    RelativeRegion,
    RiverTileSlotConfig,
    load_config,
    save_config,
    validate_config,
)
from recognizer.geometry import ScreenRegion
from recognizer.recognizer import RecognitionError, TileRecognizer
from recognizer.stability import KeyRegionSnapshot, RecognitionStabilizer
from recognizer.template_builder import build_templates_from_screenshot
from recognizer.visible_tiles import collect_visible_tiles


SEAT_LABELS = {"self": "自己", "right": "右家", "across": "对家", "left": "左家"}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("雀魂画面识别与牌效分析")
        self.resize(1180, 760)
        self.setMinimumSize(980, 640)

        self.config = load_config()
        self.capture: ScreenCapture | None = None
        self.recognizer: TileRecognizer | None = None
        self.selector: ScreenRegionSelector | None = None
        self.pending_region: str | None = None
        self.capture_in_progress = False
        self.capture_hidden = False
        self.force_publish_current = False
        self.stability = RecognitionStabilizer(required_observations=3)
        self.selector_completed = False
        self.pending_melds: list[MeldConfig] | None = None
        self.pending_meld_slots: list[tuple[int, int]] = []
        self.pending_meld_seat = "self"
        self.pending_river_seat = "self"
        self.pending_river_slots: list[RiverTileSlotConfig] | None = None
        self.pending_river_slot_indexes: list[int] = []
        self.resume_timer_after_meld_config = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.recognize_once)

        self.region_label = QLabel(self.region_text())
        self.region_label.setObjectName("summaryText")
        self.region_label.setWordWrap(True)
        self.layout_label = QLabel(self.layout_text())
        self.layout_label.setObjectName("summaryText")
        self.layout_label.setWordWrap(True)
        self.status_label = QLabel("未开始")
        self.status_label.setObjectName("statusPill")
        self.output = QPlainTextEdit()
        self.output.setObjectName("resultOutput")
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

        select_opponent_meld_button = QPushButton("框选他家副露区")
        select_opponent_meld_button.clicked.connect(self.select_opponent_meld_region)

        select_river_button = QPushButton("框选牌河区")
        select_river_button.clicked.connect(self.select_river_region)

        counts_button = QPushButton("设置牌数")
        counts_button.clicked.connect(self.set_tile_counts)

        dora_count_button = QPushButton("设置宝牌张数")
        dora_count_button.clicked.connect(self.set_dora_tile_count)

        meld_structure_button = QPushButton("配置副露结构")
        meld_structure_button.clicked.connect(self.configure_meld_structure)

        opponent_meld_structure_button = QPushButton("配置他家副露")
        opponent_meld_structure_button.clicked.connect(self.configure_opponent_meld_structure)

        river_structure_button = QPushButton("配置牌河牌槽")
        river_structure_button.clicked.connect(self.configure_river_slots)

        threshold_button = QPushButton("设置识别阈值")
        threshold_button.clicked.connect(self.set_recognition_threshold)

        self.start_button = QPushButton("开始识别")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.toggle_recognition)

        once_button = QPushButton("识别一次")
        once_button.clicked.connect(self.recognize_once)

        reload_button = QPushButton("重载模板")
        reload_button.clicked.connect(self.reload_templates)

        build_templates_button = QPushButton("从截图裁模板")
        build_templates_button.clicked.connect(self.build_templates)

        for button in (
            select_game_button, select_hand_button, select_draw_button,
            select_dora_button, select_meld_button, select_opponent_meld_button,
            select_river_button, counts_button, dora_count_button,
            meld_structure_button, opponent_meld_structure_button,
            river_structure_button, threshold_button, once_button,
            reload_button, build_templates_button,
        ):
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        eyebrow = QLabel("LOCAL VISION · TILE EFFICIENCY")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("雀魂画面识别")
        title.setObjectName("appTitle")
        subtitle = QLabel("本地截图识别与牌效分析控制台")
        subtitle.setObjectName("appSubtitle")

        brand = QVBoxLayout()
        brand.setSpacing(2)
        brand.addWidget(eyebrow)
        brand.addWidget(title)
        brand.addWidget(subtitle)

        run_bar = QHBoxLayout()
        run_bar.setSpacing(10)
        run_bar.addLayout(brand, 1)
        run_bar.addWidget(self.status_label)
        run_bar.addWidget(once_button)
        run_bar.addWidget(self.start_button)

        region_card, region_body = self._make_card("01  识别区域", "按当前游戏画面逐项框选")
        region_grid = QGridLayout()
        region_grid.setSpacing(8)
        for index, button in enumerate((
            select_game_button, select_hand_button, select_draw_button,
            select_dora_button, select_meld_button, select_opponent_meld_button,
            select_river_button,
        )):
            region_grid.addWidget(button, index // 2, index % 2)
        region_body.addLayout(region_grid)

        structure_card, structure_body = self._make_card("02  牌面结构", "配置张数、牌槽与识别条件")
        structure_grid = QGridLayout()
        structure_grid.setSpacing(8)
        for index, button in enumerate((
            counts_button, dora_count_button, meld_structure_button,
            opponent_meld_structure_button, river_structure_button,
            threshold_button,
        )):
            structure_grid.addWidget(button, index // 2, index % 2)
        structure_body.addLayout(structure_grid)

        template_card, template_body = self._make_card("03  模板工具", "维护本机牌面模板库")
        template_row = QHBoxLayout()
        template_row.setSpacing(8)
        template_row.addWidget(reload_button)
        template_row.addWidget(build_templates_button)
        template_body.addLayout(template_row)

        summary_card, summary_body = self._make_card("当前配置", "保存后立即用于下一次识别")
        summary_body.addWidget(self.region_label)
        summary_body.addWidget(self.layout_label)

        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(0, 0, 8, 0)
        sidebar_layout.setSpacing(12)
        sidebar_layout.addWidget(region_card)
        sidebar_layout.addWidget(structure_card)
        sidebar_layout.addWidget(template_card)
        sidebar_layout.addWidget(summary_card)
        sidebar_layout.addStretch(1)

        sidebar = QScrollArea()
        sidebar.setObjectName("sidebar")
        sidebar.setWidgetResizable(True)
        sidebar.setFrameShape(QFrame.Shape.NoFrame)
        sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar.setWidget(sidebar_content)

        result_card, result_body = self._make_card("识别与牌效结果", "稳定画面发布后自动更新")
        result_body.addWidget(self.output, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("workspaceSplitter")
        splitter.addWidget(sidebar)
        splitter.addWidget(result_card)
        splitter.setSizes([390, 750])
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(18)
        layout.addLayout(run_bar)
        layout.addWidget(splitter, 1)

        root = QWidget()
        root.setObjectName("appRoot")
        root.setLayout(layout)
        self.setCentralWidget(root)
        self._apply_theme()
        self.print_intro()

    @staticmethod
    def _make_card(title: str, caption: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("card")
        body = QVBoxLayout(card)
        body.setContentsMargins(16, 14, 16, 16)
        body.setSpacing(10)
        heading = QLabel(title)
        heading.setObjectName("cardTitle")
        detail = QLabel(caption)
        detail.setObjectName("cardCaption")
        body.addWidget(heading)
        body.addWidget(detail)
        return card, body

    def _apply_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget#appRoot { background: #0b1017; color: #e8eef5; }
            QLabel#eyebrow { color: #55d6a8; font-size: 10px; font-weight: 700; letter-spacing: 1px; }
            QLabel#appTitle { color: #f7fbff; font-size: 25px; font-weight: 700; }
            QLabel#appSubtitle, QLabel#cardCaption { color: #8290a3; font-size: 11px; }
            QLabel#cardTitle { color: #eef5fb; font-size: 14px; font-weight: 700; }
            QLabel#summaryText {
                color: #aeb9c8; background: #0c131c; border: 1px solid #202c3a;
                border-radius: 8px; padding: 9px; font-size: 11px;
            }
            QLabel#statusPill {
                color: #8fe7c8; background: #10271f; border: 1px solid #225640;
                border-radius: 9px; padding: 7px 12px; font-weight: 600;
            }
            QFrame#card { background: #121a24; border: 1px solid #243142; border-radius: 12px; }
            QScrollArea#sidebar { background: transparent; }
            QScrollArea#sidebar > QWidget > QWidget { background: transparent; }
            QPushButton {
                min-height: 34px; padding: 0 12px; color: #dce5ee; background: #192432;
                border: 1px solid #2c3b4d; border-radius: 7px; font-weight: 600;
            }
            QPushButton:hover { background: #223144; border-color: #3f566e; }
            QPushButton:pressed { background: #111b27; }
            QPushButton#primaryButton {
                min-height: 38px; color: #071510; background: #55d6a8; border-color: #55d6a8;
                padding: 0 20px;
            }
            QPushButton#primaryButton:hover { background: #6de0b7; border-color: #6de0b7; }
            QPlainTextEdit#resultOutput {
                color: #d9e3ec; background: #0a1017; border: 1px solid #233142;
                border-radius: 8px; padding: 14px; selection-background-color: #245c49;
                font-family: "Cascadia Mono", "Microsoft YaHei UI"; font-size: 12px;
            }
            QSplitter#workspaceSplitter::handle { background: transparent; width: 10px; }
            QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
            QScrollBar::handle:vertical { background: #344457; border-radius: 4px; min-height: 28px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

    def print_intro(self) -> None:
        template_dir = Path(self.config.templates_dir)
        self.output.setPlainText(
            "使用步骤:\n"
            "1. 点击“框选画面”，选择整个雀魂游戏画面。\n"
            "2. 框选手牌区、摸牌、宝牌；有副露时再框选副露区。\n"
            "3. 点击“设置牌数”，设置手牌、摸牌、副露可见牌及副露组数。\n"
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
        meld_structure = ",".join(item.kind for item in layout.melds) or "等宽回退"
        opponents = ", ".join(
            f"{SEAT_LABELS.get(player.seat, player.seat)}="
            f"{','.join(meld.kind for meld in player.melds) or player.tile_count}"
            for player in layout.opponent_melds
        ) or "未设置"
        rivers = ", ".join(
            f"{SEAT_LABELS.get(player.seat, player.seat)}={len(player.tiles) or player.tile_count}"
            for player in layout.rivers
        ) or "未设置"
        return (
            "牌区比例: "
            f"手牌 x={layout.hand_region.x:.3f}, y={layout.hand_region.y:.3f}, "
            f"w={layout.hand_region.width:.3f}, h={layout.hand_region.height:.3f}, "
            f"张数={layout.hand_tile_count}; "
            f"摸牌 x={layout.draw_region.x:.3f}, y={layout.draw_region.y:.3f}, "
            f"w={layout.draw_region.width:.3f}, h={layout.draw_region.height:.3f}, "
            f"张数={layout.draw_tile_count}; "
            f"宝牌 x={layout.dora_region.x:.3f}, y={layout.dora_region.y:.3f}, "
            f"w={layout.dora_region.width:.3f}, h={layout.dora_region.height:.3f}, "
            f"实际翻开={layout.dora_tile_count}张; "
            f"副露 {meld_text}, 可见张数={layout.meld_tile_count}, "
            f"组数={layout.open_meld_count}, 结构={meld_structure}; "
            f"他家副露 {opponents}; 牌河 {rivers}"
        )

    def select_game_region(self) -> None:
        self.pending_region = "game"
        self.open_region_selector()

    def select_layout_region(self, region_kind: str) -> None:
        if self.config.game_region is None:
            QMessageBox.warning(self, "缺少画面区域", "请先点击“框选画面”。")
            return
        errors = validate_config(self.config)
        if errors:
            QMessageBox.warning(self, "配置无效", "\n".join(errors))
            return
        self.pending_region = region_kind
        self.open_region_selector()

    def select_opponent_meld_region(self) -> None:
        if self.config.game_region is None:
            QMessageBox.warning(self, "缺少画面区域", "请先点击“框选画面”。")
            return
        seat = self._choose_opponent_seat("选择要框选副露区的玩家")
        if seat is None:
            return
        self.pending_region = f"opponent_meld:{seat}"
        self.open_region_selector()

    def select_river_region(self) -> None:
        if self.config.game_region is None:
            QMessageBox.warning(self, "缺少画面区域", "请先点击“框选画面”。")
            return
        seat = self._choose_seat("选择要框选牌河区的玩家", include_self=True)
        if seat is None:
            return
        self.pending_region = f"river:{seat}"
        self.open_region_selector()

    def open_region_selector(self) -> None:
        self.hide()
        self.selector = ScreenRegionSelector()
        self.selector.region_selected.connect(self.on_region_selected)
        self.selector_completed = False
        self.selector.destroyed.connect(self.on_selector_destroyed)
        QTimer.singleShot(150, self.selector.show)

    def on_selector_destroyed(self) -> None:
        if self.pending_region == "river_slot" and not self.selector_completed:
            self.pending_river_slots = None
            self.pending_river_slot_indexes = []
            self.pending_region = None
            self.status_label.setText("已取消牌河牌槽配置，原配置未修改")
            self._restore_after_meld_config()
            return
        if self.pending_region == "meld_slot" and not self.selector_completed:
            self.pending_melds = None
            self.pending_meld_slots = []
            self.pending_region = None
            self.pending_meld_seat = "self"
            self.status_label.setText("已取消副露结构配置，原配置未修改")
            self._restore_after_meld_config()
        elif self.pending_region not in {"meld_slot", "river_slot"}:
            self.show()

    def on_region_selected(self, region: ScreenRegion) -> None:
        self.selector_completed = True
        if self.pending_region == "river_slot":
            self.on_river_slot_selected(region)
            return
        if self.pending_region == "meld_slot":
            self.on_meld_slot_selected(region)
            return
        if self.pending_region and self.pending_region.startswith("river:"):
            seat = self.pending_region.split(":", 1)[1]
            relative = self.to_relative_region(region)
            if relative is None:
                QMessageBox.warning(self, "区域无效", "请选择游戏画面内部区域。")
                return
            river = self._get_river_layout(seat, create=True)
            assert river is not None
            river.region = relative
            river.tile_count = 0
            river.tiles = []
            message = f"{SEAT_LABELS[seat]}牌河区已保存；请配置当前仍可见的牌槽"
        elif self.pending_region and self.pending_region.startswith("opponent_meld:"):
            seat = self.pending_region.split(":", 1)[1]
            relative = self.to_relative_region(region)
            if relative is None:
                QMessageBox.warning(self, "区域无效", "请选择游戏画面内部区域。")
                return
            player = self._get_opponent_meld_layout(seat, create=True)
            assert player is not None
            player.region = relative
            player.tile_count = 0
            player.melds = []
            message = f"{SEAT_LABELS[seat]}副露区已保存；请继续配置逐张结构"
        elif self.pending_region == "game":
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
        self._invalidate_recognition_state()
        self.region_label.setText(self.region_text())
        self.layout_label.setText(self.layout_text())
        self.status_label.setText(message)

    def configure_meld_structure(self) -> None:
        if self.config.game_region is None or self.config.layout.meld_region is None:
            QMessageBox.warning(self, "缺少副露区域", "请先框选游戏画面和副露区。")
            return

        self.pending_meld_seat = "self"
        current = self._format_meld_structure(self.config.layout.melds)
        text, ok = QInputDialog.getMultiLineText(
            self,
            "配置副露结构",
            "每行一组：类型 牌1方向 牌2方向 ...\n"
            "类型：chi pon minkan ankan kakan unknown\n"
            "方向：upright rotated_cw rotated_ccw，可追加 @叠放层。\n"
            "示例：pon rotated_cw upright upright",
            current,
        )
        if not ok:
            return
        try:
            melds = self._parse_meld_structure(text)
        except ValueError as exc:
            QMessageBox.warning(self, "副露结构无效", str(exc))
            return
        if not melds:
            self.config.layout.melds = []
            save_config(self.config)
            self._invalidate_recognition_state()
            self.layout_label.setText(self.layout_text())
            self.status_label.setText("已清除结构化副露，恢复等宽识别")
            return

        concealed_count = self.config.layout.hand_tile_count + self.config.layout.draw_tile_count
        expected_counts = (13 - 3 * len(melds), 14 - 3 * len(melds))
        if concealed_count not in expected_counts:
            QMessageBox.warning(
                self,
                "牌数不一致",
                f"{len(melds)} 组副露时，暗牌总数应为 "
                f"{expected_counts[0]} 或 {expected_counts[1]}，当前为 {concealed_count}。",
            )
            return

        self.pending_melds = melds
        self.pending_meld_slots = [
            (group_index, tile_index)
            for group_index, meld in enumerate(melds)
            for tile_index in range(len(meld.tiles))
        ]
        self.resume_timer_after_meld_config = self.timer.isActive()
        if self.resume_timer_after_meld_config:
            self.timer.stop()
            self.start_button.setText("开始识别")
        self._select_next_meld_slot()

    def configure_opponent_meld_structure(self) -> None:
        seat = self._choose_opponent_seat("选择要配置结构的玩家")
        if seat is None:
            return
        player = self._get_opponent_meld_layout(seat)
        if player is None or player.region is None:
            QMessageBox.warning(
                self,
                "缺少副露区域",
                f"请先框选{SEAT_LABELS[seat]}副露区。",
            )
            return
        text, ok = QInputDialog.getMultiLineText(
            self,
            f"配置{SEAT_LABELS[seat]}副露结构",
            "每行一组：类型 牌1实际方向 牌2实际方向 ...\n"
            "方向支持 upright rotated_cw rotated_ccw rotated_180，可追加 @叠放层。",
            self._format_meld_structure(player.melds),
        )
        if not ok:
            return
        try:
            melds = self._parse_meld_structure(text)
        except ValueError as exc:
            QMessageBox.warning(self, "副露结构无效", str(exc))
            return
        if not melds:
            player.melds = []
            player.tile_count = 0
            save_config(self.config)
            self._invalidate_recognition_state()
            self.layout_label.setText(self.layout_text())
            self.status_label.setText(f"已清除{SEAT_LABELS[seat]}副露结构")
            return
        self.pending_meld_seat = seat
        self.pending_melds = melds
        self.pending_meld_slots = [
            (group_index, tile_index)
            for group_index, meld in enumerate(melds)
            for tile_index in range(len(meld.tiles))
        ]
        self.resume_timer_after_meld_config = self.timer.isActive()
        if self.resume_timer_after_meld_config:
            self.timer.stop()
            self.start_button.setText("开始识别")
        self._select_next_meld_slot()

    def configure_river_slots(self) -> None:
        seat = self._choose_seat("选择要配置牌槽的玩家", include_self=True)
        if seat is None:
            return
        river = self._get_river_layout(seat)
        if river is None or river.region is None:
            QMessageBox.warning(self, "缺少牌河区域", f"请先框选{SEAT_LABELS[seat]}牌河区。")
            return
        text, ok = QInputDialog.getMultiLineText(
            self,
            f"配置{SEAT_LABELS[seat]}牌河牌槽",
            "每行一张当前仍可见的牌：行 列 实际方向 normal|riichi\n"
            "例：0 0 upright normal；立直横牌可写 1 3 rotated_cw riichi\n"
            "空位或已被鸣走的牌不要配置，行列允许不连续。",
            self._format_river_slots(river.tiles),
        )
        if not ok:
            return
        try:
            slots = self._parse_river_slots(text)
        except ValueError as exc:
            QMessageBox.warning(self, "牌河结构无效", str(exc))
            return
        if not slots:
            river.tiles = []
            river.tile_count = 0
            save_config(self.config)
            self._invalidate_recognition_state()
            self.layout_label.setText(self.layout_text())
            self.status_label.setText(f"已清除{SEAT_LABELS[seat]}牌河牌槽")
            return
        self.pending_river_seat = seat
        self.pending_river_slots = slots
        self.pending_river_slot_indexes = list(range(len(slots)))
        self.resume_timer_after_meld_config = self.timer.isActive()
        if self.resume_timer_after_meld_config:
            self.timer.stop()
            self.start_button.setText("开始识别")
        self._select_next_river_slot()

    def _select_next_river_slot(self) -> None:
        if not self.pending_river_slot_indexes:
            self._finish_river_slots()
            return
        assert self.pending_river_slots is not None
        index = self.pending_river_slot_indexes[0]
        slot = self.pending_river_slots[index]
        self.pending_region = "river_slot"
        state = "立直横牌" if slot.is_riichi else "普通牌"
        self.status_label.setText(
            f"请框选{SEAT_LABELS[self.pending_river_seat]}牌河第 {slot.row + 1} 行"
            f"第 {slot.column + 1} 列（{state}，{slot.orientation}）"
        )
        self.open_region_selector()

    def on_river_slot_selected(self, selected: ScreenRegion) -> None:
        if self.pending_river_slots is None or not self.pending_river_slot_indexes:
            return
        relative = self.to_relative_river_region(selected, self.pending_river_seat)
        if relative is None:
            QMessageBox.warning(self, "区域无效", "牌槽必须位于对应玩家的牌河区内。")
            QTimer.singleShot(100, self._select_next_river_slot)
            return
        index = self.pending_river_slot_indexes.pop(0)
        self.pending_river_slots[index].region = relative
        QTimer.singleShot(100, self._select_next_river_slot)

    def _finish_river_slots(self) -> None:
        if self.pending_river_slots is None:
            return
        river = self._get_river_layout(self.pending_river_seat)
        assert river is not None
        old_tiles = river.tiles
        old_count = river.tile_count
        river.tiles = self.pending_river_slots
        river.tile_count = len(river.tiles)
        errors = validate_config(self.config)
        if errors:
            river.tiles = old_tiles
            river.tile_count = old_count
            QMessageBox.warning(self, "牌河结构无效", "\n".join(errors))
        else:
            save_config(self.config)
            self._invalidate_recognition_state()
            self.layout_label.setText(self.layout_text())
            self.status_label.setText(f"{SEAT_LABELS[self.pending_river_seat]}牌河牌槽已保存")
        self.pending_river_slots = None
        self.pending_river_slot_indexes = []
        self.pending_region = None
        self.pending_river_seat = "self"
        self._restore_after_meld_config()

    @staticmethod
    def _parse_river_slots(text: str) -> list[RiverTileSlotConfig]:
        slots: list[RiverTileSlotConfig] = []
        positions: set[tuple[int, int]] = set()
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            parts = raw_line.split()
            if not parts:
                continue
            if len(parts) != 4:
                raise ValueError(f"第 {line_number} 行需要四项：行 列 方向 normal|riichi")
            try:
                row, column = int(parts[0]), int(parts[1])
            except ValueError as exc:
                raise ValueError(f"第 {line_number} 行的行列必须是整数") from exc
            if row < 0 or column < 0:
                raise ValueError(f"第 {line_number} 行的行列不能小于 0")
            if (row, column) in positions:
                raise ValueError(f"第 {line_number} 行的行列位置重复")
            orientation = parts[2]
            if orientation not in MELD_ORIENTATIONS:
                raise ValueError(f"第 {line_number} 行方向无效: {orientation}")
            if parts[3] not in {"normal", "riichi"}:
                raise ValueError(f"第 {line_number} 行状态必须是 normal 或 riichi")
            positions.add((row, column))
            slots.append(
                RiverTileSlotConfig(
                    region=RelativeRegion(0, 0, 1, 1),
                    orientation=orientation,
                    row=row,
                    column=column,
                    is_riichi=parts[3] == "riichi",
                )
            )
        return sorted(slots, key=lambda slot: (slot.row, slot.column))

    @staticmethod
    def _format_river_slots(slots: list[RiverTileSlotConfig]) -> str:
        return "\n".join(
            f"{slot.row} {slot.column} {slot.orientation} "
            f"{'riichi' if slot.is_riichi else 'normal'}"
            for slot in sorted(slots, key=lambda item: (item.row, item.column))
        )

    def _select_next_meld_slot(self) -> None:
        if not self.pending_meld_slots:
            self._finish_meld_structure()
            return
        group_index, tile_index = self.pending_meld_slots[0]
        assert self.pending_melds is not None
        meld = self.pending_melds[group_index]
        self.pending_region = "meld_slot"
        self.status_label.setText(
            f"请框选{SEAT_LABELS[self.pending_meld_seat]}第 "
            f"{group_index + 1}/{len(self.pending_melds)} 组（{meld.kind}）"
            f"第 {tile_index + 1}/{len(meld.tiles)} 张，方向 {meld.tiles[tile_index].orientation}"
        )
        self.open_region_selector()

    def on_meld_slot_selected(self, selected: ScreenRegion) -> None:
        if self.pending_melds is None or not self.pending_meld_slots:
            return
        relative = self.to_relative_meld_region(selected, self.pending_meld_seat)
        if relative is None:
            QMessageBox.warning(self, "区域无效", "牌槽必须位于已框选的副露区内。")
            QTimer.singleShot(100, self._select_next_meld_slot)
            return
        group_index, tile_index = self.pending_meld_slots.pop(0)
        self.pending_melds[group_index].tiles[tile_index].region = relative
        QTimer.singleShot(100, self._select_next_meld_slot)

    def _finish_meld_structure(self) -> None:
        if self.pending_melds is None:
            return
        if self.pending_meld_seat == "self":
            target = self.config.layout
            old_melds = target.melds
            old_group_count = target.open_meld_count
            old_tile_count = target.meld_tile_count
            target.melds = self.pending_melds
            target.open_meld_count = len(self.pending_melds)
            target.meld_tile_count = sum(len(meld.tiles) for meld in self.pending_melds)
        else:
            target = self._get_opponent_meld_layout(self.pending_meld_seat)
            assert target is not None
            old_melds = target.melds
            old_group_count = None
            old_tile_count = target.tile_count
            target.melds = self.pending_melds
            target.tile_count = sum(len(meld.tiles) for meld in self.pending_melds)
        errors = validate_config(self.config)
        if errors:
            target.melds = old_melds
            if self.pending_meld_seat == "self":
                target.open_meld_count = old_group_count
                target.meld_tile_count = old_tile_count
            else:
                target.tile_count = old_tile_count
            QMessageBox.warning(self, "副露结构无效", "\n".join(errors))
        else:
            save_config(self.config)
            self._invalidate_recognition_state()
            self.layout_label.setText(self.layout_text())
            self.status_label.setText(
                f"{SEAT_LABELS[self.pending_meld_seat]}副露结构和逐张牌槽已保存"
            )
        self.pending_melds = None
        self.pending_meld_slots = []
        self.pending_region = None
        self.pending_meld_seat = "self"
        self._restore_after_meld_config()

    def _restore_after_meld_config(self) -> None:
        self.show()
        if self.resume_timer_after_meld_config:
            self.timer.start(int(self.config.capture_interval_seconds * 1000))
            self.start_button.setText("停止识别")
        self.resume_timer_after_meld_config = False

    @staticmethod
    def _parse_meld_structure(text: str) -> list[MeldConfig]:
        expected_slots = {"chi": 3, "pon": 3, "minkan": 4, "ankan": 4, "kakan": 4}
        melds: list[MeldConfig] = []
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            parts = raw_line.split()
            if not parts:
                continue
            kind = parts[0].lower()
            if kind not in MELD_KINDS:
                raise ValueError(f"第 {line_number} 行副露类型无效: {kind}")
            orientations = parts[1:]
            required = expected_slots.get(kind)
            if required is not None and len(orientations) != required:
                raise ValueError(f"第 {line_number} 行 {kind} 需要 {required} 张牌的方向")
            if kind == "unknown" and not orientations:
                raise ValueError(f"第 {line_number} 行 unknown 至少需要一张牌")
            slots: list[MeldTileSlotConfig] = []
            for token in orientations:
                orientation, separator, level_text = token.partition("@")
                if orientation not in MELD_ORIENTATIONS:
                    raise ValueError(f"第 {line_number} 行方向无效: {orientation}")
                try:
                    stack_level = int(level_text) if separator else 0
                except ValueError as exc:
                    raise ValueError(f"第 {line_number} 行叠放层必须是整数: {token}") from exc
                if stack_level < 0:
                    raise ValueError(f"第 {line_number} 行叠放层不能小于 0")
                slots.append(
                    MeldTileSlotConfig(
                        region=RelativeRegion(0, 0, 1, 1),
                        orientation=orientation,
                        stack_level=stack_level,
                    )
                )
            melds.append(MeldConfig(kind=kind, tiles=slots))
        if len(melds) > 4:
            raise ValueError("副露组数不能超过 4")
        return melds

    @staticmethod
    def _format_meld_structure(melds: list[MeldConfig]) -> str:
        return "\n".join(
            " ".join(
                [meld.kind]
                + [
                    slot.orientation + (f"@{slot.stack_level}" if slot.stack_level else "")
                    for slot in meld.tiles
                ]
            )
            for meld in melds
        )

    def to_relative_meld_region(
        self, selected: ScreenRegion, seat: str = "self"
    ) -> RelativeRegion | None:
        game = self.config.game_region
        if seat == "self":
            meld = self.config.layout.meld_region
        else:
            player = self._get_opponent_meld_layout(seat)
            meld = player.region if player is not None else None
        if game is None or meld is None:
            return None
        parent = ScreenRegion(
            left=game.left + round(meld.x * game.width),
            top=game.top + round(meld.y * game.height),
            width=round(meld.width * game.width),
            height=round(meld.height * game.height),
        )
        left = max(selected.left, parent.left)
        top = max(selected.top, parent.top)
        right = min(selected.left + selected.width, parent.left + parent.width)
        bottom = min(selected.top + selected.height, parent.top + parent.height)
        if right <= left or bottom <= top:
            return None
        return RelativeRegion(
            x=(left - parent.left) / parent.width,
            y=(top - parent.top) / parent.height,
            width=(right - left) / parent.width,
            height=(bottom - top) / parent.height,
        )

    def _choose_opponent_seat(self, title: str) -> str | None:
        return self._choose_seat(title, include_self=False)

    def _choose_seat(self, title: str, include_self: bool) -> str | None:
        seats = ("self", "right", "across", "left") if include_self else ("right", "across", "left")
        labels = [SEAT_LABELS[seat] for seat in seats]
        label, ok = QInputDialog.getItem(self, title, "玩家：", labels, 0, False)
        if not ok:
            return None
        return {value: key for key, value in SEAT_LABELS.items()}[label]

    def _get_opponent_meld_layout(
        self, seat: str, create: bool = False
    ) -> PlayerMeldLayoutConfig | None:
        for player in self.config.layout.opponent_melds:
            if player.seat == seat:
                return player
        if not create:
            return None
        player = PlayerMeldLayoutConfig(seat=seat)
        self.config.layout.opponent_melds.append(player)
        return player

    def _get_river_layout(
        self, seat: str, create: bool = False
    ) -> PlayerRiverLayoutConfig | None:
        for river in self.config.layout.rivers:
            if river.seat == seat:
                return river
        if not create:
            return None
        river = PlayerRiverLayoutConfig(seat=seat)
        self.config.layout.rivers.append(river)
        return river

    def to_relative_river_region(
        self, selected: ScreenRegion, seat: str
    ) -> RelativeRegion | None:
        game = self.config.game_region
        river = self._get_river_layout(seat)
        region = river.region if river is not None else None
        if game is None or region is None:
            return None
        parent = ScreenRegion(
            left=game.left + round(region.x * game.width),
            top=game.top + round(region.y * game.height),
            width=round(region.width * game.width),
            height=round(region.height * game.height),
        )
        left = max(selected.left, parent.left)
        top = max(selected.top, parent.top)
        right = min(selected.left + selected.width, parent.left + parent.width)
        bottom = min(selected.top + selected.height, parent.top + parent.height)
        if right <= left or bottom <= top:
            return None
        return RelativeRegion(
            x=(left - parent.left) / parent.width,
            y=(top - parent.top) / parent.height,
            width=(right - left) / parent.width,
            height=(bottom - top) / parent.height,
        )

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
            "输入“手牌区张数 摸牌张数 副露区可见张数 副露组数”。\n"
            "闭门摸牌: 13 1 0 0；副露后待切: 10 0 3 1；杠后补牌: 10 1 4 1",
            text=(
                f"{layout.hand_tile_count} {layout.draw_tile_count} "
                f"{layout.meld_tile_count} {layout.open_meld_count}"
            ),
        )
        if not ok:
            return
        parts = text.split()
        if len(parts) != 4:
            QMessageBox.warning(self, "格式错误", "请按格式输入四个数字，例如 10 0 3 1。")
            return
        try:
            hand_count = int(parts[0])
            draw_count = int(parts[1])
            meld_count = int(parts[2])
            open_meld_count = int(parts[3])
        except ValueError:
            QMessageBox.warning(self, "格式错误", "牌数必须是数字。")
            return
        if not (
            1 <= hand_count <= 13
            and 0 <= draw_count <= 1
            and 0 <= meld_count <= 16
            and 0 <= open_meld_count <= 4
        ):
            QMessageBox.warning(
                self,
                "牌数无效",
                "手牌区张数应为 1-13，摸牌张数应为 0 或 1，"
                "副露区可见张数应为 0-16，副露组数应为 0-4。",
            )
            return
        concealed_count = hand_count + draw_count
        expected_counts = (13 - 3 * open_meld_count, 14 - 3 * open_meld_count)
        if concealed_count not in expected_counts:
            QMessageBox.warning(
                self,
                "牌数不一致",
                f"{open_meld_count} 组副露时，暗牌总数应为 "
                f"{expected_counts[0]} 或 {expected_counts[1]}，当前为 {concealed_count}。",
            )
            return

        if layout.melds and meld_count != sum(len(meld.tiles) for meld in layout.melds):
            QMessageBox.warning(
                self,
                "副露结构不一致",
                "副露可见张数与已配置的逐张牌槽数量不一致。请先重新配置副露结构。",
            )
            return
        old_counts = (
            layout.hand_tile_count,
            layout.draw_tile_count,
            layout.meld_tile_count,
            layout.open_meld_count,
        )
        layout.hand_tile_count = hand_count
        layout.draw_tile_count = draw_count
        layout.meld_tile_count = meld_count
        layout.open_meld_count = open_meld_count
        errors = validate_config(self.config)
        if errors:
            (
                layout.hand_tile_count,
                layout.draw_tile_count,
                layout.meld_tile_count,
                layout.open_meld_count,
            ) = old_counts
            QMessageBox.warning(self, "配置无效", "\n".join(errors))
            return
        save_config(self.config)
        self._invalidate_recognition_state()
        self.layout_label.setText(self.layout_text())
        self.status_label.setText("牌数已保存")

    def set_dora_tile_count(self) -> None:
        layout = self.config.layout
        value, ok = QInputDialog.getInt(
            self,
            "设置宝牌指示牌张数",
            "输入画面中当前实际翻开的指示牌张数（1-5）。\n"
            "开杠后每翻开一张就增加 1；框选宝牌区时只包含这些正面牌，避免牌背和空位。",
            max(1, min(5, layout.dora_tile_count)),
            1,
            5,
            1,
        )
        if not ok:
            return
        self._apply_dora_tile_count(value)

    def _apply_dora_tile_count(self, value: int) -> None:
        if not 1 <= value <= 5:
            raise ValueError("宝牌指示牌张数必须在 1 至 5 之间")
        layout = self.config.layout
        layout.dora_tile_count = value
        save_config(self.config)
        self._invalidate_recognition_state()
        self.layout_label.setText(self.layout_text())
        self.status_label.setText(f"宝牌指示牌张数已保存为 {value}")

    def set_recognition_threshold(self) -> None:
        old_value = self.config.recognition.threshold
        value, ok = QInputDialog.getDouble(
            self,
            "设置识别阈值",
            "输入 0.50 至 0.99。阈值越高越严格；请优先修正框选和模板，而不是大幅降低阈值。",
            self.config.recognition.threshold,
            0.50,
            0.99,
            3,
        )
        if not ok:
            return
        self.config.recognition.threshold = value
        errors = validate_config(self.config)
        if errors:
            self.config.recognition.threshold = old_value
            QMessageBox.warning(self, "配置无效", "\n".join(errors))
            return
        save_config(self.config)
        self._invalidate_recognition_state()
        self.status_label.setText(f"识别阈值已保存为 {value:.3f}")

    def _invalidate_recognition_state(self, clear_recognizer: bool = True) -> None:
        if clear_recognizer:
            self.recognizer = None
        self.stability.reset()
        if hasattr(self, "output"):
            self.output.setPlainText("配置或模板已变更，旧识别结果已失效，请重新识别。")

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
        self._invalidate_recognition_state()
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
        self._invalidate_recognition_state()
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
        errors = validate_config(self.config)
        if errors:
            self.status_label.setText("配置无效")
            self.output.setPlainText("\n".join(errors))
            return

        self.force_publish_current = self.sender() is not self.timer
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
            snapshot = KeyRegionSnapshot.from_frame(frame, self.config.layout)
            if self.force_publish_current or self.stability.needs_recognition(snapshot):
                raw_result = self.recognizer.recognize(frame)
                if self.force_publish_current:
                    update = self.stability.publish_success(snapshot, raw_result)
                else:
                    update = self.stability.observe_success(snapshot, raw_result)
            else:
                update = self.stability.observe_reused()
        except RecognitionError as exc:
            update = self.stability.observe_error()
            debug_text = ""
            if frame is not None and self.recognizer is not None:
                try:
                    debug_dir = self.recognizer.save_debug_tiles(frame)
                    debug_text = f"\n\n已保存本次裁剪结果到:\n{debug_dir.resolve()}"
                except Exception as debug_exc:
                    debug_text = f"\n\n保存诊断图片失败: {type(debug_exc).__name__}: {debug_exc}"
            if update.published_result is not None:
                self.status_label.setText("当前帧识别失败；下方为上次稳定结果")
                self.output.setPlainText(
                    f"当前帧错误: {exc}{debug_text}\n\n"
                    "注意：以下结果可能已过时。\n\n"
                    + self._format_result_text(update.published_result, "上次稳定识别结果")
                )
            else:
                self.status_label.setText("识别失败")
                self.output.setPlainText(str(exc) + debug_text)
            self.finish_capture_display()
            return
        except Exception as exc:  # pragma: no cover - UI safety net
            self.status_label.setText("运行错误")
            self.output.setPlainText(f"{type(exc).__name__}: {exc}")
            self.finish_capture_display()
            return

        if update.published_result is None:
            self.status_label.setText(
                f"识别成功，正在确认画面稳定性（{update.pending_count}/3）"
            )
            self.output.setPlainText("尚无稳定结果，请保持画面不变。")
        elif update.just_published:
            prefix = "单次识别完成" if self.force_publish_current else "画面已稳定，识别完成"
            self._display_result(update.published_result, prefix)
        elif update.pending_count < self.stability.required_observations:
            self.status_label.setText(
                f"检测到新画面，正在确认（{update.pending_count}/3）；下方为上次稳定结果"
            )
        elif update.reused:
            self.status_label.setText("画面未变化，沿用稳定结果（已跳过重复匹配）")
        self.finish_capture_display()

    def _display_result(self, result, status_prefix: str) -> None:
        assert self.recognizer is not None
        status = (
            f"{status_prefix}，平均置信度 {result.confidence:.3f}，"
            f"模板 {len(self.recognizer.templates.templates)} 个"
        )
        if result.meld_error:
            status += "；副露部分未知但牌效分析已继续"
        self.status_label.setText(status)
        self.output.setPlainText(self._format_result_text(result))

    def _format_result_text(self, result, title: str = "识别结果") -> str:
        tiles = list(result.hand)
        if result.draw:
            tiles.append(result.draw)

        try:
            analysis = analyze_hand(
                tiles,
                open_meld_count=self.config.layout.open_meld_count,
                visible_tiles=collect_visible_tiles(result),
            )
            analysis_text = self.format_analysis(analysis)
        except ValueError as exc:
            analysis_text = f"牌效分析不可用: {exc}"

        meld_text = " ".join(result.meld_tiles) if result.meld_tiles else "-"
        if result.meld_error:
            meld_text += f"（部分未知：{result.meld_error}）"
        opponent_lines = []
        opponents_by_seat = {player.seat: player for player in result.opponent_melds}
        for seat in ("right", "across", "left"):
            player = opponents_by_seat.get(seat)
            if player is None:
                continue
            text = " ".join(player.meld_tiles) if player.meld_tiles else "-"
            if player.error:
                text += f"（部分未知：{player.error}）"
            opponent_lines.append(f"{SEAT_LABELS[seat]}副露: {text}")
        opponent_text = ("\n".join(opponent_lines) + "\n") if opponent_lines else ""
        river_lines = []
        rivers_by_seat = {player.seat: player for player in result.rivers}
        for seat in ("self", "right", "across", "left"):
            river = rivers_by_seat.get(seat)
            if river is None:
                continue
            slots = []
            for tile in sorted(river.tiles, key=lambda item: (item.row, item.column)):
                name = tile.name or "?"
                riichi = "(立直)" if tile.is_riichi else ""
                slots.append(f"r{tile.row + 1}c{tile.column + 1}={name}{riichi}")
            text = " ".join(slots) if slots else "-"
            if river.error:
                text += f"（部分未知：{river.error}）"
            river_lines.append(f"{SEAT_LABELS[seat]}牌河: {text}")
        river_text = ("\n".join(river_lines) + "\n") if river_lines else ""
        return (
            f"{title}\n"
            f"手牌: {' '.join(result.hand) if result.hand else '-'}\n"
            f"摸牌: {result.draw or '-'}\n"
            f"宝牌: {' '.join(result.dora_indicators) if result.dora_indicators else '-'}\n"
            f"副露: {meld_text}\n"
            f"{opponent_text}"
            f"{river_text}"
            f"置信度: {result.confidence:.3f}\n\n"
            "牌效分析（已知可见牌修正）\n"
            f"{analysis_text}"
        )

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
