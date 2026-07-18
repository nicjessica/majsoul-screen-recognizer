from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from recognizer.geometry import ScreenRegion


CONFIG_PATH = Path("data/config.json")


@dataclass
class RelativeRegion:
    x: float
    y: float
    width: float
    height: float


MELD_KINDS = {"unknown", "chi", "pon", "minkan", "ankan", "kakan"}
PLAYER_SEATS = ("self", "right", "across", "left")
MELD_ORIENTATIONS = {"upright", "rotated_cw", "rotated_ccw", "rotated_180"}


@dataclass
class MeldTileSlotConfig:
    region: RelativeRegion
    orientation: str = "upright"
    stack_level: int = 0


@dataclass
class MeldConfig:
    kind: str = "unknown"
    tiles: list[MeldTileSlotConfig] = field(default_factory=list)


@dataclass
class PlayerMeldLayoutConfig:
    seat: str
    region: RelativeRegion | None = None
    tile_count: int = 0
    melds: list[MeldConfig] = field(default_factory=list)


@dataclass
class RiverTileSlotConfig:
    region: RelativeRegion
    orientation: str = "upright"
    row: int = 0
    column: int = 0
    is_riichi: bool = False


@dataclass
class PlayerRiverLayoutConfig:
    seat: str
    region: RelativeRegion | None = None
    tile_count: int = 0
    tiles: list[RiverTileSlotConfig] = field(default_factory=list)


@dataclass
class PlayerScoreLayoutConfig:
    seat: str
    region: RelativeRegion
    orientation: str = "upright"


@dataclass
class TableStateLayoutConfig:
    round_region: RelativeRegion | None = None
    self_wind_region: RelativeRegion | None = None
    scores: list[PlayerScoreLayoutConfig] = field(default_factory=list)


@dataclass
class LayoutConfig:
    hand_region: RelativeRegion
    draw_region: RelativeRegion
    dora_region: RelativeRegion
    meld_region: RelativeRegion | None = None
    hand_tile_count: int = 13
    draw_tile_count: int = 1
    dora_tile_count: int = 1
    meld_tile_count: int = 0
    open_meld_count: int = 0
    melds: list[MeldConfig] = field(default_factory=list)
    opponent_melds: list[PlayerMeldLayoutConfig] = field(default_factory=list)
    rivers: list[PlayerRiverLayoutConfig] = field(default_factory=list)
    table_state: TableStateLayoutConfig = field(default_factory=TableStateLayoutConfig)


@dataclass
class RecognitionConfig:
    threshold: float = 0.78
    auto_detect_tile_state: bool = True


@dataclass
class OverlayConfig:
    position_x: float = 0.016
    position_y: float = 0.022


@dataclass
class AppConfig:
    game_region: ScreenRegion | None
    layout: LayoutConfig
    recognition: RecognitionConfig
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    templates_dir: str = "data/templates"
    table_templates_dir: str = "data/table_templates"
    capture_interval_seconds: float = 0.75


def default_config() -> AppConfig:
    return AppConfig(
        game_region=None,
        layout=LayoutConfig(
            hand_region=RelativeRegion(x=0.115, y=0.852, width=0.650, height=0.126),
            draw_region=RelativeRegion(x=0.773, y=0.852, width=0.055, height=0.126),
            dora_region=RelativeRegion(x=0.506, y=0.088, width=0.052, height=0.064),
        ),
        recognition=RecognitionConfig(),
    )


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    if not path.exists():
        config = default_config()
        save_config(config, path)
        return config

    data = json.loads(path.read_text(encoding="utf-8"))
    return _config_from_dict(data)


def save_config(config: AppConfig, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(config.templates_dir).mkdir(parents=True, exist_ok=True)
    Path(config.table_templates_dir).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_config_to_dict(config), indent=2, ensure_ascii=False), encoding="utf-8")


def _config_to_dict(config: AppConfig) -> dict[str, Any]:
    data = asdict(config)
    return data


def _config_from_dict(data: dict[str, Any]) -> AppConfig:
    region_data = data.get("game_region")
    region = ScreenRegion(**region_data) if region_data else None

    layout_data = data.get("layout", {})
    default = default_config()
    default_layout = default.layout
    hand_tile_count = int(layout_data.get("hand_tile_count", default_layout.hand_tile_count))
    draw_tile_count = int(layout_data.get("draw_tile_count", default_layout.draw_tile_count))
    meld_tile_count = int(layout_data.get("meld_tile_count", default_layout.meld_tile_count))
    open_meld_count = layout_data.get("open_meld_count")
    if open_meld_count is None:
        open_meld_count = _infer_open_meld_count(hand_tile_count, draw_tile_count)

    layout = LayoutConfig(
        hand_region=RelativeRegion(**layout_data.get("hand_region", asdict(default_layout.hand_region))),
        draw_region=RelativeRegion(**layout_data.get("draw_region", asdict(default_layout.draw_region))),
        dora_region=RelativeRegion(**layout_data.get("dora_region", asdict(default_layout.dora_region))),
        meld_region=(
            RelativeRegion(**layout_data["meld_region"])
            if layout_data.get("meld_region")
            else default_layout.meld_region
        ),
        hand_tile_count=hand_tile_count,
        draw_tile_count=draw_tile_count,
        dora_tile_count=int(layout_data.get("dora_tile_count", default_layout.dora_tile_count)),
        meld_tile_count=meld_tile_count,
        open_meld_count=int(open_meld_count),
        melds=[_meld_from_dict(item) for item in layout_data.get("melds", [])],
        opponent_melds=[
            _player_meld_layout_from_dict(item)
            for item in layout_data.get("opponent_melds", [])
        ],
        rivers=[_player_river_layout_from_dict(item) for item in layout_data.get("rivers", [])],
        table_state=_table_state_layout_from_dict(layout_data.get("table_state", {})),
    )

    recognition_data = data.get("recognition", {})
    recognition = RecognitionConfig(
        threshold=float(recognition_data.get("threshold", default.recognition.threshold)),
        auto_detect_tile_state=bool(
            recognition_data.get(
                "auto_detect_tile_state",
                default.recognition.auto_detect_tile_state,
            )
        ),
    )
    overlay_data = data.get("overlay", {})
    overlay = OverlayConfig(
        position_x=float(overlay_data.get("position_x", default.overlay.position_x)),
        position_y=float(overlay_data.get("position_y", default.overlay.position_y)),
    )

    return AppConfig(
        game_region=region,
        layout=layout,
        recognition=recognition,
        overlay=overlay,
        templates_dir=str(data.get("templates_dir", default.templates_dir)),
        table_templates_dir=str(data.get("table_templates_dir", default.table_templates_dir)),
        capture_interval_seconds=float(
            data.get("capture_interval_seconds", default.capture_interval_seconds)
        ),
    )


def _infer_open_meld_count(hand_tile_count: int, draw_tile_count: int) -> int:
    concealed_count = hand_tile_count + draw_tile_count
    candidates = [
        count
        for count in range(5)
        if concealed_count in (13 - 3 * count, 14 - 3 * count)
    ]
    return candidates[0] if candidates else 0


def _meld_from_dict(data: dict[str, Any]) -> MeldConfig:
    kind = str(data.get("kind", "unknown"))
    if kind not in MELD_KINDS:
        kind = "unknown"

    tiles: list[MeldTileSlotConfig] = []
    for tile_data in data.get("tiles", []):
        orientation = str(tile_data.get("orientation", "upright"))
        if orientation not in MELD_ORIENTATIONS:
            orientation = "upright"
        tiles.append(
            MeldTileSlotConfig(
                region=RelativeRegion(**tile_data["region"]),
                orientation=orientation,
                stack_level=int(tile_data.get("stack_level", 0)),
            )
        )
    return MeldConfig(kind=kind, tiles=tiles)


def _player_meld_layout_from_dict(data: dict[str, Any]) -> PlayerMeldLayoutConfig:
    region_data = data.get("region")
    return PlayerMeldLayoutConfig(
        seat=str(data.get("seat", "")),
        region=RelativeRegion(**region_data) if region_data else None,
        tile_count=int(data.get("tile_count", 0)),
        melds=[_meld_from_dict(item) for item in data.get("melds", [])],
    )


def _player_river_layout_from_dict(data: dict[str, Any]) -> PlayerRiverLayoutConfig:
    region_data = data.get("region")
    tiles = []
    for tile_data in data.get("tiles", []):
        orientation = str(tile_data.get("orientation", "upright"))
        if orientation not in MELD_ORIENTATIONS:
            orientation = "upright"
        tiles.append(RiverTileSlotConfig(
            region=RelativeRegion(**tile_data["region"]),
            orientation=orientation,
            row=int(tile_data.get("row", 0)),
            column=int(tile_data.get("column", 0)),
            is_riichi=bool(tile_data.get("is_riichi", False)),
        ))
    return PlayerRiverLayoutConfig(
        seat=str(data.get("seat", "")),
        region=RelativeRegion(**region_data) if region_data else None,
        tile_count=int(data.get("tile_count", 0)),
        tiles=tiles,
    )


def _table_state_layout_from_dict(data: dict[str, Any]) -> TableStateLayoutConfig:
    round_data = data.get("round_region")
    wind_data = data.get("self_wind_region")
    scores = []
    for item in data.get("scores", []):
        orientation = str(item.get("orientation", "upright"))
        if orientation not in MELD_ORIENTATIONS:
            orientation = "upright"
        scores.append(PlayerScoreLayoutConfig(
            seat=str(item.get("seat", "")),
            region=RelativeRegion(**item["region"]),
            orientation=orientation,
        ))
    return TableStateLayoutConfig(
        round_region=RelativeRegion(**round_data) if round_data else None,
        self_wind_region=RelativeRegion(**wind_data) if wind_data else None,
        scores=scores,
    )


def validate_config(config: AppConfig) -> list[str]:
    """返回供 UI 展示的配置错误；加载旧配置时不会自动调用。"""
    errors: list[str] = []
    layout = config.layout

    if not 0 <= layout.open_meld_count <= 4:
        errors.append("副露组数必须在 0 至 4 之间。")
    if not 0.0 <= config.recognition.threshold <= 1.0:
        errors.append("识别置信度阈值必须在 0 至 1 之间。")
    if not 0.0 <= config.overlay.position_x <= 1.0:
        errors.append("建议浮层水平位置必须在 0 至 1 之间。")
    if not 0.0 <= config.overlay.position_y <= 1.0:
        errors.append("建议浮层垂直位置必须在 0 至 1 之间。")

    table_state = layout.table_state
    if table_state.round_region is not None and not _is_valid_relative_region(table_state.round_region):
        errors.append("Round region is outside 0..1.")
    if table_state.self_wind_region is not None and not _is_valid_relative_region(table_state.self_wind_region):
        errors.append("Self wind region is outside 0..1.")
    seen_score_seats: set[str] = set()
    for score in table_state.scores:
        if score.seat not in PLAYER_SEATS:
            errors.append(f"Score seat must be self/right/across/left: {score.seat}")
        elif score.seat in seen_score_seats:
            errors.append(f"Score seat is duplicated: {score.seat}")
        seen_score_seats.add(score.seat)
        if not _is_valid_relative_region(score.region):
            errors.append(f"Score {score.seat} region is outside 0..1.")
        if score.orientation not in MELD_ORIENTATIONS:
            errors.append(f"Score {score.seat} has invalid orientation.")

    seen_opponent_seats: set[str] = set()
    for player in layout.opponent_melds:
        if player.seat not in PLAYER_SEATS[1:]:
            errors.append(f"Opponent meld seat must be right/across/left: {player.seat}")
        elif player.seat in seen_opponent_seats:
            errors.append(f"Opponent meld seat is duplicated: {player.seat}")
        seen_opponent_seats.add(player.seat)
        if player.tile_count < 0:
            errors.append(f"Opponent {player.seat} meld tile count cannot be negative.")
        if (player.tile_count > 0 or player.melds) and player.region is None:
            errors.append(f"Opponent {player.seat} meld region is required.")
        if player.region is not None and not _is_valid_relative_region(player.region):
            errors.append(f"Opponent {player.seat} meld region is outside 0..1.")
        if player.melds and player.tile_count not in (0, sum(len(meld.tiles) for meld in player.melds)):
            errors.append(f"Opponent {player.seat} meld tile count does not match structured slots.")
        errors.extend(_validate_meld_groups(player.melds, f"Opponent {player.seat}"))

    seen_river_seats: set[str] = set()
    for river in layout.rivers:
        if river.seat not in PLAYER_SEATS:
            errors.append(f"River seat must be self/right/across/left: {river.seat}")
        elif river.seat in seen_river_seats:
            errors.append(f"River seat is duplicated: {river.seat}")
        seen_river_seats.add(river.seat)
        if river.tile_count < 0:
            errors.append(f"River {river.seat} tile count cannot be negative.")
        if (river.tile_count > 0 or river.tiles) and river.region is None:
            errors.append(f"River {river.seat} region is required.")
        if river.region is not None and not _is_valid_relative_region(river.region):
            errors.append(f"River {river.seat} region is outside 0..1.")
        if river.tiles and river.tile_count not in (0, len(river.tiles)):
            errors.append(f"River {river.seat} tile count does not match explicit slots.")
        seen_positions: set[tuple[int, int]] = set()
        seen_regions: set[tuple[float, float, float, float]] = set()
        for index, tile in enumerate(river.tiles, start=1):
            if tile.orientation not in MELD_ORIENTATIONS:
                errors.append(f"River {river.seat} tile {index} has invalid orientation.")
            if tile.row < 0 or tile.column < 0:
                errors.append(f"River {river.seat} tile {index} row/column cannot be negative.")
            if not _is_valid_relative_region(tile.region):
                errors.append(f"River {river.seat} tile {index} region is outside 0..1.")
            position = (tile.row, tile.column)
            if position in seen_positions:
                errors.append(
                    f"River {river.seat} tile {index} duplicates row/column {position}."
                )
            seen_positions.add(position)
            region_key = (
                tile.region.x,
                tile.region.y,
                tile.region.width,
                tile.region.height,
            )
            if region_key in seen_regions:
                errors.append(
                    f"River {river.seat} tile {index} duplicates an explicit region."
                )
            seen_regions.add(region_key)

    if not layout.melds:
        return errors

    if layout.meld_region is None:
        errors.append("已配置结构化副露时必须先框选副露区。")
    if len(layout.melds) != layout.open_meld_count:
        errors.append(
            f"结构化副露共有 {len(layout.melds)} 组，与副露组数 "
            f"{layout.open_meld_count} 不一致。"
        )

    expected_slot_counts = {
        "chi": 3,
        "pon": 3,
        "minkan": 4,
        "ankan": 4,
        "kakan": 4,
    }
    for meld_index, meld in enumerate(layout.melds, start=1):
        if meld.kind not in MELD_KINDS:
            errors.append(f"第 {meld_index} 组副露类型无效: {meld.kind}。")
        expected = expected_slot_counts.get(meld.kind)
        if expected is not None and len(meld.tiles) != expected:
            errors.append(
                f"第 {meld_index} 组 {meld.kind} 应配置 {expected} 张可见牌，"
                f"当前为 {len(meld.tiles)} 张。"
            )
        elif meld.kind == "unknown" and not meld.tiles:
            errors.append(f"第 {meld_index} 组 unknown 副露至少需要 1 张可见牌。")

        for tile_index, tile in enumerate(meld.tiles, start=1):
            if tile.orientation not in MELD_ORIENTATIONS:
                errors.append(
                    f"第 {meld_index} 组第 {tile_index} 张牌方向无效: "
                    f"{tile.orientation}。"
                )
            if tile.stack_level < 0:
                errors.append(
                    f"第 {meld_index} 组第 {tile_index} 张牌叠放层级不能小于 0。"
                )
            region = tile.region
            if not (
                0.0 <= region.x < 1.0
                and 0.0 <= region.y < 1.0
                and region.width > 0.0
                and region.height > 0.0
                and region.x + region.width <= 1.0
                and region.y + region.height <= 1.0
            ):
                errors.append(
                    f"第 {meld_index} 组第 {tile_index} 张牌区域必须位于副露区的 "
                    "0 至 1 相对坐标范围内，且宽高必须大于 0。"
                )

    return errors


def _validate_meld_groups(melds: list[MeldConfig], label: str) -> list[str]:
    errors: list[str] = []
    expected_slot_counts = {"chi": 3, "pon": 3, "minkan": 4, "ankan": 4, "kakan": 4}
    for meld_index, meld in enumerate(melds, start=1):
        expected = expected_slot_counts.get(meld.kind)
        if meld.kind not in MELD_KINDS:
            errors.append(f"{label} meld {meld_index} has invalid kind: {meld.kind}")
        if expected is not None and len(meld.tiles) != expected:
            errors.append(f"{label} meld {meld_index} {meld.kind} requires {expected} tile slots.")
        elif meld.kind == "unknown" and not meld.tiles:
            errors.append(f"{label} meld {meld_index} requires at least one tile slot.")
        for tile_index, tile in enumerate(meld.tiles, start=1):
            if tile.orientation not in MELD_ORIENTATIONS:
                errors.append(f"{label} meld {meld_index} tile {tile_index} has invalid orientation.")
            if tile.stack_level < 0:
                errors.append(f"{label} meld {meld_index} tile {tile_index} stack level cannot be negative.")
            region = tile.region
            if not (
                0.0 <= region.x < 1.0 and 0.0 <= region.y < 1.0
                and region.width > 0.0 and region.height > 0.0
                and region.x + region.width <= 1.0 and region.y + region.height <= 1.0
            ):
                errors.append(f"{label} meld {meld_index} tile {tile_index} region is outside 0..1.")
    return errors


def _is_valid_relative_region(region: RelativeRegion) -> bool:
    return (
        0.0 <= region.x < 1.0
        and 0.0 <= region.y < 1.0
        and region.width > 0.0
        and region.height > 0.0
        and region.x + region.width <= 1.0
        and region.y + region.height <= 1.0
    )
