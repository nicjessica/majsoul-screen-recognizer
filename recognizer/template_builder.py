from __future__ import annotations

from pathlib import Path

from PIL import Image

from recognizer.config import LayoutConfig, RelativeRegion


def build_templates_from_screenshot(
    screenshot_path: str | Path,
    tile_names: list[str],
    output_dir: str | Path,
    layout: LayoutConfig,
    include_draw: bool | None = None,
) -> list[Path]:
    image = Image.open(screenshot_path).convert("RGB")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    draw_count = layout.draw_tile_count if include_draw is None else int(include_draw)
    expected = layout.hand_tile_count + draw_count
    if len(tile_names) != expected:
        raise ValueError(f"需要 {expected} 个牌名，当前 {len(tile_names)} 个")

    saved: list[Path] = []
    hand_region = crop_relative_pil(image, layout.hand_region)
    hand_width, hand_height = hand_region.size
    tile_width = hand_width / layout.hand_tile_count

    for index in range(layout.hand_tile_count):
        left = round(index * tile_width)
        right = round((index + 1) * tile_width)
        tile_image = hand_region.crop((left, 0, right, hand_height))
        saved.append(save_tile(tile_image, tile_names[index], output))

    if draw_count:
        draw_region = crop_relative_pil(image, layout.draw_region)
        saved.append(save_tile(draw_region, tile_names[-1], output))

    return saved


def crop_relative_pil(image: Image.Image, region: RelativeRegion) -> Image.Image:
    width, height = image.size
    left = max(0, min(width - 1, round(region.x * width)))
    top = max(0, min(height - 1, round(region.y * height)))
    right = max(left + 1, min(width, round((region.x + region.width) * width)))
    bottom = max(top + 1, min(height, round((region.y + region.height) * height)))
    return image.crop((left, top, right, bottom))


def save_tile(image: Image.Image, tile_name: str, output_dir: Path) -> Path:
    safe_name = tile_name.strip()
    if not safe_name:
        raise ValueError("牌名不能为空")
    path = output_dir / f"{safe_name}.png"
    image.save(path)
    return path
