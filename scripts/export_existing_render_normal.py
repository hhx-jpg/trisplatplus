"""Extract already-rendered normal panels from a compact contact sheet."""

from pathlib import Path
import argparse

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sheet", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    image = Image.open(args.sheet).convert("RGB")
    width, height = image.size
    panel_width = width // 3
    # The source sheet is generated at 448x224 panels and uniformly resized;
    # derive the row geometry from its known four-target layout.
    header = round(height * 28 / 1020)
    row_height = height * 248 / 1020
    panel_height = round(height * 224 / 1020)
    args.out.mkdir(parents=True, exist_ok=True)

    views = []
    for index in range(4):
        top = round(header + index * row_height)
        bottom = min(height, top + panel_height)
        normal = image.crop((panel_width, top, panel_width * 2, bottom))
        views.append(normal)
        normal.save(args.out / f"render_normal_target{index}.jpg", quality=92, optimize=True)

    gap = 4
    title_height = 24
    grid = Image.new("RGB", (panel_width, title_height + sum(v.height for v in views) + gap * 3), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((6, 6), "CUDA rendered face normal (step 7000)", fill="black")
    y = title_height
    for normal in views:
        grid.paste(normal, (0, y))
        y += normal.height + gap
    grid.save(args.out / "render_normal_step_7000.jpg", quality=90, optimize=True, progressive=True)


if __name__ == "__main__":
    main()
