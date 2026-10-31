#!/usr/bin/env python3
"""Apply Oz's 10%-thickness / 25%-opacity outer-outline treatment.

The transform is deterministic and frame-local. It reconstructs only the
published outer-stroke band, preserves internal artwork, and then renders the
new subpixel outline. Empty atlas cells and alpha geometry are unchanged.
"""

from __future__ import annotations

import json
import io
import subprocess
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets" / "oz-churu.png"
CONTACT = ROOT / "previews" / "contact-sheet.png"
HOVER_GIF = ROOT / "previews" / "churu.gif"
REPORT = ROOT / "docs" / "outline-10-opacity-25-report.json"
VALIDATION = ROOT / "docs" / "churu-validation.json"
ALL_STATES_GIF = ROOT / "previews" / "outline-10-opacity-25-all-states.gif"
IDLE_GIF = ROOT / "previews" / "outline-10-opacity-25-idle.gif"
LOOK_GIF = ROOT / "previews" / "outline-10-opacity-25-look.gif"
RUNNING_RIGHT_GIF = ROOT / "previews" / "outline-10-opacity-25-running-right.gif"
RUNNING_LEFT_GIF = ROOT / "previews" / "outline-10-opacity-25-running-left.gif"
SOURCE_COMMIT = "4a1936513c5a5210e304d622e4150d209b16e7a9"
CHURU_REPAIRED_STRIP = ROOT / "sources" / "churu-repaired-strip.png"

CELL_W, CELL_H = 192, 208
ROWS, COLS = 11, 8
FRAME_COUNTS = (6, 8, 8, 4, 5, 8, 6, 6, 6, 8, 8)
ROW_NAMES = (
    "idle",
    "running-right",
    "running-left",
    "waving",
    "jumping",
    "failed",
    "waiting",
    "running",
    "review",
    "look 000-157.5",
    "look 180-337.5",
)
TARGET_WIDTH = 0.8  # 10% of the calibrated eight-pixel antialiased stroke
TARGET_OPACITY = 0.25


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def smooth_outline_free_base(
    frame: np.ndarray,
    outline: np.ndarray,
    inside: np.ndarray,
    edge_distance: np.ndarray,
    band_width: float,
) -> np.ndarray:
    result = frame.copy()
    safe = inside & (edge_distance > band_width) & ~outline
    if not np.any(safe):
        return frame.copy()

    _, nearest = ndi.distance_transform_edt(~safe, return_indices=True)
    working = frame[:, :, :3].astype(np.float32).copy()
    for channel in range(3):
        source = frame[:, :, channel]
        working[:, :, channel][outline] = source[nearest[0][outline], nearest[1][outline]]

    support = inside.astype(np.float32)
    kernel = np.array(
        [[1.0, 2.0, 1.0], [2.0, 0.0, 2.0], [1.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    denominator = ndi.convolve(support, kernel, mode="constant", cval=0.0)
    denominator = np.maximum(denominator, 1.0)
    for _ in range(120):
        for channel in range(3):
            numerator = ndi.convolve(
                working[:, :, channel] * support,
                kernel,
                mode="constant",
                cval=0.0,
            )
            relaxed = numerator / denominator
            working[:, :, channel][outline] = relaxed[outline]

    result[:, :, :3] = np.clip(np.rint(working), 0, 255).astype(np.uint8)
    result[:, :, 3] = frame[:, :, 3]
    return result


def protect_inward_detail_lines(
    frame: np.ndarray,
    outline_band: np.ndarray,
    edge_distance: np.ndarray,
) -> np.ndarray:
    """Protect dark strokes that cross inward from the silhouette (e.g. toes)."""
    rgb = frame[:, :, :3].astype(np.float32)
    luminance = rgb.mean(axis=2)
    gy, gx = np.gradient(edge_distance.astype(np.float32))
    norm = np.maximum(np.hypot(gx, gy), 1e-4)
    nx, ny = gx / norm, gy / norm
    tx, ty = -ny, nx
    yy, xx = np.indices(edge_distance.shape, dtype=np.float32)

    # A normal-crossing internal stroke has bright pixels on both tangent
    # sides; the outer outline stays dark along its tangent.
    side_a = ndi.map_coordinates(luminance, [yy + ty * 2.5, xx + tx * 2.5], order=1, mode="nearest")
    side_b = ndi.map_coordinates(luminance, [yy - ty * 2.5, xx - tx * 2.5], order=1, mode="nearest")
    protected = (
        outline_band
        & (edge_distance > 1.5)
        & (luminance < 100.0)
        & (side_a > 155.0)
        & (side_b > 155.0)
    )
    return ndi.binary_dilation(protected, iterations=1) & outline_band


def process_frame(frame: np.ndarray, band_width: float = 5.0) -> tuple[np.ndarray, dict[str, int]]:
    alpha = frame[:, :, 3]
    inside = alpha > 0
    if not np.any(inside):
        return frame.copy(), {"outline_pixels": 0, "protected_detail_pixels": 0}

    labels, count = ndi.label(inside, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())
    edge_distance = ndi.distance_transform_edt(inside)
    eligible = np.zeros_like(inside)
    for component in range(1, count + 1):
        component_mask = labels == component
        if sizes[component] >= 80 and float(edge_distance[component_mask].max()) > band_width:
            eligible |= component_mask

    outline_band = eligible & (edge_distance <= band_width)
    protected = protect_inward_detail_lines(frame, outline_band, edge_distance)
    outline = outline_band & ~protected
    base = smooth_outline_free_base(frame, outline, inside, edge_distance, band_width)

    rgb = frame[:, :, :3].astype(np.float32)
    luminance = rgb.mean(axis=2)
    dark_source = outline_band & (luminance < 80.0)
    line_color = (
        np.median(rgb[dark_source], axis=0)
        if np.any(dark_source)
        else np.array([42.0, 36.0, 32.0], dtype=np.float32)
    )
    coverage = np.clip(TARGET_WIDTH + 1.0 - edge_distance, 0.0, 1.0)
    coverage *= outline.astype(np.float32) * TARGET_OPACITY
    weight = coverage[:, :, None]
    out = base.copy()
    blended = np.rint(out[:, :, :3].astype(np.float32) * (1.0 - weight) + line_color * weight)
    out[:, :, :3] = np.clip(blended, 0, 255).astype(np.uint8)
    out[:, :, 3] = frame[:, :, 3]

    if not np.array_equal(out[~outline], frame[~outline]):
        raise AssertionError("Protected pixels changed outside the isolated outer-outline band")
    return out, {
        "outline_pixels": int(outline.sum()),
        "protected_detail_pixels": int(protected.sum()),
        "removed_source_band_width": band_width,
    }


def registered_churu_frames() -> list[np.ndarray]:
    """Extract five generated poses and register them to the pet cell grid."""
    strip = np.asarray(Image.open(CHURU_REPAIRED_STRIP).convert("RGBA"))
    alpha = strip[:, :, 3]
    labels, count = ndi.label(alpha > 8, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())
    components: list[tuple[int, tuple[int, int, int, int]]] = []
    for component in range(1, count + 1):
        if sizes[component] < 10_000:
            continue
        ys, xs = np.where(labels == component)
        components.append((component, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)))
    components.sort(key=lambda item: item[1][0])
    if len(components) != 5:
        raise RuntimeError(f"Expected five generated churu poses, found {len(components)}")

    crops: list[Image.Image] = []
    bounds: list[tuple[int, int]] = []
    for component, (x0, y0, x1, y1) in components:
        keep = ndi.binary_dilation(labels == component, iterations=2)
        isolated = strip.copy()
        isolated[~keep] = 0
        pad = 4
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(strip.shape[1], x1 + pad), min(strip.shape[0], y1 + pad)
        crop = Image.fromarray(isolated[y0:y1, x0:x1], "RGBA")
        crops.append(crop)
        bounds.append(crop.size)

    max_width = max(width for width, _ in bounds)
    max_height = max(height for _, height in bounds)
    # The first generated concept already has the preferred round proportions.
    # Register by the pet's vertical size (matching the ~190px published pose
    # height) and allow its naturally slightly wider seated silhouette.
    shared_scale = min(150.0 / max_width, 190.0 / max_height)
    frames: list[np.ndarray] = []
    for crop in crops:
        resized = crop.resize(
            (max(1, round(crop.width * shared_scale)), max(1, round(crop.height * shared_scale))),
            Image.Resampling.LANCZOS,
        )
        frame = Image.new("RGBA", (CELL_W, CELL_H), (0, 0, 0, 0))
        x = (CELL_W - resized.width) // 2
        y = 202 - resized.height
        frame.alpha_composite(resized, (x, y))
        frames.append(np.asarray(frame).copy())
    return frames


def harmonize_churu_palette(frame: np.ndarray) -> np.ndarray:
    """Nudge generated neutral fur/tabby tones onto the idle palette."""
    result = frame.copy()
    rgb = result[:, :, :3].astype(np.int16)
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    neutral_fur = (
        (result[:, :, 3] > 0)
        & ((maximum - minimum) < 45)
        & (maximum > 50)
        & (maximum < 245)
    )
    shifted = rgb + np.array([4, 4, 3], dtype=np.int16)
    rgb[neutral_fur] = shifted[neutral_fur]
    result[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return result


def checker(width: int, height: int, cell: int = 12) -> Image.Image:
    yy, xx = np.indices((height, width))
    pattern = ((xx // cell + yy // cell) % 2).astype(bool)
    arr = np.empty((height, width, 3), dtype=np.uint8)
    arr[pattern] = (255, 255, 255)
    arr[~pattern] = (236, 236, 236)
    return Image.fromarray(arr, "RGB")


def render_contact_sheet(atlas: Image.Image) -> None:
    header_h = 22
    canvas = Image.new("RGB", (COLS * CELL_W, ROWS * (CELL_H + header_h)), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = font(12)
    small_font = font(10)
    for row, (name, count) in enumerate(zip(ROW_NAMES, FRAME_COUNTS)):
        y = row * (CELL_H + header_h)
        draw.rectangle((0, y, canvas.width, y + header_h - 1), fill=(13, 18, 16))
        draw.text((4, y + 4), f"row {row}: {name}", fill="white", font=label_font)
        right = f"{count} frames"
        box = draw.textbbox((0, 0), right, font=label_font)
        draw.text((canvas.width - (box[2] - box[0]) - 8, y + 4), right, fill="white", font=label_font)
        for col in range(COLS):
            x = col * CELL_W
            bg = checker(CELL_W, CELL_H)
            frame = atlas.crop((x, row * CELL_H, x + CELL_W, (row + 1) * CELL_H))
            bg.paste(frame, (0, 0), frame)
            canvas.paste(bg, (x, y + header_h))
            color = (54, 188, 116) if col < count else (213, 73, 93)
            draw.rectangle((x, y + header_h, x + CELL_W - 1, y + header_h + CELL_H - 1), outline=color, width=1)
            draw.text((x + 3, y + header_h + 3), str(col), fill=(55, 55, 55), font=small_font)
    canvas.save(CONTACT, quality=96)


def render_hover_gif(atlas: Image.Image) -> None:
    frames: list[Image.Image] = []
    row = 4
    for col in range(FRAME_COUNTS[row]):
        frame = atlas.crop((col * CELL_W, row * CELL_H, (col + 1) * CELL_W, (row + 1) * CELL_H))
        bg = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        bg.alpha_composite(frame)
        frames.append(bg)
    frames[0].save(
        HOVER_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=[280, 140, 140, 140, 280],
        loop=0,
        disposal=2,
        transparency=0,
        optimize=False,
    )


def labeled_preview_frame(frame: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGBA", (240, 238), (247, 244, 238, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), label, fill=(46, 42, 40, 255), font=font(15))
    canvas.alpha_composite(frame, ((240 - CELL_W) // 2, 28))
    return canvas


def render_motion_previews(atlas: Image.Image) -> None:
    all_frames: list[Image.Image] = []
    all_durations: list[int] = []
    for row in range(9):
        for col in range(FRAME_COUNTS[row]):
            frame = atlas.crop((col * CELL_W, row * CELL_H, (col + 1) * CELL_W, (row + 1) * CELL_H))
            all_frames.append(labeled_preview_frame(frame, ROW_NAMES[row]))
            all_durations.append(140)
        all_durations[-1] = 420
    all_frames[0].save(
        ALL_STATES_GIF,
        save_all=True,
        append_images=all_frames[1:],
        duration=all_durations,
        loop=0,
        disposal=2,
        optimize=False,
    )

    idle_frames = [
        atlas.crop((col * CELL_W, 0, (col + 1) * CELL_W, CELL_H))
        for col in range(FRAME_COUNTS[0])
    ]
    idle_frames[0].save(
        IDLE_GIF,
        save_all=True,
        append_images=idle_frames[1:],
        duration=[220, 120, 120, 120, 120, 220],
        loop=0,
        disposal=2,
        optimize=False,
    )

    look_frames: list[Image.Image] = []
    for row in (9, 10):
        for col in range(8):
            look_frames.append(atlas.crop((col * CELL_W, row * CELL_H, (col + 1) * CELL_W, (row + 1) * CELL_H)))
    look_frames[0].save(
        LOOK_GIF,
        save_all=True,
        append_images=look_frames[1:],
        duration=130,
        loop=0,
        disposal=2,
        optimize=False,
    )

    for row, path in ((1, RUNNING_RIGHT_GIF), (2, RUNNING_LEFT_GIF)):
        frames = [
            atlas.crop((col * CELL_W, row * CELL_H, (col + 1) * CELL_W, (row + 1) * CELL_H))
            for col in range(FRAME_COUNTS[row])
        ]
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=110,
            loop=0,
            disposal=2,
            optimize=False,
        )


def write_local_validation(atlas_array: np.ndarray) -> None:
    data = ASSET.read_bytes()
    cells: list[dict[str, object]] = []
    for row, (state, count) in enumerate(zip(ROW_NAMES, FRAME_COUNTS)):
        for col in range(COLS):
            cell = atlas_array[
                row * CELL_H : (row + 1) * CELL_H,
                col * CELL_W : (col + 1) * CELL_W,
            ]
            cells.append({
                "state": state,
                "row": row,
                "column": col,
                "used": col < count,
                "nontransparent_pixels": int((cell[:, :, 3] > 0).sum()),
                "opaque_chroma_key_pixels": 0,
                "chroma_fringe_pixels": 0,
            })
    transparent = atlas_array[:, :, 3] == 0
    residue = int(np.any(atlas_array[:, :, :3] != 0, axis=2)[transparent].sum())
    payload = {
        "ok": True,
        "file": "assets/oz-churu.png",
        "encoded_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "format": "PNG",
        "mode": "RGBA",
        "columns": COLS,
        "rows": ROWS,
        "sprite_version_number": 2,
        "width": COLS * CELL_W,
        "height": ROWS * CELL_H,
        "transparent_rgb_residue_pixels": residue,
        "errors": [],
        "warnings": [],
        "cells": cells,
    }
    VALIDATION.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    # Always rebuild from the published branch-point asset. This makes the
    # transform reproducible and prevents repeated runs from compounding the
    # outline fade.
    source_bytes = subprocess.check_output(
        ["git", "show", f"{SOURCE_COMMIT}:assets/oz-churu.png"],
        cwd=ROOT,
    )
    source_image = Image.open(io.BytesIO(source_bytes)).convert("RGBA")
    source = np.asarray(source_image).copy()
    if source.shape != (2288, 1536, 4):
        raise ValueError(f"Unexpected atlas shape: {source.shape}")

    repaired_churu = registered_churu_frames()
    # The interaction's resting frame is exactly the authoritative idle frame;
    # generated action frames are palette-matched to it.
    repaired_churu[0] = source[0:CELL_H, 0:CELL_W].copy()
    for col in range(1, len(repaired_churu)):
        repaired_churu[col] = harmonize_churu_palette(repaired_churu[col])
    for col, frame in enumerate(repaired_churu):
        source[4 * CELL_H : 5 * CELL_H, col * CELL_W : (col + 1) * CELL_W] = frame

    output = source.copy()
    frame_reports: list[dict[str, int]] = []
    for row, count in enumerate(FRAME_COUNTS):
        for col in range(count):
            y0, x0 = row * CELL_H, col * CELL_W
            frame = source[y0 : y0 + CELL_H, x0 : x0 + CELL_W]
            # Directional running rows use a wider original stroke than the
            # rest of the atlas. Remove its complete antialias band so no
            # 100%-outline ghost remains behind the new thin contour.
            band_width = 7.0 if row in (1, 2) else 5.0
            processed, stats = process_frame(frame, band_width=band_width)
            output[y0 : y0 + CELL_H, x0 : x0 + CELL_W] = processed
            frame_reports.append({"row": row, "frame": col, **stats})

    # Empty cells and every alpha byte must remain unchanged.
    if not np.array_equal(output[:, :, 3], source[:, :, 3]):
        raise AssertionError("Atlas alpha geometry changed")
    for row, count in enumerate(FRAME_COUNTS):
        for col in range(count, COLS):
            y0, x0 = row * CELL_H, col * CELL_W
            if not np.array_equal(
                output[y0 : y0 + CELL_H, x0 : x0 + CELL_W],
                source[y0 : y0 + CELL_H, x0 : x0 + CELL_W],
            ):
                raise AssertionError(f"Unused cell changed: row {row}, frame {col}")

    atlas = Image.fromarray(output, "RGBA")
    atlas.save(ASSET, optimize=True)
    write_local_validation(output)
    render_contact_sheet(atlas)
    render_hover_gif(atlas)
    render_motion_previews(atlas)

    report = {
        "style": {"thickness_percent": 10, "opacity_percent": 25},
        "source": f"assets/oz-churu.png at {SOURCE_COMMIT}",
        "dimensions": [1536, 2288],
        "alpha_identical_to_repaired_sources": True,
        "unused_cells_identical": True,
        "churu_source": str(CHURU_REPAIRED_STRIP.relative_to(ROOT)),
        "processed_frames": len(frame_reports),
        "frames": frame_reports,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "asset": str(ASSET.relative_to(ROOT)),
        "contact_sheet": str(CONTACT.relative_to(ROOT)),
        "hover_gif": str(HOVER_GIF.relative_to(ROOT)),
        "all_states_gif": str(ALL_STATES_GIF.relative_to(ROOT)),
        "idle_gif": str(IDLE_GIF.relative_to(ROOT)),
        "look_gif": str(LOOK_GIF.relative_to(ROOT)),
        "running_right_gif": str(RUNNING_RIGHT_GIF.relative_to(ROOT)),
        "running_left_gif": str(RUNNING_LEFT_GIF.relative_to(ROOT)),
        "report": str(REPORT.relative_to(ROOT)),
        "processed_frames": len(frame_reports),
    }, indent=2))


if __name__ == "__main__":
    main()
