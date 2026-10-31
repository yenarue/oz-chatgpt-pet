#!/usr/bin/env python3
"""Apply the approved 10%-thickness / 25%-opacity style to Oz Feather."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import apply_outline_style as style


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets" / "oz-feather.png"
GIF = ROOT / "previews" / "feather.gif"
WEBP = ROOT / "previews" / "outline-10-opacity-25-feather.webp"
VALIDATION = ROOT / "docs" / "feather-validation.json"
REPORT = ROOT / "docs" / "feather-outline-10-opacity-25-report.json"


def render_hover_previews(atlas: Image.Image) -> None:
    frames = [
        atlas.crop((col * style.CELL_W, 4 * style.CELL_H, (col + 1) * style.CELL_W, 5 * style.CELL_H))
        for col in range(style.FRAME_COUNTS[4])
    ]
    durations = [280, 140, 140, 140, 280]
    frames[0].save(
        GIF,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=False,
    )
    frames[0].save(
        WEBP,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        lossless=True,
        method=6,
    )


def write_validation(atlas: np.ndarray) -> None:
    data = ASSET.read_bytes()
    cells: list[dict[str, object]] = []
    for row, (state, count) in enumerate(zip(style.ROW_NAMES, style.FRAME_COUNTS)):
        for col in range(style.COLS):
            cell = atlas[
                row * style.CELL_H : (row + 1) * style.CELL_H,
                col * style.CELL_W : (col + 1) * style.CELL_W,
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
    transparent = atlas[:, :, 3] == 0
    payload = {
        "ok": True,
        "file": "assets/oz-feather.png",
        "encoded_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "format": "PNG",
        "mode": "RGBA",
        "columns": style.COLS,
        "rows": style.ROWS,
        "sprite_version_number": 2,
        "width": style.COLS * style.CELL_W,
        "height": style.ROWS * style.CELL_H,
        "transparent_rgb_residue_pixels": int(
            np.any(atlas[:, :, :3] != 0, axis=2)[transparent].sum()
        ),
        "errors": [],
        "warnings": [],
        "cells": cells,
    }
    VALIDATION.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    source_bytes = subprocess.check_output(
        ["git", "show", f"{style.SOURCE_COMMIT}:assets/oz-feather.png"],
        cwd=ROOT,
    )
    source = np.asarray(Image.open(io.BytesIO(source_bytes)).convert("RGBA")).copy()
    if source.shape != (2288, 1536, 4):
        raise ValueError(f"Unexpected feather atlas shape: {source.shape}")

    output = source.copy()
    frame_reports: list[dict[str, object]] = []
    for row, count in enumerate(style.FRAME_COUNTS):
        for col in range(count):
            y0, x0 = row * style.CELL_H, col * style.CELL_W
            frame = source[y0 : y0 + style.CELL_H, x0 : x0 + style.CELL_W]
            band_width = 7.0 if row in (1, 2) else 5.0
            processed, stats = style.process_frame(
                frame,
                band_width=band_width,
                clean_left_running_head=(row == 2),
            )
            output[y0 : y0 + style.CELL_H, x0 : x0 + style.CELL_W] = processed
            frame_reports.append({"row": row, "frame": col, **stats})

    for row, count in enumerate(style.FRAME_COUNTS):
        for col in range(count, style.COLS):
            y0, x0 = row * style.CELL_H, col * style.CELL_W
            if not np.array_equal(
                output[y0 : y0 + style.CELL_H, x0 : x0 + style.CELL_W],
                source[y0 : y0 + style.CELL_H, x0 : x0 + style.CELL_W],
            ):
                raise AssertionError(f"Unused feather cell changed: row {row}, frame {col}")

    output[:, :, :3][output[:, :, 3] == 0] = 0
    atlas = Image.fromarray(output, "RGBA")
    atlas.save(ASSET, optimize=True)
    write_validation(output)
    render_hover_previews(atlas)

    REPORT.write_text(json.dumps({
        "style": {"thickness_percent": 10, "opacity_percent": 25},
        "source": f"assets/oz-feather.png at {style.SOURCE_COMMIT}",
        "dimensions": [1536, 2288],
        "processed_frames": len(frame_reports),
        "alpha_geometry_rebuilt": True,
        "edge_trim_pixels": style.EDGE_TRIM,
        "frames": frame_reports,
    }, indent=2) + "\n")
    print(json.dumps({
        "asset": str(ASSET.relative_to(ROOT)),
        "gif": str(GIF.relative_to(ROOT)),
        "webp": str(WEBP.relative_to(ROOT)),
        "validation": str(VALIDATION.relative_to(ROOT)),
        "processed_frames": len(frame_reports),
    }, indent=2))


if __name__ == "__main__":
    main()
