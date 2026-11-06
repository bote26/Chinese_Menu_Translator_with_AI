"""Presentation helpers: turn a MenuResult into things a UI can draw.

Nothing here talks to Streamlit. Keeping the drawing and tabulating logic in
plain functions means it can be unit-tested without a browser, and reused by
the CLI, the future FastAPI HTML view, or a notebook.
"""

from __future__ import annotations

import io
import statistics
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from .models import MenuResult

# Overlay colours: kept dish, line the classifier rejected, unclaimed box.
COLOUR_KEPT = (26, 127, 55)
COLOUR_REJECTED = (192, 57, 43)
COLOUR_NEUTRAL = (150, 150, 150)


def summary(result: MenuResult) -> dict[str, Any]:
    """Headline numbers for the result: counts, timing, price spread."""
    prices = sorted(
        item.price_value for item in result.items if item.price_value is not None
    )
    tokens = result.metrics.get("tokens") or {}
    return {
        "items": len(result.items),
        "rejected": len(result.dropped),
        "priced": len(prices),
        "price_min": prices[0] if prices else None,
        "price_max": prices[-1] if prices else None,
        "price_median": statistics.median(prices) if prices else None,
        "extractor": result.extractor,
        "total_ms": result.metrics.get("total_ms"),
        "ocr_ms": result.metrics.get("ocr_ms"),
        "llm_ms": result.metrics.get("llm_ms"),
        "vision_ms": result.metrics.get("vision_ms"),
        "tokens": tokens.get("total_tokens"),
    }


def items_dataframe(result: MenuResult) -> pd.DataFrame:
    """One row per dish. Empty columns (e.g. no LLM run) are dropped."""
    rows = []
    for item in result.items:
        dish = item.dish
        rows.append(
            {
                "#": item.id,
                "Chinese": item.cn,
                "Pinyin": dish.pinyin,
                "English": dish.en,
                "Price (CNY)": item.price_value,
                "Category": dish.category,
                "Spice": f"{dish.spice_level}/3" if dish.category else "",
                "Vegetarian": {True: "yes", False: "no"}.get(dish.vegetarian, ""),
                "Description": dish.description,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in ("Pinyin", "English", "Category", "Spice", "Vegetarian", "Description"):
        if frame[column].astype(str).str.strip().eq("").all():
            frame = frame.drop(columns=column)
    return frame


def rejected_dataframe(result: MenuResult) -> pd.DataFrame:
    """Lines the classifier threw away, with the model's reason."""
    return pd.DataFrame(
        [{"Line": d.get("cn", ""), "Reason": d.get("reason", "")} for d in result.dropped]
    )


def thumbnail(
    image: bytes | str | Path,
    size: tuple[int, int] = (420, 300),
    background: tuple[int, int, int] | None = None,
) -> Image.Image:
    """Fit an image inside `size`, centred on a canvas.

    A gallery of samples needs equal-size previews, otherwise the buttons
    underneath land at different heights. Padding (not cropping) keeps the whole
    menu visible.

    The canvas is transparent by default so it blends into whatever page it sits
    on. Pass `background` when a row of thumbnails must read as equal tiles: a
    transparent canvas makes a portrait photo look like a narrow floating strip
    while a landscape one fills the box, and the row looks ragged even though
    every canvas is the same size.
    """
    raw = bytes(image) if isinstance(image, (bytes, bytearray)) else Path(image).read_bytes()
    with Image.open(io.BytesIO(raw)) as opened:
        art = ImageOps.exif_transpose(opened).convert("RGBA")
    art.thumbnail(size, Image.LANCZOS)
    fill = (*background, 255) if background else (0, 0, 0, 0)
    canvas = Image.new("RGBA", size, fill)
    canvas.paste(art, ((size[0] - art.width) // 2, (size[1] - art.height) // 2), art)
    return canvas


def draw_overlay(image: bytes | str | Path, result: MenuResult) -> Image.Image:
    """Draw the OCR boxes over the photo, coloured by classification.

    Requires `result.boxes`, i.e. the scan must have run with
    `include_boxes=True`. Coordinates come from the EXIF-corrected image, so the
    same transposition is applied here or boxes land on the wrong pixels.
    """
    raw = bytes(image) if isinstance(image, (bytes, bytearray)) else Path(image).read_bytes()
    with Image.open(io.BytesIO(raw)) as opened:
        canvas = ImageOps.exif_transpose(opened).convert("RGB").copy()

    kept = {item.cn for item in result.items}
    rejected = {d.get("cn", "") for d in result.dropped}
    draw = ImageDraw.Draw(canvas)

    for box in result.boxes or []:
        text = str(box.get("text", "")).strip()
        if text in kept:
            colour, width = COLOUR_KEPT, 3
        elif text in rejected:
            colour, width = COLOUR_REJECTED, 3
        else:
            colour, width = COLOUR_NEUTRAL, 1
        draw.rectangle(
            [box["x0"], box["y0"], box["x1"], box["y1"]], outline=colour, width=width
        )
    return canvas


def to_csv_bytes(result: MenuResult) -> bytes:
    """CSV for Excel: utf-8-sig so Chinese does not turn into mojibake."""
    return items_dataframe(result).to_csv(index=False).encode("utf-8-sig")
