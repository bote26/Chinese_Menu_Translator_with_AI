"""Command-line harness: same pipeline the API will call, no server needed.

    python -m menu_vision.cli test.jpg test2.jpg test3.jpg --json out/
    python -m menu_vision.cli test.jpg --no-llm --overlay out/test_debug.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .models import MenuResult
from .pipeline import scan_menu

_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
]


def _print_table(result: MenuResult) -> None:
    reader = (
        f"vision={result.llm_model or '-'}"
        if result.extractor == "vision"
        else f"ocr={result.ocr_engine} model={result.llm_model or '-'}"
    )
    print(f"\n=== {result.source}  {result.image_size[0]}x{result.image_size[1]}  "
          f"extractor={result.extractor}  {reader}")
    print(f"    {result.metrics}")
    if not result.items:
        print("    (no items)")
    for item in result.items:
        dish = item.dish
        price = item.price_display or "(no price)"
        line = f"  {item.id:>2}. {item.cn}  {price}"
        if dish.en:
            line += f"  ->  {dish.en}"
        print(line)
        if dish.pinyin:
            print(f"      pinyin: {dish.pinyin}   spice: {dish.spice_level}/3"
                  f"   veg: {dish.vegetarian}   cat: {dish.category}")
        if dish.description:
            print(f"      {dish.description}")
        if dish.price_comment:
            print(f"      price: {dish.price_comment}")
    if result.dropped:
        print("  -- dropped as non-menu text:")
        for dropped in result.dropped:
            print(f"     {dropped['cn']}  ({dropped['reason']})")
    for warning in result.warnings:
        print(f"  ! {warning}")


def _draw_overlay(image_path: Path, result: MenuResult, out_path: Path) -> None:
    """Draw accepted boxes + read prices, like the notebook's verification step."""
    from PIL import Image, ImageDraw, ImageFont

    from .ocr import extract_boxes

    settings = Settings.from_env()
    boxes, _, _ = extract_boxes(image_path, settings)
    accepted = {item.cn for item in result.items}

    with Image.open(image_path) as img:
        canvas = img.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font_path = next((p for p in _FONT_CANDIDATES if p.exists()), None)
    font = ImageFont.truetype(str(font_path), 18) if font_path else ImageFont.load_default()

    for box in boxes:
        hit = box.text.strip() in accepted
        colour = (0, 160, 60) if hit else (200, 200, 200)
        draw.rectangle([box.x0, box.y0, box.x1, box.y1], outline=colour, width=2)
        draw.text((box.x0, max(0, box.y0 - 18)), f"{box.text}", font=font, fill=(200, 30, 30))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"  overlay -> {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan Chinese menu images.")
    parser.add_argument("images", nargs="+", help="image paths")
    parser.add_argument("--json", metavar="DIR", help="write one JSON file per image into DIR")
    parser.add_argument("--overlay", metavar="PNG", help="write a debug overlay (single image only)")
    parser.add_argument(
        "--extractor",
        choices=["vision", "ocr"],
        default=None,
        help="how to read the photo (default: MENU_EXTRACTOR, else vision)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="vision: transcribe names and prices only (fast). ocr: stop after parsing.",
    )
    parser.add_argument("--ocr-only", action="store_true", help="print raw OCR boxes and exit")
    parser.add_argument("--vector", action="store_true",
                        help="print the Chinese {id, cn, price} records sent to the model")
    args = parser.parse_args(argv)

    if args.ocr_only:
        from .ocr import extract_boxes

        settings = Settings.from_env()
        for name in args.images:
            boxes, size, _ = extract_boxes(name, settings)
            print(f"=== {name} {size} -> {len(boxes)} boxes")
            for box in sorted(boxes, key=lambda b: (b.cy, b.x0)):
                print(f"  y={box.y0:>5}-{box.y1:<5} x={box.x0:>5}-{box.x1:<5} "
                      f"c={box.confidence:.2f}  {box.text}")
        return 0

    if args.overlay and len(args.images) > 1:
        print("--overlay supports a single image", file=sys.stderr)
        return 2
    if args.overlay and args.extractor == "vision":
        print(
            "--overlay draws OCR boxes, which the vision extractor does not produce. "
            "Add --extractor ocr.",
            file=sys.stderr,
        )
        return 2

    for name in args.images:
        path = Path(name)
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            continue
        try:
            result = scan_menu(
                path,
                extractor=args.extractor,
                enrich=not args.no_llm,
                include_vector=args.vector,
            )
        except Exception as exc:  # one bad photo should not kill the batch
            print(f"{path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        _print_table(result)
        if args.vector:
            print("  vector sent to model:")
            print("  " + json.dumps(result.vector, ensure_ascii=False)[:1200])
        if args.json:
            out_dir = Path(args.json)
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / f"{path.stem}.json"
            target.write_text(
                json.dumps(result.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"  json -> {target}")
        if args.overlay:
            _draw_overlay(path, result, Path(args.overlay))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
