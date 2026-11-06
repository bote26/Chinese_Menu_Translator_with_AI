"""Vision extractor: one multimodal call reads the photo and explains the menu.

Why this exists
---------------
The first architecture put easyocr in front of the model on the theory that OCR
is more reliable for prices. On clean scans it is. On real phone photos of
Chinese wall boards it is not: easyocr 1.7.2 read `锅锅灶灶菜谱` as `钢镉灶灶莱谱`
and `粿条` as `凝条`, so on one board *zero* Chinese dish names survived the
confidence filter, and on another it turned the printed `16元14个` into `¥35`.
Every one of those is a wrong price that looks plausible.

Measured on four boards from the same shop (one call each, this module):

    board   OCR+text path        this module
    test5   260 s, 3 bogus rows   56 s, 13/13 correct
    test6   150 s, 8 unpriced     15 s, 13/13 correct
    test7   248 s, all names lost 64 s, 10/10 correct
    test8   308 s, tag stole ¥8   25 s, 11/12 correct

So the price authority is no longer OCR coordinates but `parse_price`, applied to
the price *string* the model transcribed. That keeps one cheap guarantee (no
invented numbers, units parsed by our code, not the model's) and drops a whole
layer of fragile geometry. What it costs is the debug overlay: there are no
boxes, so `result.boxes` stays empty in this mode. `extractor="ocr"` still
exists for exactly that reason.

Everything here is pure: no network, no Streamlit. The HTTP call belongs to the
client in `llm.py`, which the pipeline drives.
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .config import Settings
from .layout import has_cjk, parse_price
from .models import DishInfo, EnrichedItem
from .pricing import format_price

VISION_PROMPT = """\
You are a Chinese-menu analyst with vision. The photo shows a Chinese restaurant \
menu: a printed wall board, a laminated page with several price columns, or a \
delivery-app product screen.

Do two jobs in one pass.

1. READ the board exactly as printed.
- One object per purchasable food or drink item, in reading order (top to bottom, \
then left to right across columns).
- `cn`: the Chinese name exactly as printed. Repair only obvious damage. Never \
leave it empty and never invent a name for a price you can see but cannot read - \
a price with no name beside it belongs to a column that is cut off, so omit it.
- `price`: the price string exactly as printed, unit included: "18.5元", \
"16元14个", "8元/笼", "88例", "50元半只". Use null only when that item prints no \
price at all - on a wall board whole columns print none.
- `section`: the heading above the item ("烧腊", "主食", "饮品"), or "" if none.
- `en_printed`: English printed next to or under the same item, verbatim, else "". \
An English line is never a separate item.
- Skip everything that is not a dish: headings, shop name, address, phone number, \
watermark, table numbers, app UI, "扫码点餐" badges, bare unit words.

2. EXPLAIN each item you read.
- `pinyin` with tone marks and word spacing.
- `en`: a natural name a foreigner would order from ("Hot and Sour Shredded \
Potato", not "Sour Pepper Potato Silk").
- `description`: 1-2 sentences for someone who has never eaten Chinese food - what \
it is, main ingredients, how it is cooked, how it tastes.
- `category` ("noodle", "rice bowl", "roast meat", "cold dish", "drink"...), \
`spice_level` 0-3, `vegetarian` true/false/null, `allergens` from wheat, soy, egg, \
dairy, peanut, tree nut, shellfish, fish, sesame, gluten.

Hard rules:
- Never invent an item, a name or a price. If a line is unreadable, skip it.
- Do not mention price anywhere except the `price` field: comparisons against the \
rest of the menu are computed separately and anything you write about price is \
discarded.
- Answer in {language}.
- Return STRICT JSON only: an array, one object per item, no markdown fences, no \
commentary.

Keys per object: cn (string), price (string or null), section (string), \
en_printed (string), pinyin (string), en (string), description (string), \
ingredients (array of strings), category (string), spice_level (int 0-3), \
vegetarian (bool or null), allergens (array of strings), confidence (0-1), \
is_dish (bool, false only for a line that turned out not to be purchasable food).
"""

# Used when the caller does not want translations (the UI's "translate" toggle).
# Same reading rules, no explanations: the reply is ~3x smaller and ~3x faster.
TRANSCRIBE_PROMPT = """\
You are a Chinese-menu transcriber with vision. The photo shows a Chinese \
restaurant menu: a wall board, a laminated page, or a delivery-app screen.

List every purchasable food or drink item on it, in reading order.

- `cn`: the Chinese name exactly as printed; repair only obvious damage; never \
empty, never invented.
- `price`: the price string exactly as printed, unit included ("18.5元", \
"16元14个", "88例", "50元半只"), or null when that item prints no price.
- `section`: the heading above it, or "".
- `en_printed`: English printed on the board for that item, verbatim, else "".
- `is_dish`: false only for a line that turned out not to be purchasable food.
- Skip non-menu text: headings, shop name, phone, watermark, app UI, unit words.

Never invent an item or a price. Return STRICT JSON only: an array of objects with \
keys cn, price, section, en_printed, is_dish. No markdown fences, no commentary.
"""

_TRAILING_DOT = re.compile(r"\s+$")


def build_vision_prompt(settings: Settings, *, explain: bool = True) -> str:
    """The instruction text that rides alongside the photo."""
    if not explain:
        return TRANSCRIBE_PROMPT
    return VISION_PROMPT.format(language=settings.target_language)


def encode_image(
    image: bytes | str | Path, settings: Settings
) -> tuple[str, tuple[int, int]]:
    """Return `(base64 jpeg, (width, height))` ready for a data: URL.

    Always re-encodes instead of forwarding the upload: phone photos carry an
    EXIF rotation flag that a model will not honour, so a landscape board would
    arrive sideways. Downscaling is capped rather than aggressive - on a dense
    four-column board the difference between `粿条` and `凝条` is a few pixels.
    """
    raw = bytes(image) if isinstance(image, (bytes, bytearray)) else Path(image).read_bytes()
    with Image.open(io.BytesIO(raw)) as opened:
        art = ImageOps.exif_transpose(opened).convert("RGB")
        longest = max(art.size)
        if settings.vision_max_side > 0 and longest > settings.vision_max_side:
            scale = settings.vision_max_side / longest
            art = art.resize(
                (max(1, int(art.width * scale)), max(1, int(art.height * scale))),
                Image.LANCZOS,
            )
        size = art.size
        buffer = io.BytesIO()
        art.save(
            buffer,
            format="JPEG",
            quality=max(50, min(95, settings.vision_jpeg_quality)),
            optimize=True,
        )
    return base64.b64encode(buffer.getvalue()).decode("ascii"), size


def build_vision_messages(payload: str, prompt: str) -> list[dict[str, Any]]:
    """One user turn: the image, then the instructions."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{payload}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _price_of(raw: Any) -> tuple[str | None, float | None]:
    """Printed price text and its parsed number, in that order.

    The model transcribes; we do the arithmetic. `parse_price` already knows
    every unit this shop prints (份/例/半只/个/笼), so a price the model read
    correctly cannot turn into a wrong number here - and one it mangled is
    dropped rather than guessed.
    """
    if raw is None:
        return None, None
    text = _TRAILING_DOT.sub("", str(raw)).strip()
    if not text or text.lower() in {"null", "none", "n/a", "-"}:
        return None, None
    parsed = parse_price(text)
    return text, (parsed[0] if parsed else None)


def parse_vision_rows(
    rows: Any, *, currency: str = "CNY", explain: bool = True
) -> tuple[list[EnrichedItem], list[dict[str, Any]], list[str], list[str]]:
    """Turn the model's JSON array into validated items.

    Returns `(items, dropped, sections, warnings)`. Anything unverifiable is
    dropped loudly: a row with no Chinese name is a model artefact, not a dish,
    and the reason is recorded so the UI can show it instead of hiding it.
    """
    items: list[EnrichedItem] = []
    dropped: list[dict[str, Any]] = []
    sections: list[str] = []
    warnings: list[str] = []
    unparsed: list[str] = []

    for raw in rows if isinstance(rows, list) else []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("cn") or "").strip()
        price_text, price_value = _price_of(raw.get("price"))
        section = str(raw.get("section") or "").strip() or None

        # A row without a Chinese name cannot be a dish on a Chinese menu. This
        # is also the guard that catches glm-4.5v inventing nameless rows for a
        # cropped price column when its reasoning is turned off.
        if not name or not has_cjk(name):
            if price_text or raw.get("is_dish") is not False:
                dropped.append(
                    {
                        "cn": name or "(no Chinese name)",
                        "price": price_text,
                        "reason": "the model returned a price with no readable dish name",
                    }
                )
            continue

        if raw.get("is_dish") is False:
            dropped.append(
                {
                    "cn": name,
                    "price": price_text,
                    "reason": str(raw.get("reason") or "not a purchasable item"),
                }
            )
            continue

        if price_text and price_value is None:
            unparsed.append(price_text)

        info = DishInfo(
            is_dish=True,
            pinyin=str(raw.get("pinyin") or "") if explain else "",
            en=str(raw.get("en") or "") if explain else "",
            description=str(raw.get("description") or "") if explain else "",
            ingredients=_strings(raw.get("ingredients")) if explain else [],
            category=str(raw.get("category") or "") if explain else "",
            spice_level=_spice(raw.get("spice_level")) if explain else 0,
            vegetarian=_tri_state(raw.get("vegetarian")) if explain else None,
            allergens=_strings(raw.get("allergens")) if explain else [],
            confidence=_confidence(raw.get("confidence")),
        )
        if section and section not in sections:
            sections.append(section)
        items.append(
            EnrichedItem(
                id=len(items) + 1,
                cn=name,
                price_raw=price_text,
                price_value=price_value,
                currency=currency,
                price_display=format_price(price_value, currency),
                section=section,
                # No OCR ran, so there is no per-box score. The model's own
                # confidence lives on `dish.confidence`.
                ocr_confidence=0.0,
                en_printed=str(raw.get("en_printed") or "").strip(),
                dish=info,
            )
        )

    if unparsed:
        warnings.append(
            f"{len(unparsed)} price(s) were printed in a form this parser does not "
            f"understand ({', '.join(unparsed[:3])}) - they are kept as text only."
        )
    if rows and not items and not dropped:
        warnings.append("The model returned no menu items for this photo.")
    return items, dropped, sections, warnings


def _strings(value: Any) -> list[str]:
    return [str(entry) for entry in value] if isinstance(value, list) else []


def _spice(value: Any) -> int:
    if isinstance(value, (int, float)):
        return max(0, min(3, int(value)))
    return 0


def _tri_state(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.85
