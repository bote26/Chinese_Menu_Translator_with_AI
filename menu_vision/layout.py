"""Turn a flat list of OCR boxes into menu rows with prices attached.

OCR gives no structure: a menu line like

    酸辣土豆丝   11元

comes back as two independent boxes, and on a phone screenshot the price can
sit *above* the dish name. Everything in this module is pure geometry + regex,
so it is fast, deterministic and cheap to unit-test. Semantic decisions (is
this line even a dish?) are left to the LLM.

Layouts this handles, all seen in real test images:
  * name and price in one box          "酸辣土豆丝11元"
  * name box + price box, same row     (printed menu, two columns)
  * price box above the name           (delivery-app product card)
  * no prices at all                   (price stays None, never invented)
  * section headers                    (larger or repeated text)
"""

from __future__ import annotations

import re
import statistics
from typing import Iterable

from .config import Settings
from .models import OcrBox, RawItem

# A box holding nothing but a price.
#
# The leading group is limited to a currency marker or a known OCR confusion for
# one ("苹" for ¥, "羊" for ¥, "Y"/"y" for ¥). A looser "any two junk characters"
# rule looked tempting but silently turned "销量96" (units sold) into a price of
# 96, so the allowed prefixes are explicit.
#
# Cantonese roast-meat boards price by portion ("88例") or per half bird
# ("50元半只"), with no 元 at all, so a bare number followed by a unit word is a
# price. Without this, "88例" is a plausible-looking name and every one of those
# rows ends up as a phantom dish.
_PRICE_UNITS = r"(?:半只|份|例|只|个|串|碗|盘|盒|杯|瓶|罐|笼|碟|斤|两|位|人|条|打|起)"

PRICE_ONLY_RE = re.compile(
    r"^\s*(?:(?:RMB|CNY)\s*)?[¥￥$苹羊Yy]?\s*"
    r"(?P<num>\d{1,4}(?:[.,]\d{1,2})?)"
    r"\s*(?:元|块钱|块|圆)?"
    # A per-unit price may also print how many units it buys: "16元/15个",
    # "16元14个", "8元/笼". Without the optional count these are read as dish
    # names and the whole row loses its price.
    rf"(?:\s*[/／]?\s*(?:\d{{1,3}}\s*)?{_PRICE_UNITS})?"
    r"\s*$",
    re.IGNORECASE,
)

# A name with the price glued on the end. Requires a currency marker or a
# separating space so dish names that merely end in a digit are left alone.
TRAILING_PRICE_RE = re.compile(
    r"^(?P<name>.+?)\s*(?:[¥￥]\s*(?P<num1>\d{1,4}(?:[.,]\d{1,2})?)"
    r"|(?P<num2>\d{1,4}(?:[.,]\d{1,2})?)\s*(?:元|块钱|块|圆))\s*$"
)

# Characters that can never make up a dish name on their own.
_JUNK_ONLY = re.compile(r"^[\s\-_.,;:!?/\\|·•*#@%&+=~^<>\[\]()（）—…\"'`]+$")

# Section banners. Height alone misses them on real boards: "烧味套餐" is printed
# at the same size as the dishes beneath it, just in a coloured pill. These are
# deliberately narrow - a bare "汤" would swallow 番茄蛋汤, so only exact section
# words and the 套餐 / 系列 / 类 suffixes are trusted.
_SECTION_WORDS = frozenset(
    {
        "小吃", "主食", "甜品", "甜点", "饮料", "饮品", "酒水", "凉菜", "冷菜",
        "热菜", "招牌菜", "推荐", "素食", "烧烤", "套餐", "例排", "烫菜",
    }
)
_SECTION_SUFFIXES = ("套餐", "系列", "类", "菜谱")


def is_section_word(text: str) -> bool:
    """True for a banner like "烧味套餐" / "小吃", false for a dish name."""
    stripped = re.sub(r"\s+", "", text)
    if not stripped or len(stripped) > 8:
        return False
    return stripped in _SECTION_WORDS or stripped.endswith(_SECTION_SUFFIXES)


_OPEN_BRACKETS = "([【（〔"
_CLOSE_BRACKETS = ")]】）〕"


def looks_like_note(text: str) -> bool:
    """True for a printed footnote such as "（以上双拼按价高加5元）".

    Notes are not dishes and must not take a price: a clipped note fragment once
    inherited a neighbouring price and the model dutifully invented a "mixed
    plate" dish for it. An unbalanced bracket is the giveaway - a note cut off
    by the frame edge keeps only its closing bracket.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0] in _OPEN_BRACKETS:
        return True
    return stripped[-1] in _CLOSE_BRACKETS and not any(
        char in stripped for char in _OPEN_BRACKETS
    )


def _normalise_number(raw: str) -> float:
    return float(raw.replace(",", "."))


def parse_price(text: str, max_len: int = 8) -> tuple[float, str] | None:
    """Return (value, raw) if `text` is only a price, else None."""
    stripped = text.strip()
    if not stripped or len(stripped) > max_len:
        return None
    match = PRICE_ONLY_RE.match(stripped)
    if not match:
        return None
    value = _normalise_number(match.group("num"))
    # Menus do sell 0.5 RMB items, but a bare "0" or absurd value is noise.
    if value <= 0 or value > 9999:
        return None
    return value, stripped


def split_name_and_price(text: str) -> tuple[str, float, str] | None:
    """Split "酸辣土豆丝 11元" into ("酸辣土豆丝", 11.0, "11元")."""
    match = TRAILING_PRICE_RE.match(text.strip())
    if not match:
        return None
    name = match.group("name").strip(" :：.-·")
    number = match.group("num1") or match.group("num2")
    if not name or len(name) < 2:
        return None
    price_part = text.strip()[len(match.group("name")):].strip(" :：")
    return name, _normalise_number(number), price_part


def is_plausible_name(text: str) -> bool:
    """Reject boxes that cannot be dish names."""
    stripped = text.strip()
    if not stripped or _JUNK_ONLY.match(stripped):
        return False
    if len(stripped) == 1 and not re.match(r"[\u4e00-\u9fff]", stripped):
        return False
    # A name must carry at least one CJK char or latin word; bare numbers and
    # symbols belong to the price/decorative columns.
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", stripped))


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def is_latin_only(text: str) -> bool:
    """True for a box printed purely in latin script ("Noodles with Sauce")."""
    return bool(_LATIN_WORD_RE.search(text)) and not _CJK_RE.search(text)


def _translation_owner(latin: OcrBox, chinese: list[OcrBox]) -> OcrBox | None:
    """The Chinese dish a printed English line belongs to, if any.

    Two layouts cover the bilingual boards seen so far: the English on the same
    line as the Chinese name (before the price), and the English on the line
    directly underneath, left-aligned with it.
    """
    best: OcrBox | None = None
    best_score: float | None = None
    for box in chinese:
        row_limit = 0.6 * min(box.height, latin.height)
        if abs(box.cy - latin.cy) <= row_limit and box.x1 <= latin.x0 + 12:
            score = latin.x0 - box.x1          # same row, English sits to the right
        else:
            below = latin.y0 - box.y1
            # Centred blocks put the English under a narrower Chinese name, so
            # left edges do not line up - accept a shared centre too.
            aligned = (
                abs(box.x0 - latin.x0) <= 1.3 * box.height
                or abs(box.cx - latin.cx) <= max(0.5 * latin.width, 1.3 * box.height)
            )
            if not (-4 <= below <= 1.8 * box.height and aligned):
                continue
            score = below                      # next line down
        if best_score is None or score < best_score:
            best, best_score = box, score
    return best


def _merge_translations(boxes: list[OcrBox]) -> tuple[list[OcrBox], dict[int, str], int]:
    """Fold printed English lines into the Chinese dish they translate.

    A bilingual board prints both languages, so a naive parse turns every
    English line into a second, price-less copy of the same dish: one 18-dish
    board came back as 31 "items", with prices bound to English fragments.

    English is not a dish - it is a *label* on one. It is attached to its
    Chinese dish as `en_hint` (the model can then refine a name the restaurant
    already chose) and removed from the candidate list. Only applied when the
    board actually contains Chinese dish names, so a genuinely English-only
    menu keeps its lines.
    """
    # Only a purely Chinese line can own a translation. When OCR merges a name
    # and its English into one fat box, that box already carries the English.
    chinese = [
        box
        for box in boxes
        if has_cjk(box.text)
        and parse_price(box.text) is None
        and not _LATIN_WORD_RE.search(box.text)
    ]
    latin = [box for box in boxes if is_latin_only(box.text)]
    if len(chinese) < 3 or not latin:
        return boxes, {}, 0

    owners: dict[int, OcrBox] = {}
    for box in sorted(latin, key=lambda item: (item.cy, item.x0)):
        owner = _translation_owner(box, chinese)
        if owner is not None:
            owners[id(box)] = owner

    # A long English name is often split into several boxes ("Noodles with",
    # "Black Pepper", "Sauce"). Once one fragment is owned, the neighbouring
    # fragments continue it and inherit the same dish.
    assigned = sorted(
        (box for box in latin if id(box) in owners), key=lambda item: (item.cy, item.x0)
    )
    for box in sorted(latin, key=lambda item: (item.cy, item.x0)):
        if id(box) in owners:
            continue
        for previous in reversed(assigned):
            same_row = (
                abs(previous.cy - box.cy) <= 0.6 * min(previous.height, box.height)
                and 0 <= box.x0 - previous.x1 <= 2.5 * previous.height
            )
            next_line = (
                0 <= box.y0 - previous.y1 <= 1.2 * previous.height
                and abs(box.x0 - previous.x0) <= 1.5 * previous.height
            )
            if same_row or next_line:
                owners[id(box)] = owners[id(previous)]
                break

    if not owners:
        return boxes, {}, 0

    merged: dict[int, list[OcrBox]] = {}
    for box in latin:
        owner = owners.get(id(box))
        if owner is not None:
            merged.setdefault(id(owner), []).append(box)
    # Keep each dish's fragments in reading order and space-join them.
    hints: dict[int, str] = {}
    for owner_id, fragments in merged.items():
        fragments.sort(key=lambda item: (item.cy, item.x0))
        text = " ".join(item.text.strip() for item in fragments).strip()
        hints[owner_id] = re.sub(r"\s{2,}", " ", text)

    consumed = set(owners)  # owners is keyed by id(box) already
    kept = [box for box in boxes if id(box) not in consumed]
    return kept, hints, len(consumed)


def _vertical_overlap(a: OcrBox, b: OcrBox) -> float:
    overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    if overlap <= 0:
        return 0.0
    return overlap / max(1, min(a.height, b.height))


def group_rows(boxes: Iterable[OcrBox], y_tolerance: float = 0.55) -> list[list[OcrBox]]:
    """Cluster boxes into visual rows, each sorted left-to-right.

    Each box is compared with the *centre* of an existing row, never with the
    row's growing bounding span. Span matching chains: one tall box stretches
    the span, the next row's box now overlaps it, and two menu rows silently
    merge - which is how a price ends up detached from its dish.
    """
    ordered = sorted(boxes, key=lambda box: box.cy)
    rows: list[list[OcrBox]] = []
    for box in ordered:
        best_index = None
        best_distance = None
        for index, row in enumerate(rows):
            row_cy = statistics.fmean(member.cy for member in row)
            row_height = statistics.fmean(member.height for member in row)
            limit = y_tolerance * min(box.height, row_height)
            distance = abs(box.cy - row_cy)
            if distance <= limit and (best_distance is None or distance < best_distance):
                best_index, best_distance = index, distance
        if best_index is None:
            rows.append([box])
        else:
            rows[best_index].append(box)
    for row in rows:
        row.sort(key=lambda box: box.x0)
    rows.sort(key=lambda row: statistics.fmean(box.cy for box in row))
    return rows


def _pair_row(row: list[OcrBox]) -> tuple[list[tuple[OcrBox, tuple[float, str] | None]], list[tuple[OcrBox, tuple[float, str]]]]:
    """Attach at most one price to each name box within a single row.

    A price binds to the nearest name on its left (the usual "name .... price"
    column layout); if nothing sits to the left it binds to the nearest name on
    the right.
    """
    names: list[OcrBox] = []
    prices: list[tuple[OcrBox, tuple[float, str]]] = []
    for box in row:
        price = parse_price(box.text)
        if price is not None:
            prices.append((box, price))
        elif is_plausible_name(box.text):
            names.append(box)

    assigned: dict[int, tuple[float, str]] = {}
    orphans: list[tuple[OcrBox, tuple[float, str]]] = []
    for box, price in prices:
        left = [name for name in names if name.x1 <= box.x0 + 4]
        right = [name for name in names if name.x0 >= box.x1 - 4]
        candidate = None
        if left:
            # Nearest name to the left; on an x tie the vertically closest wins,
            # so stacked columns sharing a right edge stay unambiguous.
            candidate = max(left, key=lambda name: (name.x1, -abs(name.cy - box.cy)))
        elif right:
            nearest = min(right, key=lambda name: (name.x0, abs(name.cy - box.cy)))
            # Only a genuinely adjacent name can own a price printed to its left.
            # Without this limit the leftmost column of a board - whose dish
            # names are cropped off at the frame edge - hands each of its prices
            # to the first dish of the *next* column, corrupting both columns.
            if nearest.x0 - box.x1 <= max(2.5 * box.height, 1):
                candidate = nearest
        if candidate is None:
            orphans.append((box, price))
            continue
        existing = assigned.get(id(candidate))
        # Keep the closer/clashing-safe option if a name somehow sees two prices
        # (e.g. "12元 15元" for small/large portions): the first one wins and the
        # rest stay unassigned rather than overwriting real data.
        if existing is None:
            assigned[id(candidate)] = price
        else:
            orphans.append((box, price))

    return [(name, assigned.get(id(name))) for name in names], orphans


def parse_menu(
    boxes: list[OcrBox], settings: Settings, currency: str = "CNY"
) -> tuple[list[RawItem], list[str], list[str]]:
    """Boxes -> ordered RawItems, section names, and warnings."""
    warnings: list[str] = []
    usable = [box for box in boxes if is_plausible_name(box.text) or parse_price(box.text)]
    if not usable:
        return [], [], ["No usable text detected in the image."]

    # Printed footnotes and text sliced by the frame edge are not menu lines.
    # Drop them before anything can bind a price to them, and say so: silently
    # deleting rows would look like the scanner simply missed them.
    notes = [box for box in usable if looks_like_note(box.text)]
    fragments = [
        box
        for box in usable
        if not looks_like_note(box.text)
        and len(box.text.strip()) == 1
        and box.x0 <= 1
    ]
    if notes or fragments:
        dropped_ids = {id(box) for box in notes} | {id(box) for box in fragments}
        usable = [box for box in usable if id(box) not in dropped_ids]
        if notes:
            listed = ", ".join(f"“{box.text.strip()}”" for box in notes[:3])
            warnings.append(f"Skipped {len(notes)} printed note line(s): {listed}")
        if fragments:
            warnings.append(
                f"Dropped {len(fragments)} one-character fragment(s) cut off at the "
                "image edge - the dish names in that column are off-frame."
            )

    # Bilingual boards: fold the printed English into the Chinese dish it
    # labels, before English lines can turn into duplicate dishes.
    usable, translation_hints, merged_lines = _merge_translations(usable)
    if merged_lines:
        warnings.append(
            f"Folded {merged_lines} printed English line(s) into their Chinese dishes."
        )

    heights = [box.height for box in usable if len(box.text.strip()) >= 2]
    median_height = statistics.median(heights) if heights else 0.0
    max_height = max(heights) if heights else 0.0

    text_counts: dict[str, int] = {}
    for box in usable:
        key = re.sub(r"\s+", "", box.text)
        text_counts[key] = text_counts.get(key, 0) + 1

    price_boxes = [box for box in usable if parse_price(box.text) is not None]

    def has_price_beside(box: OcrBox) -> bool:
        """A dish line is followed by its price; a section banner is not.

        This is what separates a banner ("烧味套餐") from a dish that happens to
        be called a set meal ("咖喱鸡套餐"): text alone cannot tell them apart.
        """
        return any(
            price.x0 >= box.x1 - 4 and box.y0 - 4 <= price.cy <= box.y1 + 4
            for price in price_boxes
        )

    headers: list[OcrBox] = []
    content: list[OcrBox] = []
    for box in usable:
        tall = median_height and box.height >= median_height * settings.header_height_ratio
        repeated = text_counts[re.sub(r"\s+", "", box.text)] >= 2
        # A line printed twice at the largest size is a title, not a dish: menus
        # repeat the restaurant/section banner on each panel. Comparing against
        # the tallest text (not just the median) keeps this working on sparse
        # pages, where a handful of boxes skews the median upward.
        repeated_title = repeated and max_height and box.height >= max_height * 0.95
        banner = is_section_word(box.text) and not has_price_beside(box)
        # A line with a price beside it is a dish, however tall it is printed -
        # OCR happily merges a Chinese name and its English onto one fat box,
        # which pushed real dishes into the header list.
        title_like = (tall or repeated_title) and not has_price_beside(box)
        if (title_like or banner) and parse_price(box.text) is None:
            headers.append(box)
        else:
            content.append(box)

    rows = group_rows(content, settings.row_tolerance)

    ordered_items: list[RawItem] = []
    pending_prices: list[tuple[OcrBox, tuple[float, str]]] = []
    for row in rows:
        pairs, orphans = _pair_row(row)
        pending_prices.extend(orphans)
        for box, price in pairs:
            name = box.text.strip()
            price_raw = None
            price_value = None
            if price is not None:
                price_value, price_raw = price
            else:
                split = split_name_and_price(name)
                if split is not None:
                    name, price_value, price_raw = split
            ordered_items.append(
                RawItem(
                    cn=name,
                    price_raw=price_raw,
                    price_value=price_value,
                    currency=currency,
                    confidence=round(box.confidence, 3),
                    bbox=(box.x0, box.y0, box.x1, box.y1),
                    en_hint=translation_hints.get(id(box), ""),
                )
            )

    # Prices with no name on their row (delivery-app cards put the price above
    # the dish). Attach each to the closest name directly below it. Claims are
    # resolved nearest-first so a struck-through original price next to the real
    # one cannot win just because it was seen first.
    # Pass 1: a price printed on the name's own line, to its right.
    #
    # Row clustering is fragile here: on the bilingual boards the price is
    # bottom-aligned with the two-line name/English block, so its centre sits
    # ~36px below the name's and the two land in different rows. Vertical
    # overlap plus "to the right" is the relation that actually holds.
    used_prices: set[int] = set()

    def _overlap_ratio(price: OcrBox, item: RawItem) -> float:
        overlap = min(item.bbox[3], price.y1) - max(item.bbox[1], price.y0)
        item_height = item.bbox[3] - item.bbox[1]
        if overlap <= 0 or item_height <= 0:
            return 0.0
        return overlap / min(price.height, item_height)

    same_line: list[tuple[float, int, RawItem, tuple[float, str]]] = []
    for price_index, (box, price) in enumerate(pending_prices):
        for item in ordered_items:
            if box.x0 >= item.bbox[2] - 4 and _overlap_ratio(box, item) >= 0.3:
                # Nearest name to the left wins: on a multi-column board that is
                # always the dish this price is printed beside.
                same_line.append((box.x0 - item.bbox[2], price_index, item, price))
    for _, price_index, item, price in sorted(same_line, key=lambda claim: claim[0]):
        if price_index in used_prices or item.price_value is not None:
            continue
        item.price_value, item.price_raw = price[0], price[1]
        used_prices.add(price_index)

    # Pass 2: prices with no name on their row (delivery-app cards put the price
    # above the dish). Attach each to the closest name directly below it. Claims
    # are resolved nearest-first so a struck-through original price next to the
    # real one cannot win just because it was seen first.
    claims: list[tuple[float, int, RawItem, tuple[float, str]]] = []
    for price_index, (box, price) in enumerate(pending_prices):
        if price_index in used_prices:
            continue
        for item in ordered_items:
            gap = item.bbox[1] - box.y1
            centre_gap = abs((item.bbox[0] + item.bbox[2]) / 2 - box.cx)
            # Being "the next name below" is not enough: on a multi-column board
            # the nearest name below a clipped left-hand price is in the column
            # to its right, 300px away. Require horizontal alignment - centres
            # match (a centred card) or an edge matches (a left-aligned card).
            aligned = (
                centre_gap <= 1.5 * box.width
                or abs(item.bbox[0] - box.x0) <= 1.5 * box.height
                or abs(item.bbox[2] - box.x1) <= 1.5 * box.height
            )
            # 1.6x the price's own height: tight enough that a sold-count line
            # far below the price is not a candidate.
            if -6 <= gap <= max(1.6 * box.height, 1) and aligned:
                claims.append((max(gap, 0), price_index, item, price))
    for _, price_index, item, price in sorted(claims, key=lambda claim: claim[0]):
        if price_index in used_prices or item.price_value is not None:
            continue
        item.price_value, item.price_raw = price[0], price[1]
        used_prices.add(price_index)

    # Prices no dish claimed are worth reporting rather than discarding in
    # silence: on this kind of photo it means a column is cut off at the edge.
    stranded = len(pending_prices) - len(used_prices)
    if stranded:
        warnings.append(
            f"{stranded} price(s) had no dish name beside them - a column is "
            "probably cut off at the edge of the photo."
        )

    # Assign sections from the header above each item. Sections are *vertical*:
    # a banner sits over one column, so an item only inherits a header it is
    # horizontally close to. Assigning by y alone gives every column on a
    # four-column board whatever banner happened to be lowest.
    sections: list[str] = []
    if headers:
        headers.sort(key=lambda box: box.cy)
        slack = max(statistics.median(header.width for header in headers) * 1.5, 80)
        for item in ordered_items:
            item_cx = (item.bbox[0] + item.bbox[2]) / 2
            above = [header for header in headers if header.cy < item.bbox[1]]
            if not above:
                continue
            # Nearest banner to the left/right; on an x tie the lower one wins.
            nearest = min(above, key=lambda header: (abs(header.cx - item_cx), -header.cy))
            if abs(nearest.cx - item_cx) <= slack:
                item.section = nearest.text.strip()
        sections = [header.text.strip() for header in headers]
        # Drop header text that also leaked in as an item.
        ordered_items = [item for item in ordered_items if item.cn not in sections]

    # Collapse exact duplicates (overlapping detections, two-page reprints).
    deduped: list[RawItem] = []
    seen: dict[tuple[str, float | None], int] = {}
    for item in ordered_items:
        key = (item.cn, item.price_value)
        if key in seen:
            index = seen[key]
            if item.confidence > deduped[index].confidence:
                deduped[index] = item
            continue
        seen[key] = len(deduped)
        deduped.append(item)

    if not any(item.price_value is not None for item in deduped):
        warnings.append("No prices found; items will be returned without price data.")

    # Single-character lines are usually unit words ("份") or a stray glyph;
    # two-character names are perfectly normal (韭菜, 豆腐), so they are not flagged.
    single_char = [item.cn for item in deduped if len(item.cn) == 1]
    if single_char:
        warnings.append(f"Single-character line(s) detected (likely unit words): {', '.join(single_char[:5])}")

    return deduped, sections, warnings
