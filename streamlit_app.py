"""Streamlit demo: upload a Chinese menu photo, get it classified and translated.

Run it with the interpreter that has easyocr installed:

    C:/Python314/python.exe -m streamlit run streamlit_app.py

The page is deliberately thin: all OCR/LLM work happens in `menu_vision`, and
all drawing/table building happens in `menu_vision.report`. This file only
arranges widgets, so the same pipeline keeps serving the CLI and the future API.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

PROJECT_DIR = Path(__file__).resolve().parent
# `streamlit run some/other/dir/app.py` does not guarantee this folder is
# importable, so make the local package resolvable from any working directory.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from menu_vision.config import Settings  # noqa: E402  (after sys.path setup)
from menu_vision.models import EnrichedItem, MenuResult, has_dish_facts  # noqa: E402
from menu_vision.pipeline import scan_menu  # noqa: E402
from menu_vision.advisor import (  # noqa: E402
    Answer,
    ask_about_dish,
    ask_about_menu,
    text_client,
)
from menu_vision.preferences import (  # noqa: E402
    TasteProfile,
    append_event,
    clear_history,
    derive_profile,
    event_from_item,
    load_events,
    load_profile,
    save_profile,
    venue_key,
)
from menu_vision.recommend import (  # noqa: E402
    Recommendation,
    ScoredDish,
    build_plan,
    budget_line,
)
from menu_vision.report import (  # noqa: E402
    draw_overlay,
    items_dataframe,
    rejected_dataframe,
    summary,
    thumbnail,
    to_csv_bytes,
)

SAMPLES: dict[str, str] = {
    "test7.jpg": "Bilingual, soft focus — OCR reads no Chinese",
    "test4.jpg": "Four columns, left one cropped",
    "test.jpg": "Printed canteen menu, 21 prices",
    "test2.jpg": "Skewers, no prices printed",
}

def _page_setup() -> None:
    """Page config and the one layout rule that has no config token.

    A function rather than two bare calls because Streamlit caches imported
    modules between reruns: `out/preview_ui.py` imports this file once, so
    module-level commands only ever run for the first session. It calls this
    again on every run to get the same frame as production.
    """
    st.set_page_config(
        page_title="Menu Vision",
        page_icon=":material/restaurant_menu:",
        layout="wide",
    )
    # Colour, type and radius live in .streamlit/config.toml, where Streamlit
    # keeps them consistent across every element and across versions. Width is
    # the exception: there is no token for it, and layout="wide" stretches a
    # four-up gallery across a whole monitor.
    st.html(
        """
        <style>
          .block-container, [data-testid="stMainBlockContainer"] {
            padding-top: 2.2rem !important; padding-bottom: 3rem !important;
            max-width: 1180px !important;
          }
        </style>
        """
    )


_page_setup()


def _mat() -> tuple[int, int, int]:
    """Mat colour behind a thumbnail.

    Menus are printed on light paper, so a mat reads as a photo border. It uses
    the theme's secondary surface, one step off the page: a mat in the page
    colour disappears and leaves a portrait photo looking like a narrow strip.
    """
    return (36, 30, 25) if st.context.theme.get("type") == "dark" else (244, 239, 230)


def _fit(size: tuple[int, int], max_width: int, max_height: int) -> tuple[int, int]:
    """Largest box with `size`'s aspect ratio that fits the given bounds.

    `thumbnail()` pads to a fixed canvas, which is right for a row of equal
    gallery tiles and wrong for a single preview: padding a portrait menu into a
    4:3 canvas turns it into a small strip floating in a mat.
    """
    width, height = size
    if width <= 0 or height <= 0:
        return (max_width, max_height)
    scale = min(max_width / width, max_height / height)
    return (max(1, round(width * scale)), max(1, round(height * scale)))


def _display_size(image_bytes: bytes) -> tuple[int, int]:
    """The photo's size as it will actually be shown, EXIF rotation applied.

    `result.image_size` is measured before the orientation tag is honoured, so a
    phone photo that is stored landscape but displayed portrait would be padded
    into a landscape box - the exact strip-in-a-frame this avoids.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            return ImageOps.exif_transpose(opened).size
    except Exception:
        return (0, 0)


# --------------------------------------------------------------------------- #
# Data acquisition
# --------------------------------------------------------------------------- #
def _image_choice() -> tuple[bytes | None, str, str]:
    """Sidebar image picker. Returns (bytes, filename, source note)."""
    with st.sidebar:
        st.subheader("Choose a photo", icon=":material/photo_camera:", divider=False)
        upload = st.file_uploader(
            "Upload a photo",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="A straight-on photo works best; the model can cope with a "
                 "hand-held one at an angle.",
        )
        available = ["(none)"] + [name for name in SAMPLES if (PROJECT_DIR / name).exists()]
        if st.session_state.get("sample_choice") not in available:
            st.session_state.sample_choice = "(none)"
        sample = st.selectbox("or use a sample image", available, key="sample_choice")
        # Only the selected sample needs a note here; the landing cards already
        # describe all four, and repeating them made the sidebar the densest
        # part of the page.
        if sample != "(none)":
            st.caption(SAMPLES[sample])

    if upload is not None and upload.size:
        return upload.getvalue(), upload.name, "uploaded"
    if sample != "(none)":
        path = PROJECT_DIR / sample
        if path.exists():
            return path.read_bytes(), sample, "sample"
    return None, "", ""


# --------------------------------------------------------------------------- #
# Running the pipeline (cached for the session)
# --------------------------------------------------------------------------- #
def _run_scan(image_bytes: bytes, enrich: bool, extractor: str) -> MenuResult:
    with st.status("Scanning the image", expanded=True) as status:
        try:
            result = scan_menu(
                image_bytes,
                extractor=extractor,
                enrich=enrich,
                # Boxes only exist on the OCR path; asking for them on the
                # vision path would just add a warning nobody can act on.
                include_boxes=extractor == "ocr",
                progress=lambda message: status.write(message),
            )
        except Exception as exc:  # a demo should explain, not dump a traceback
            status.update(label="Scan failed", state="error", expanded=True)
            st.error(f"{type(exc).__name__}: {exc}")
            st.stop()
        seconds = (result.metrics.get("total_ms") or 0) / 1000
        status.update(
            label=f"Scan finished in {seconds:.1f}s", state="complete", expanded=False
        )
    return result


def _cached_scan(image_bytes: bytes, enrich: bool, extractor: str) -> MenuResult:
    """Re-run only when the image or the actual options change.

    Display options (tabs, overlay toggle) then cost nothing on re-render. The
    extractor is part of the key because it changes how the photo is read, not
    just how it is displayed.
    """
    key = f"{hashlib.sha256(image_bytes).hexdigest()}:{enrich}:{extractor}"
    cached = st.session_state.get("scan")
    if cached and cached["key"] == key:
        return cached["result"]
    result = _run_scan(image_bytes, enrich, extractor)
    st.session_state["scan"] = {"key": key, "result": result}
    return result


# --------------------------------------------------------------------------- #
# Taste memory
# --------------------------------------------------------------------------- #
def _data_dir() -> Path:
    return Settings.from_env().data_dir


def _stated_profile() -> tuple[TasteProfile, list[str]]:
    """The profile as the sidebar has it, loaded from disk on first run."""
    if "stated" not in st.session_state:
        stored, warnings = load_profile(_data_dir())
        st.session_state["stated"] = stored
        st.session_state.setdefault("data_warnings", []).extend(warnings)
    return st.session_state["stated"], st.session_state.get("data_warnings", [])


def _events() -> list:
    if "events" not in st.session_state:
        events, warnings = load_events(_data_dir())
        st.session_state["events"] = events
        st.session_state.setdefault("data_warnings", []).extend(warnings)
    return st.session_state["events"]


def _menu_key() -> str:
    """Identifies the current scan for caching answers and events."""
    return (st.session_state.get("scan") or {}).get("key", "")


def _record_feedback(item: EnrichedItem, kind: str, venue: str, source: str) -> None:
    """Log a like or dislike.

    Runs as a widget *callback*, i.e. before the page body - the same rule as
    `_pick_sample`: it may write session state, but never a key a widget owns.
    The write is immediate rather than buffered to the end of the run, because a
    tap on a phone that then gets locked has to survive.
    """
    known = {event.name_key for event in _events()}
    event = event_from_item(item, kind, venue=venue, menu_source=source, known=known)
    warning = append_event(event, _data_dir())
    _events().append(event)
    if warning:
        st.session_state.setdefault("data_warnings", []).append(warning)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _pick_sample(name: str) -> None:
    """Select a sample image from the landing gallery.

    This runs as a widget *callback*, i.e. before the script body is re-run, so
    the sidebar selectbox that owns `sample_choice` has not been instantiated
    yet. Assigning the key from the page body instead raises
    StreamlitAPIException: "cannot be modified after the widget with key
    `sample_choice` is instantiated".
    """
    st.session_state.sample_choice = name


def _landing() -> None:
    st.markdown("# Menu Vision")
    st.caption(
        "A photo of a Chinese menu goes in; every line is classified as a dish or as "
        "decoration, then translated with its price, spice and allergens."
    )
    st.space("medium")
    st.subheader("Start from a sample", divider=False)
    st.caption("Or upload your own photo in the sidebar.")

    # Equal-height cards with the button pinned to the floor: the notes run one
    # to three lines, so without the stretch every "Scan this" sat at a
    # different height.
    columns = st.columns(len(SAMPLES), gap="medium")
    for column, (name, note) in zip(columns, SAMPLES.items()):
        path = PROJECT_DIR / name
        with column:
            with st.container(border=True, height="stretch", gap="xsmall"):
                if path.exists():
                    st.image(
                        thumbnail(path, background=_mat()),
                        width="stretch",
                        alt=f"Sample menu photo: {name}",
                    )
                st.markdown(f"**{name}**")
                st.caption(note)
                st.space("stretch")
                if path.exists():
                    st.button(
                        "Scan this",
                        key=f"pick_{name}",
                        icon=":material/photo_camera:",
                        width="stretch",
                        on_click=_pick_sample,
                        args=(name,),
                    )


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def _cap(text: str) -> str:
    """Sentence case for a chip label.

    The model returns categories and allergens lower-case ("rice bowl", "soy");
    chips read as labels rather than as prose, so the first letter is raised to
    match the hand-written chips next to them.
    """
    text = text.strip()
    return text[:1].upper() + text[1:] if text else text


def _price_caption(stats: dict) -> str:
    if not stats["priced"]:
        return "No prices printed on this menu"
    median = f"median ¥{stats['price_median']:g}"
    if stats["priced"] == 1:
        return f"Only one priced dish · ¥{stats['price_min']:g}"
    if stats["priced"] == stats["items"]:
        return f"All {_plural(stats['items'], 'dish', 'dishes')} are priced · {median}"
    return f"{stats['priced']} of {stats['items']} dishes are priced · {median}"


def _header(result: MenuResult, filename: str, source: str, image_bytes: bytes) -> None:
    stats = summary(result)

    left, right = st.columns([0.27, 0.73], gap="medium", vertical_alignment="top")
    with left:
        # A fixed-size preview keeps the header height the same whether the
        # input is a landscape menu or a tall phone screenshot. The full-size
        # image is in "Behind the scenes".
        st.image(
            thumbnail(
                image_bytes,
                _fit(_display_size(image_bytes) or result.image_size, 700, 420),
                background=_mat(),
            ),
            width="stretch",
            alt=f"The scanned photo, {filename}",
        )
    with right:
        st.markdown("## Menu Vision")
        # Say which layer read the photo: it explains both the timings and why
        # "Behind the scenes" may or may not have boxes to draw.
        if result.extractor == "vision":
            reader = f"vision model {result.llm_model or 'not called'}"
            reader_badge = "Vision model"
        else:
            reader = (
                f"OCR {result.ocr_engine or 'none'} + "
                f"{result.llm_model or 'no model'}"
            )
            reader_badge = "OCR + text model"
        with st.container(horizontal=True, gap="xsmall", wrap=True):
            st.badge(
                _plural(stats["items"], "dish", "dishes"),
                color="primary",
                icon=":material/restaurant:",
            )
            if stats["priced"]:
                span = (
                    f"¥{stats['price_min']:g}–{stats['price_max']:g}"
                    if stats["priced"] > 1
                    else f"¥{stats['price_min']:g}"
                )
            else:
                span = "no prices printed"
            st.badge(span, color="orange" if stats["priced"] else "gray")
            st.badge(reader_badge, color="gray", icon=":material/visibility:")
        st.caption(
            f"{filename} · {source} · {result.image_size[0]}×{result.image_size[1]}px · "
            f"{reader}"
        )
        st.caption(_price_caption(stats))
        # All warnings, not just failures: "a column is cut off at the edge" is
        # exactly what the reader needs to know before trusting these numbers.
        # The parser only emits a warning when data was actually dropped. One
        # callout instead of up to four stacked ones - the same sentences.
        if result.warnings:
            st.warning("\n\n".join(result.warnings[:4]), icon=":material/warning:")
        if st.button("Re-scan this image", icon=":material/refresh:", key="rescan"):
            st.session_state.pop("scan", None)
            st.rerun()

    # Metrics get their own full-width row: four of them squeezed into a split
    # column truncate to "Lines rej…" and "¥1…" on a 1400px screen.
    price_label, price_help = "no prices", "No price was printed on this menu."
    if stats["priced"]:
        price_label = (
            f"¥{stats['price_min']:g}–{stats['price_max']:g}"
            if stats["priced"] > 1
            else f"¥{stats['price_min']:g}"
        )
        price_help = (
            f"{stats['priced']} of {stats['items']} items carry a price; "
            f"median ¥{stats['price_median']:g}."
        )
    seconds = (stats["total_ms"] or 0) / 1000
    if stats["extractor"] == "vision":
        timing_help = (
            f"Image prep {result.metrics.get('encode_ms', 0):.0f} ms · vision call "
            f"{stats['vision_ms'] or 0:.0f} ms · one call reads and explains the board"
        )
    else:
        timing_help = (
            f"OCR {stats['ocr_ms'] or 0:.0f} ms · layout "
            f"{result.metrics.get('layout_ms', 0):.0f} ms · LLM "
            f"{stats['llm_ms'] or 0:.0f} ms"
        )
    a, b, c, d = st.columns(4)
    a.metric("Dishes", stats["items"], border=True, icon=":material/restaurant:")
    b.metric("Rejected", stats["rejected"], border=True, icon=":material/block:")
    c.metric("Prices", price_label, help=price_help, border=True, icon=":material/payments:")
    d.metric(
        "Scan time",
        f"{seconds:.0f}s" if seconds >= 10 else f"{seconds:.1f}s",
        help=timing_help,
        border=True,
        icon=":material/timer:",
    )


def _answer_body(
    item: EnrichedItem,
    profile: TasteProfile,
    reasons: tuple,
    scope: str,
    client: object | None,
) -> None:
    """Per-dish Q&A, cached so a rerun never re-asks the model.

    Called from inside the card's expander, so the keys are unchanged from when
    this was its own expander - moving it must not invalidate a cached answer.
    """
    key = f"q_{scope}_{item.id}"
    answer_key = (_menu_key(), item.id, scope)
    question = st.text_input(
        "Your question",
        key=key,
        placeholder="Is it spicy? Does it have pork? How big is it?",
        label_visibility="collapsed",
    )
    if st.button("Ask", key=f"ask_{scope}_{item.id}", icon=":material/send:"):
        if not question.strip():
            st.caption("Type a question first.")
        else:
            with st.spinner("Thinking"):
                st.session_state.setdefault("answers", {})[answer_key] = ask_about_dish(
                    item, question, profile,
                    client=client,
                    reasons=reasons,
                    language=Settings.from_env().target_language,
                )
    answer: Answer | None = st.session_state.get("answers", {}).get(answer_key)
    if answer is not None:
        st.write(answer.text)
        for note in answer.notes:
            st.caption(note)


def _details_expander(
    item: EnrichedItem,
    profile: TasteProfile | None,
    reasons: tuple,
    scope: str,
    client: object | None,
) -> None:
    """Everything that is reference rather than decision, behind one disclosure.

    Ingredients, the price sentence and the Q&A used to be three captions plus a
    second expander on every card. That is what made the grid read as a wall of
    text; nothing is dropped, it just stops competing with the name and price.
    """
    dish = item.dish
    if not (dish.ingredients or dish.allergens or dish.price_comment) and profile is None:
        return
    with st.expander("Details and questions", icon=":material/more_horiz:"):
        if dish.ingredients:
            st.caption("**Ingredients:** " + ", ".join(dish.ingredients))
        if dish.allergens:
            st.caption("**Allergens:** " + ", ".join(dish.allergens))
        if dish.price_comment:
            st.caption("**Price:** " + dish.price_comment)
        if profile is not None:
            _answer_body(item, profile, reasons, scope, client)


def _top_reason(reasons: tuple) -> object | None:
    """The most informative reason, for a one-line row.

    The budget line is the first reason for almost every dish, so quoting it here
    would put the same sentence under every row. Prefer something that actually
    distinguishes this dish.
    """
    if not reasons:
        return None
    distinguishing = [
        reason
        for reason in reasons
        if reason.polarity == "pro" and reason.component != "budget"
    ]
    return (distinguishing or list(reasons))[0]


def _feedback_buttons(item: EnrichedItem, scope: str, source: str) -> None:
    """Two small buttons wired to the event log.

    `on_click` keeps the write out of the page body: Streamlit forbids assigning
    a widget-owned key there, and this way the tap lands before the rerun. The
    keys are unchanged from the emoji-button version, so nothing already logged
    is lost.
    """
    venue = st.session_state.get("venue", "")
    st.button(
        "Liked it",
        key=f"like_{scope}_{item.id}",
        icon=":material/thumb_up:",
        type="tertiary",
        help="Remember that I liked this",
        on_click=_record_feedback, args=(item, "like", venue, source),
    )
    st.button(
        "Not again",
        key=f"dislike_{scope}_{item.id}",
        icon=":material/thumb_down:",
        type="tertiary",
        help="Remember that I did not",
        on_click=_record_feedback, args=(item, "dislike", venue, source),
    )


def _dish_card(
    item: EnrichedItem,
    profile: TasteProfile | None = None,
    reasons: tuple = (),
    *,
    scope: str = "all",
    source: str = "",
    client: object | None = None,
) -> None:
    dish = item.dish
    avoid = set(profile.avoid_allergens) if profile is not None else set()

    with st.container(border=True, height="stretch", gap="small"):
        head_left, head_right = st.columns([0.66, 0.34], vertical_alignment="top")
        with head_left:
            st.markdown(f"#### {item.cn}", anchors=False)
        with head_right:
            if item.price_display:
                st.markdown(
                    f"**{item.price_display}**",
                    text_alignment="right",
                    width="stretch",
                )
            else:
                with st.container(horizontal_alignment="right"):
                    st.badge("no price", color="gray")

        if dish.en:
            st.caption(dish.en)
        if dish.pinyin:
            # Plain caption rather than inline code: the code chip reads as
            # another badge and fights the allergen chips two lines below.
            st.caption(dish.pinyin)
        if dish.description:
            st.write(dish.description)

        # Gated on the package's own definition of "somebody described this
        # dish": a line the model never explained must not sprout a "spice 0/3"
        # chip, which would read as a claim about the food.
        if has_dish_facts(item):
            with st.container(horizontal=True, gap="xsmall", wrap=True):
                if item.section:
                    st.badge(_cap(item.section), color="gray")
                if dish.category:
                    # Blue rather than the accent: the brick red is reserved for
                    # counts, plan slots and the one actionable thing in a card.
                    # In dark mode brick and the allergen red are close enough
                    # that two `primary` chips read as two warnings.
                    st.badge(_cap(dish.category), color="blue")
                st.badge(
                    SPICE_CHIPS[dish.spice_level],
                    color=SPICE_COLOURS[dish.spice_level],
                    icon=":material/local_fire_department:",
                )
                if dish.vegetarian is True:
                    st.badge("Vegetarian", color="green", icon=":material/eco:")
                elif dish.vegetarian is False:
                    st.badge("Meat / fish", color="orange")
                for allergen in dish.allergens[:4]:
                    st.badge(
                        _cap(allergen),
                        # Grey unless it is on the diner's own avoid list: three
                        # orange chips in a row (spice, meat, allergen) read as
                        # one long warning and bury the one that matters.
                        color="red" if allergen in avoid else "gray",
                        icon=":material/warning:",
                        help=(
                            "On your avoid list"
                            if allergen in avoid
                            else "Contains this allergen"
                        ),
                    )
                if len(dish.allergens) > 4:
                    st.badge(f"+{len(dish.allergens) - 4} more", color="gray")

        st.space("stretch")

        if profile is not None:
            with st.container(horizontal=True, gap="xsmall", wrap=True):
                _feedback_buttons(item, scope, source)
        _details_expander(item, profile, reasons, scope, client)


def _tab_dishes(
    result: MenuResult,
    translate: bool,
    profile: TasteProfile,
    recs: Recommendation,
    client: object | None,
) -> None:
    if not result.items:
        st.info(
            "No menu lines were kept. If the photo is rotated, blurry or very small, "
            "try a straighter shot."
        )
        return
    if not translate:
        st.caption("Translation is off, so only the recognised text and price are shown.")

    # Reasons live on the ranked list; look them up by id so a card can explain
    # itself without re-scoring.
    by_id: dict[int, ScoredDish] = {dish.item_id: dish for dish in recs.shortlist}
    # Pairs per row, not one two-column flow: with a single column pair the two
    # stacks run independently and neighbouring cards end at different heights.
    for start in range(0, len(result.items), 2):
        row = st.columns(2, gap="medium")
        for column, item in zip(row, result.items[start:start + 2]):
            scored = by_id.get(item.id)
            with column:
                _dish_card(
                    item,
                    profile,
                    scored.reasons if scored else (),
                    scope="all",
                    source=result.source,
                    client=client,
                )


def _plan_card(slot, dish: ScoredDish | None = None) -> None:
    with st.container(border=True, gap="xsmall"):
        left, right = st.columns([0.7, 0.3], vertical_alignment="top")
        with left:
            st.badge(slot.slot.upper(), color="primary")
            st.markdown(f"#### {slot.cn}", anchors=False)
            if slot.en:
                st.caption(slot.en)
        with right:
            if slot.price_display:
                st.markdown(
                    f"**{slot.price_display}**", text_alignment="right", width="stretch"
                )
            else:
                with st.container(horizontal_alignment="right"):
                    st.badge("no price", color="gray")
        # The budget sentence is identical in every slot ("¥12 against your ¥220
        # for 4 - about ¥55 each") and the plan total is already in the caption
        # above, so it is dropped here. What is left distinguishes this dish from
        # the five beside it.
        why = (
            tuple(reason.text for reason in dish.reasons if reason.component != "budget")[:2]
            if dish is not None
            else slot.why[:1]
        )
        if why:
            st.caption(" · ".join(why))

def _shortlist_row(dish: ScoredDish, item: EnrichedItem, source: str) -> None:
    with st.container(border=True):
        left, middle, right = st.columns([0.46, 0.16, 0.38], vertical_alignment="center")
        with left:
            label = f"**{dish.cn}**"
            if dish.en:
                label += f" — {dish.en}"
            st.markdown(label)
            reason = _top_reason(dish.reasons)
            if reason is not None:
                st.caption(reason.text)
        with middle:
            if dish.price_display:
                st.markdown(
                    f"**{dish.price_display}**", text_alignment="right", width="stretch"
                )
            else:
                with st.container(horizontal_alignment="right"):
                    st.badge("no price", color="gray")
        with right:
            with st.container(horizontal=True, gap="xsmall", wrap=True):
                _feedback_buttons(item, "short", source)


def _tab_plan(
    result: MenuResult,
    profile: TasteProfile,
    recs: Recommendation,
    client: object | None,
) -> None:
    if not result.items:
        st.info("No dishes were read from this photo, so there is nothing to plan.")
        return

    plan = recs.plan
    st.subheader(
        f"Dinner for {profile.party_size}",
        icon=":material/groups:",
        divider=False,
    )
    if plan.slots:
        st.caption(budget_line(plan))

    total_label = f"¥{plan.total_known:g}" if plan.total_known is not None else "unknown"
    per_head = f"¥{plan.per_head:g}" if plan.per_head is not None else "—"
    a, b, c, d = st.columns(4)
    a.metric("To order", len(plan.slots), border=True, icon=":material/receipt_long:")
    b.metric("Total", total_label, border=True, icon=":material/payments:")
    c.metric("Per person", per_head, border=True, icon=":material/person:")
    d.metric(
        "Basis",
        "full" if recs.evidence >= 0.5 else "price only",
        border=True,
        icon=":material/fact_check:",
        help=(
            "How much of the ranking came from the menu's own descriptions. "
            "'Price only' means the dishes were never analysed, so nothing about "
            "ingredients, spice or allergens is known."
        ),
    )

    if recs.unknown_allergen_ids and profile.avoid_allergens:
        st.warning(
            "Some dishes in this plan list no allergens. That means unknown, not "
            "safe - ask the staff before ordering."
        )
    for note in plan.notes:
        st.caption(note)

    if not plan.slots:
        st.info(
            "Nothing could be planned from this menu. Loosen a restriction in the "
            "sidebar to see the dishes."
        )
        return

    scored_by_id = {dish.item_id: dish for dish in recs.shortlist}
    for slot in plan.slots:
        _plan_card(slot, scored_by_id.get(slot.item_id))

    by_id = {item.id: item for item in result.items}
    planned = {slot.item_id for slot in plan.slots}
    others = [dish for dish in recs.shortlist if dish.item_id not in planned]
    if others:
        st.subheader("If you want something else", divider=False)
        for dish in others[:6]:
            item = by_id.get(dish.item_id)
            if item is not None:
                _shortlist_row(dish, item, result.source)

    # Menu-wide questions. A text area rather than `st.chat_input`, which is
    # pinned to the bottom of the page and implies a conversation history this
    # app deliberately does not keep.
    st.subheader("Ask about this menu", icon=":material/forum:", divider=False)
    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        question = st.text_area(
            "Your question",
            key="menu_q",
            height=70,
            label_visibility="collapsed",
            placeholder="Which dish is the mildest? What would you skip for four people?",
        )
        ask = st.button("Ask", key="menu_ask", icon=":material/send:")
    menu_answer_key = ("menu", _menu_key())
    if ask:
        if question.strip():
            with st.spinner("Thinking"):
                st.session_state.setdefault("answers", {})[menu_answer_key] = ask_about_menu(
                    question, result, profile, recs,
                    client=client,
                    language=Settings.from_env().target_language,
                )
        else:
            st.caption("Type a question first.")
    menu_answer: Answer | None = st.session_state.get("answers", {}).get(menu_answer_key)
    if menu_answer is not None:
        st.write(menu_answer.text)


def _tab_table(result: MenuResult) -> None:
    frame = items_dataframe(result)
    if frame.empty:
        st.info("No items to tabulate.")
        return
    config: dict[str, object] = {}
    if "#" in frame.columns:
        config["#"] = st.column_config.NumberColumn(width="small")
    if "Price (CNY)" in frame.columns:
        config["Price (CNY)"] = st.column_config.NumberColumn(format="¥%.2f", width="small")
    if "Pinyin" in frame.columns:
        config["Pinyin"] = st.column_config.TextColumn(width="medium")
    # Descriptions stay in the CSV export but not on screen: nine columns do not
    # fit without horizontal scrolling, and the cards already show the prose.
    on_screen = [
        column
        for column in ("#", "Chinese", "Pinyin", "English", "Price (CNY)",
                       "Category", "Spice", "Vegetarian")
        if column in frame.columns
    ]
    st.dataframe(
        frame, hide_index=True, width="stretch",
        column_config=config, column_order=on_screen or None,
    )
    download_left, download_right = st.columns(2)
    download_left.download_button(
        "Download CSV",
        to_csv_bytes(result),
        file_name=f"{Path(result.source).stem or 'menu'}.csv",
        mime="text/csv",
        width="stretch",
        icon=":material/table_view:",
    )
    download_right.download_button(
        "Download JSON",
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{Path(result.source).stem or 'menu'}.json",
        mime="application/json",
        width="stretch",
        icon=":material/data_object:",
    )


def _tab_classification(result: MenuResult) -> None:
    stats = summary(result)
    total = stats["items"] + stats["rejected"]
    if stats["extractor"] == "vision":
        st.caption(
            f"The vision model read {_plural(stats['items'], 'dish', 'dishes')} off the "
            f"board and discarded {_plural(stats['rejected'], 'line')} it judged not to "
            "be food (headings, shop name, watermarks, app chrome)."
        )
    else:
        st.caption(
            f"OCR produced {_plural(total, 'candidate line')}. The classifier kept "
            f"{_plural(stats['items'], 'dish', 'dishes')} and rejected "
            f"{_plural(stats['rejected'], 'line')} as decoration, app chrome or watermarks."
        )
    if not result.llm_model or result.llm_model == "offline-stub":
        st.info(
            "No language model was used, so nothing was filtered: every line is listed "
            "as a dish. Enable translation in the sidebar to see the classifier at work."
        )

    # Rejected gets the wider half: its "reason" sentences are the long text here.
    kept_column, rejected_column = st.columns([0.4, 0.6], gap="medium")
    with kept_column:
        st.markdown(f"**Kept — {_plural(stats['items'], 'dish', 'dishes')}**")
        if result.items:
            # A bullet list rather than a table: three table columns in half a
            # page clipped the price, and a list never needs horizontal scroll.
            lines = []
            for item in result.items:
                right_side = item.dish.en or ""
                if item.price_display:
                    right_side = f"{right_side} · {item.price_display}" if right_side else item.price_display
                lines.append(f"- **{item.cn}** — {right_side}" if right_side else f"- **{item.cn}**")
            st.markdown("\n".join(lines))
        else:
            st.caption("Nothing kept.")
    with rejected_column:
        st.markdown(f"**Rejected — {_plural(stats['rejected'], 'line')}**")
        frame = rejected_dataframe(result)
        if frame.empty:
            st.caption("No line was rejected.")
        else:
            st.dataframe(
                frame,
                hide_index=True,
                width="stretch",
                column_config={
                    "Line": st.column_config.TextColumn(width="medium"),
                    "Reason": st.column_config.TextColumn(width="large"),
                },
            )

    categories = pd.Series(
        [item.dish.category.strip().lower() for item in result.items if item.dish.category.strip()]
    )
    counts = categories.value_counts()
    # One category is not a distribution - a single bar filling the chart says
    # nothing, so only draw this when there is something to compare.
    if len(counts) >= 2:
        st.markdown("**What kind of menu is this?**")
        st.bar_chart(counts, horizontal=True, height=min(60 + 28 * len(counts), 260))


def _tab_debug(result: MenuResult, image_bytes: bytes) -> None:
    stats = summary(result)
    if stats["tokens"]:
        st.caption(
            f"{stats['tokens']} tokens for {stats['items']} dishes · model {result.llm_model}"
        )

    if stats["extractor"] == "vision":
        st.caption(
            "The vision extractor returns text and prices, not coordinates, so there "
            "are no boxes to draw. Switch the reader to “OCR + text model” in the "
            "sidebar to see the overlay."
        )
    elif st.toggle("Show the OCR boxes", value=False, key="show_overlay"):
        legend_left, legend_right = st.columns([0.6, 0.4])
        with legend_left:
            st.image(draw_overlay(image_bytes, result), width="stretch")
        with legend_right:
            st.markdown(
                "Green boxes are lines kept as dishes, red boxes are lines the "
                "classifier rejected, and grey boxes never became a menu line "
                "(part of a longer name, or noise)."
            )
            st.caption(f"{len(result.boxes)} boxes passed the OCR confidence filter.")

    if stats["extractor"] == "vision":
        with st.expander("What the model transcribed"):
            st.caption(
                "The photo went to the model; this is the Chinese text and printed "
                "price it sent back. Prices were parsed here, not by the model."
            )
            st.json(result.vector or _rebuild_vector(result))
    else:
        with st.expander("Chinese vector sent to the model"):
            st.caption(
                "The model never sees the photo. It receives only this list of "
                "{id, Chinese name, price} and returns the translations."
            )
            st.json(result.vector or _rebuild_vector(result))

    with st.expander("Timings, tokens and warnings"):
        st.json(result.metrics)
        if result.warnings:
            for warning in result.warnings:
                st.warning(warning)

    with st.expander("Full JSON response"):
        st.json(result.model_dump())


def _rebuild_vector(result: MenuResult) -> list[dict[str, object]]:
    """The vector is only filled when requested; rebuild it for display."""
    return [
        {"id": item.id, "cn": item.cn, "price": item.price_raw,
         "price_value": item.price_value, "section": item.section}
        for item in result.items
    ]


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
EXTRACTORS: dict[str, str] = {
    "vision": "Vision",
    "ocr": "OCR + text",
}


def _settings_sidebar() -> tuple[bool, str]:
    settings = Settings.from_env()
    with st.sidebar:
        st.subheader("Reading options", icon=":material/tune:", divider=False)
        # A segmented control rather than a radio: two mutually exclusive
        # options belong side by side, and the long labels now live in the help.
        choice = st.segmented_control(
            "Reader",
            options=list(EXTRACTORS),
            default=settings.extractor if settings.extractor in EXTRACTORS else "vision",
            required=True,
            format_func=lambda key: EXTRACTORS[key],
            width="stretch",
            key="reader",
            label_visibility="collapsed",
            help=(
                "The vision model reads the board and explains it in one call. "
                "The OCR path runs easyocr first, which is slower and weaker on "
                "phone photos, but it keeps per-word boxes you can inspect."
            ),
        ) or settings.extractor
        translate = st.toggle(
            "Translate and explain dishes",
            value=True,
            help=(
                "Turns the explanations off: names and prices only, and much faster "
                "(with the vision reader the model just transcribes)."
            ),
        )
        with st.expander("Model and timings", icon=":material/memory:"):
            if settings.llm_api_key:
                model = settings.vision_model if choice == "vision" else settings.llm_model
                st.caption(f"`{model}` at `{settings.llm_base_url}`")
                st.caption(
                    f"Key configured ({settings.llm_api_key[-4:]}…), reasoning "
                    f"{settings.vision_thinking if choice == 'vision' else settings.llm_thinking}"
                    ", JSON mode "
                    f"{'on' if settings.llm_json_mode else 'off'}."
                )
            else:
                st.caption(
                    "No API key set — the vision reader needs one, so the scan falls "
                    "back to OCR with empty translations."
                )
            st.caption(
                "A vision scan takes about 20-70 s and is one call. The OCR path takes "
                "40-120 s of OCR plus 60-140 s of model time. Results are cached until "
                "you change the image or options."
            )
    return translate, choice


SPICE_LABELS = {0: "None", 1: "A little", 2: "Medium", 3: "Very spicy"}

# Card chips. `spice_level == 0` is the model's default, not a claim of
# mildness, so the chip stays grey and is only drawn when the dish was
# actually described (see has_dish_facts).
SPICE_CHIPS = {0: "Not spicy", 1: "Mild", 2: "Medium spicy", 3: "Very spicy"}
SPICE_COLOURS = {0: "gray", 1: "green", 2: "orange", 3: "red"}

COMMON_ALLERGENS = [
    "peanut", "tree nut", "sesame", "soy", "wheat", "gluten", "egg", "milk",
    "dairy", "fish", "shellfish", "msg",
]


def _table_sidebar() -> TasteProfile:
    """Party, budget, taste and restrictions - the four things only the user knows.

    Placed after the reader settings on purpose: the tests index toggles and
    selectboxes by position, and a new toggle above them would shift those.

    Values are seeded into session state before the widgets are created, which is
    how a saved profile pre-fills the form without assigning a widget-owned key
    from the page body.
    """
    stored, _warnings = _stated_profile()

    def seed(key: str, value: object) -> None:
        if key not in st.session_state:
            st.session_state[key] = value

    with st.sidebar:
        st.subheader("Your table", icon=":material/groups:", divider=False)
        seed("party_size", stored.party_size)
        # Integers, because number_input infers its type from min/max and a
        # seeded float against int bounds is a type clash.
        seed("budget_total", int(stored.budget_total or 0))
        seed("spice_tolerance", stored.spice_tolerance)
        seed("spice_learn", stored.spice_tolerance_source == "inferred")
        seed("veg_only", stored.vegetarian_only)
        seed("avoid", list(stored.avoid_allergens))
        seed("adventure", float(stored.adventurousness))
        seed("diet_notes", stored.dietary_notes)
        seed("venue_label", "")

        st.number_input("How many people?", 1, 12, key="party_size")
        st.number_input(
            "Total budget (CNY, 0 = no limit)", 0, 5000, step=10, key="budget_total",
            help="The whole table, not per person.",
        )
        st.select_slider(
            "Spice tolerance", options=[0, 1, 2, 3], key="spice_tolerance",
            format_func=lambda value: f"{SPICE_LABELS[value]} ({value}/3)",
        )
        st.checkbox(
            "Learn my spice level from what I like", key="spice_learn",
            help="Slowly adjusts the slider above from the dishes you react to.",
        )
        st.checkbox("Vegetarian only", key="veg_only")
        st.multiselect(
            "Avoid", sorted(set(COMMON_ALLERGENS)), key="avoid",
            help="Only the allergens the menu actually lists will hide a dish. "
                 "An unlisted allergen is treated as unknown, not safe.",
        )
        st.slider(
            "Adventurousness", 0.0, 1.0, key="adventure", step=0.1,
            help="0 keeps to dishes you have liked before; 1 pushes new ones up.",
        )
        st.text_area(
            "Anything else? (e.g. no pork, halal)", key="diet_notes", height=68,
            help="Passed to the assistant as context. Not turned into filters: a "
                 "wrong guess here is how someone gets served pork.",
        )
        st.text_input(
            "Restaurant name (optional)", key="venue_label",
            help="Keeps price memory attached to the right restaurant.",
        )

        stated = TasteProfile(
            party_size=int(st.session_state["party_size"]),
            budget_total=float(st.session_state["budget_total"]) or None,
            spice_tolerance=int(st.session_state["spice_tolerance"]),
            spice_tolerance_source="inferred" if st.session_state["spice_learn"] else "stated",
            vegetarian_only=bool(st.session_state["veg_only"]),
            avoid_allergens=tuple(st.session_state["avoid"]),
            adventurousness=float(st.session_state["adventure"]),
            dietary_notes=str(st.session_state["diet_notes"]).strip(),
        )
        if stated.model_dump(exclude={"updated_at"}) != stored.model_dump(
            exclude={"updated_at"}
        ):
            warning = save_profile(stated, _data_dir())
            st.session_state["stated"] = stated
            if warning:
                st.session_state.setdefault("data_warnings", []).append(warning)

        _memory_footer(stated)
    return stated


def _memory_footer(stated: TasteProfile) -> None:
    """What the app has learned, and a way to make it forget."""
    events = _events()
    derived = derive_profile(events, stated)
    st.subheader("Memory", icon=":material/history:", divider=False)
    if not events:
        st.caption(
            "Nothing learned yet. Tap the like or dislike buttons on a dish and "
            "this fills in."
        )
        return
    st.caption(f"{len(events)} reaction(s) remembered, across every menu you scan.")
    if derived.liked_ingredients:
        with st.container(horizontal=True, gap="xsmall", wrap=True):
            st.badge("likes", color="green", icon=":material/thumb_up:")
            for ingredient in derived.liked_ingredients[:5]:
                st.badge(ingredient, color="green")
    if derived.disliked_ingredients:
        with st.container(horizontal=True, gap="xsmall", wrap=True):
            st.badge("avoids", color="red", icon=":material/thumb_down:")
            for ingredient in derived.disliked_ingredients[:5]:
                st.badge(ingredient, color="red")
    if derived.inferred_spice is not None and stated.spice_tolerance_source == "stated":
        st.caption(f"Your taps suggest spice {derived.inferred_spice}/3.")
    if st.button("Forget what I liked", key="forget", icon=":material/delete:"):
        warning = clear_history(_data_dir())
        st.session_state["events"] = []
        st.session_state["answers"] = {}
        if warning:
            st.session_state.setdefault("data_warnings", []).append(warning)
        st.rerun()


def main() -> None:
    image_bytes, filename, source = _image_choice()
    translate, extractor = _settings_sidebar()
    stated = _table_sidebar()
    for warning in st.session_state.get("data_warnings", []):
        st.sidebar.warning(warning)

    if image_bytes is None:
        _landing()
        return

    result = _cached_scan(image_bytes, translate, extractor)

    # The venue tag is what keeps price memory attached to the right restaurant.
    st.session_state["venue"] = venue_key(result, st.session_state.get("venue_label", ""))

    events = _events()
    profile = derive_profile(events, stated)
    settings = Settings.from_env()
    recs = build_plan(
        result.items, profile, events,
        shortlist_size=settings.recommend_shortlist,
    )
    client = text_client(settings)

    _header(result, filename, source, image_bytes)
    st.space("medium")

    tabs = st.tabs(["Plan", "Dishes", "Price table", "Classification", "Behind the scenes"])
    with tabs[0]:
        _tab_plan(result, profile, recs, client)
    with tabs[1]:
        _tab_dishes(result, translate, profile, recs, client)
    with tabs[2]:
        _tab_table(result)
    with tabs[3]:
        _tab_classification(result)
    with tabs[4]:
        _tab_debug(result, image_bytes)


# Streamlit runs this file as __main__; the guard lets tests import the render
# helpers (and lets a checker reuse them) without executing the page.
if __name__ == "__main__":
    main()
