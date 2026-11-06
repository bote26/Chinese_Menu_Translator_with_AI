"""Pipeline orchestration: image bytes in, structured translated menu out.

This module is the intended seam for the HTTP layer. A FastAPI handler should
be no more than:

    result = scan_menu(await file.read())
    return result.model_dump()

Two extractors live behind that one call, chosen by `Settings.extractor`:

    vision (default)  photo -> one multimodal call -> items
    ocr               photo -> easyocr -> geometric layout -> text model -> items

They converge on the same `EnrichedItem` list, so the API, the CLI and the UI
never branch on which one ran - only on `MenuResult.extractor` when they want to
show something extractor-specific (the OCR overlay).

`scan_menu` is synchronous and CPU/IO bound. Under a real server run it in a
threadpool (`run_in_threadpool`) or a worker so it does not block the event loop.
The OCR engine is cached per process, so keep one process warm per worker.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from .config import Settings
from .layout import parse_menu
from .llm import (
    LLMClient,
    LLMError,
    VisionClient,
    build_system_prompt,
    build_user_prompt,
    default_client,
    parse_dish_info,
    vision_client,
)
from .models import DishInfo, EnrichedItem, MenuResult
from .ocr import OcrEngine, extract_boxes
from .pricing import annotate_prices, format_price
from .vision import (
    build_vision_messages,
    build_vision_prompt,
    encode_image,
    parse_vision_rows,
)

_EXTRACTORS = {"vision", "ocr"}


def _batch(records: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    size = max(1, size)
    return [records[index : index + size] for index in range(0, len(records), size)]


def _resolve_extractor(
    requested: str | None, settings: Settings, client: Any
) -> tuple[str, list[str]]:
    """Pick the extractor, downgrading to OCR when vision cannot run.

    A missing API key is the one real blocker: OCR still produces names and
    prices offline, while a vision call with no key produces nothing. Failing
    over silently would be dishonest, so the reason is returned as a warning.
    """
    mode = (requested or settings.extractor or "vision").strip().lower()
    if mode not in _EXTRACTORS:
        return "vision", [f"Unknown extractor {mode!r}; using the vision model."]
    if mode == "vision" and client is None and not settings.llm_api_key:
        return "ocr", [
            "No MENU_LLM_API_KEY is configured, so the vision extractor cannot run: "
            "fell back to OCR with no translations."
        ]
    return mode, []


def scan_menu(
    image: bytes | str | Path,
    *,
    settings: Settings | None = None,
    client: LLMClient | VisionClient | None = None,
    engine: OcrEngine | None = None,
    extractor: str | None = None,
    enrich: bool = True,
    include_vector: bool = False,
    include_boxes: bool = False,
    progress: Callable[[str], None] | None = None,
) -> MenuResult:
    """Scan one menu image end to end.

    Args:
        image: raw bytes from an upload, or a path on disk.
        settings: injected config; defaults to environment variables.
        client: injected model client. For `extractor="vision"` it needs
            `chat_json_messages`; for `"ocr"` it needs `chat_json`. Defaults to
            the configured endpoint, or an offline stub when no key is present.
        engine: injected OCR engine; defaults to the process-wide easyocr one.
        extractor: "vision" or "ocr"; defaults to `Settings.extractor`.
        enrich: with the vision extractor, False asks only for names and prices
            (roughly 3x faster). With OCR it stops before the model call.
        include_vector: also return the Chinese `{id, cn, price}` records. On the
            OCR path these are what was sent to the model; on the vision path
            they are what the model transcribed.
        include_boxes: also return the filtered OCR boxes for an overlay. Only
            the OCR path can produce these - the vision path has no coordinates.
        progress: optional `callback(message)` invoked at each stage boundary.
            Long stages (OCR, LLM) take tens of seconds, so a UI can report
            what is happening instead of showing one frozen spinner.

    Returns:
        MenuResult, ready to serialize as the API response.
    """
    settings = settings or Settings.from_env()
    mode, notes = _resolve_extractor(extractor, settings, client)

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    if mode == "vision":
        result = _scan_vision(
            image,
            settings=settings,
            client=client,
            explain=enrich,
            include_vector=include_vector,
            progress=report,
        )
    else:
        result = _scan_ocr(
            image,
            settings=settings,
            client=client,
            engine=engine,
            enrich=enrich,
            include_vector=include_vector,
            include_boxes=include_boxes,
            progress=report,
        )
    if include_boxes and mode == "vision":
        notes.append(
            "OCR boxes are unavailable with the vision extractor: run it with "
            "extractor='ocr' to get the debug overlay."
        )
    for note in notes:
        if note not in result.warnings:
            result.warnings.append(note)
    return result


# --------------------------------------------------------------------------- #
# Vision extractor
# --------------------------------------------------------------------------- #
def _scan_vision(
    image: bytes | str | Path,
    *,
    settings: Settings,
    client: Any,
    explain: bool,
    include_vector: bool,
    progress: Callable[[str], None],
) -> MenuResult:
    client = client or vision_client(settings)
    model = getattr(client, "model", "") or settings.vision_model
    source = "<bytes>" if isinstance(image, (bytes, bytearray)) else str(image)

    progress("Preparing the photo")
    started = perf_counter()
    payload, image_size = encode_image(image, settings)
    encode_ms = (perf_counter() - started) * 1000

    result = MenuResult(
        source=source,
        image_size=image_size,
        extractor="vision",
        ocr_engine="",
        llm_model=model,
        metrics={
            "extractor": "vision",
            "explain": explain,
            "encode_ms": round(encode_ms, 1),
            "image_kb": round(len(payload) * 3 / 4 / 1024, 1),
        },
    )

    progress(
        f"Reading the menu with {model}" if explain else f"Transcribing with {model}"
    )
    started = perf_counter()
    try:
        rows = client.chat_json_messages(
            build_vision_messages(payload, build_vision_prompt(settings, explain=explain))
        )
    except LLMError as exc:
        # There is nothing to fall back on: the model *is* the reader. Say so
        # plainly instead of returning an empty menu that looks like a bad photo.
        result.warnings.append(f"Vision extraction failed: {exc}")
        result.metrics["total_ms"] = round(encode_ms + (perf_counter() - started) * 1000, 1)
        return result
    model_ms = (perf_counter() - started) * 1000

    usage = getattr(client, "last_usage", None) or {}
    if usage.get("total_tokens"):
        result.metrics["tokens"] = {
            key: int(usage.get(key) or 0)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens")
        }

    items, dropped, sections, warnings = parse_vision_rows(
        rows, currency=settings.currency, explain=explain
    )
    result.items = items
    result.dropped = dropped
    result.sections = sections
    result.warnings.extend(warnings)
    if not items:
        result.warnings.append(
            "The model found no dishes in this photo. Check that the menu is in "
            "frame and legible."
        )
    if include_vector:
        result.vector = [
            {
                "id": item.id,
                "cn": item.cn,
                "price": item.price_raw,
                "price_value": item.price_value,
                "section": item.section,
            }
            for item in items
        ]

    # Same arithmetic pass as the OCR path: category grouping is available here
    # only when the model explained the dishes, and it degrades to a flat
    # comparison when it did not.
    annotate_prices(result.items)

    result.metrics["vision_ms"] = round(model_ms, 1)
    result.metrics["items_returned"] = len(result.items)
    result.metrics["items_dropped"] = len(result.dropped)
    result.metrics["total_ms"] = round(encode_ms + model_ms, 1)
    return result


# --------------------------------------------------------------------------- #
# OCR extractor (the original path, kept for debugging and for boxes)
# --------------------------------------------------------------------------- #
def _scan_ocr(
    image: bytes | str | Path,
    *,
    settings: Settings,
    client: LLMClient | None,
    engine: OcrEngine | None,
    enrich: bool,
    include_vector: bool,
    include_boxes: bool,
    progress: Callable[[str], None],
) -> MenuResult:
    source = "<bytes>" if isinstance(image, (bytes, bytearray)) else str(image)

    progress("Reading image and running OCR")
    started = perf_counter()
    boxes, image_size, scale = extract_boxes(image, settings, engine)
    ocr_ms = (perf_counter() - started) * 1000

    progress(f"Grouping {len(boxes)} text boxes into menu lines")
    started = perf_counter()
    raw_items, sections, warnings = parse_menu(boxes, settings, currency=settings.currency)
    layout_ms = (perf_counter() - started) * 1000

    result = MenuResult(
        source=source,
        image_size=image_size,
        extractor="ocr",
        ocr_engine=getattr(engine, "name", None) or "easyocr",
        sections=sections,
        warnings=list(warnings),
        metrics={
            "extractor": "ocr",
            "ocr_boxes": len(boxes),
            "candidates": len(raw_items),
            "ocr_ms": round(ocr_ms, 1),
            "layout_ms": round(layout_ms, 1),
            "upscale": round(scale, 2),
        },
    )
    if include_boxes:
        result.boxes = [box.model_dump() for box in boxes]

    if not raw_items:
        return result

    # ---- LLM enrichment --------------------------------------------------
    records = [item.vector_entry(index + 1) for index, item in enumerate(raw_items)]
    if include_vector:
        result.vector = records
    by_id: dict[int, DishInfo] = {}
    llm_ms = 0.0
    tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if enrich:
        client = client or default_client(settings)
        result.llm_model = getattr(client, "model", "unknown")
        if getattr(client, "model", "") == "offline-stub":
            result.warnings.append(
                "No LLM API key configured: translations and dish context are empty. "
                "Set MENU_LLM_API_KEY to enable them."
            )
        system = build_system_prompt(settings)
        chunks = _batch(records, settings.llm_batch_size)
        started = perf_counter()
        for index, chunk in enumerate(chunks, start=1):
            progress(
                f"Translating and describing {len(chunk)} items with {result.llm_model}"
                + (f" (batch {index}/{len(chunks)})" if len(chunks) > 1 else "")
            )
            try:
                replies = client.chat_json(system, build_user_prompt(chunk))
            except LLMError as exc:
                # Partial results beat a failed request: keep OCR + layout for
                # the batches already done and tell the caller what happened.
                result.warnings.append(f"LLM enrichment failed: {exc}")
                break
            usage = getattr(client, "last_usage", None) or {}
            for field in tokens:
                tokens[field] += int(usage.get(field) or 0)
            for raw in replies if isinstance(replies, list) else []:
                if not isinstance(raw, dict):
                    continue
                try:
                    item_id = int(raw.get("id"))
                except (TypeError, ValueError):
                    continue
                by_id[item_id] = parse_dish_info(raw)
        llm_ms = (perf_counter() - started) * 1000
        result.metrics["llm_ms"] = round(llm_ms, 1)
        if tokens["total_tokens"]:
            result.metrics["tokens"] = tokens

    for index, item in enumerate(raw_items, start=1):
        info = by_id.get(index) or DishInfo()
        if enrich and not info.is_dish:
            result.dropped.append(
                {
                    "cn": item.cn,
                    "price": item.price_raw,
                    "reason": info.reason or "not a menu item",
                }
            )
            continue
        result.items.append(
            EnrichedItem(
                id=index,
                cn=item.cn,
                price_raw=item.price_raw,
                price_value=item.price_value,
                currency=item.currency,
                price_display=format_price(item.price_value, item.currency),
                section=item.section,
                ocr_confidence=item.confidence,
                en_printed=item.en_hint,
                dish=info,
            )
        )

    # Price comparisons are arithmetic: computed here from the parsed numbers so
    # the sentence can never contradict `price_value`, and the same price always
    # gets the same description.
    annotate_prices(result.items)

    result.metrics["items_returned"] = len(result.items)
    result.metrics["items_dropped"] = len(result.dropped)
    result.metrics["total_ms"] = round(
        result.metrics["ocr_ms"] + result.metrics["layout_ms"] + llm_ms, 1
    )

    if enrich and not result.items:
        result.warnings.append("All detected lines were rejected as non-menu text.")
    return result
