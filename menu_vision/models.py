"""Typed data structures for the menu-scanning pipeline.

The flow is: image -> OcrBox[] -> RawItem[] -> EnrichedItem[] -> MenuResult.

Every model is a pydantic model so the FastAPI layer (added later) can return
`MenuResult.model_dump()` straight from the endpoint.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# 0 = not spicy, 1 = mild, 2 = medium, 3 = very spicy.
SpiceLevel = Literal[0, 1, 2, 3]


class OcrBox(BaseModel):
    """One raw text detection coming out of the OCR engine.

    Coordinates are pixels in the (EXIF-corrected) image.
    """

    text: str
    confidence: float
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


class RawItem(BaseModel):
    """A candidate menu line after layout parsing, before any LLM call.

    `cn` and `price_raw` are exactly what OCR read — no translation yet.
    That Chinese pair is what we hand to the model.
    """

    cn: str
    price_raw: str | None = None
    price_value: float | None = None
    currency: str = "CNY"
    confidence: float = 0.0
    section: str | None = None
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    # English already printed on the board next to/below this dish, if any.
    # Free evidence: the restaurant chose that wording themselves.
    en_hint: str = ""

    def vector_entry(self, item_id: int) -> dict[str, Any]:
        """Compact record for the LLM payload: dish name + price, in Chinese."""
        entry: dict[str, Any] = {
            "id": item_id,
            "cn": self.cn,
            "price": self.price_raw,
            "price_value": self.price_value,
            "section": self.section,
        }
        if self.en_hint:
            entry["en_printed"] = self.en_hint
        return entry


class DishInfo(BaseModel):
    """Everything the LLM is asked to produce for one dish."""

    is_dish: bool = True
    pinyin: str = ""
    en: str = ""
    description: str = ""
    ingredients: list[str] = Field(default_factory=list)
    category: str = ""
    spice_level: SpiceLevel = 0
    vegetarian: bool | None = None
    allergens: list[str] = Field(default_factory=list)
    price_comment: str = ""
    confidence: float = 1.0
    # Filled when is_dish is False: why this line was discarded.
    reason: str = ""


class EnrichedItem(BaseModel):
    """A menu item: OCR ground truth + the model's reading of it."""

    id: int
    cn: str
    price_raw: str | None = None
    price_value: float | None = None
    currency: str = "CNY"
    price_display: str = ""
    section: str | None = None
    ocr_confidence: float = 0.0
    # English printed on the board for this dish (empty on Chinese-only menus).
    en_printed: str = ""
    dish: DishInfo = Field(default_factory=DishInfo)


class MenuResult(BaseModel):
    """Full result of scanning one image. This is the API response body."""
    source: str = ""
    image_size: tuple[int, int] = (0, 0)
    # Which layer read the photo: "vision" (one multimodal call) or "ocr"
    # (easyocr + layout + a text-only model). Callers need this to know whether
    # `boxes` can be drawn and whether `ocr_engine` means anything.
    extractor: str = "ocr"
    ocr_engine: str = ""
    llm_model: str = ""
    items: list[EnrichedItem] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    # The exact Chinese vector handed to the model (only filled on request).
    vector: list[dict[str, Any]] = Field(default_factory=list)
    # Raw OCR boxes after filtering (only filled on request). Lets a UI draw a
    # debug overlay without paying for a second OCR pass.
    boxes: list[dict[str, Any]] = Field(default_factory=list)
    # Lines the model rejected (app buttons, watermarks, store names...).
    dropped: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


def has_dish_facts(item: EnrichedItem) -> bool:
    """True when the model actually described this dish.

    Content-based on purpose. `confidence` cannot carry this: the OCR path keeps
    `DishInfo()`'s default of 1.0 for a dish nobody described, and the vision path
    keeps the model's own score even when `explain=False` returned no fields. And
    `spice_level == 0` is the default, not a claim that the dish is mild.

    Everything downstream keys off this. A dish without facts must never be
    described as vegetarian, mild, or free of an allergen - nobody looked.
    """
    dish = item.dish
    return bool((dish.category or "").strip() or (dish.en or "").strip()) and (
        dish.confidence > 0
    )

