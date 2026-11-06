"""Tests for the presentation helpers (overlay, tables, summary).

These build a MenuResult by hand, so they run in milliseconds and need neither
easyocr nor a language model.
"""

from __future__ import annotations

import io
import unittest

from PIL import Image

from menu_vision.models import DishInfo, EnrichedItem, MenuResult
from menu_vision.report import (
    thumbnail,
    COLOUR_KEPT,
    COLOUR_NEUTRAL,
    COLOUR_REJECTED,
    draw_overlay,
    items_dataframe,
    rejected_dataframe,
    summary,
    to_csv_bytes,
)


def _blank_png(width: int = 200, height: int = 120) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def _result(*, with_translation: bool = True) -> MenuResult:
    return MenuResult(
        source="synthetic.jpg",
        image_size=(200, 120),
        ocr_engine="fake",
        llm_model="fake-model" if with_translation else "offline-stub",
        items=[
            EnrichedItem(
                id=1,
                cn="酸辣土豆丝",
                price_raw="11元",
                price_value=11.0,
                price_display="¥11",
                dish=DishInfo(
                    en="Hot and Sour Shredded Potato" if with_translation else "",
                    pinyin="suān là tǔ dòu sī" if with_translation else "",
                    description="Shredded potato with vinegar and chilli."
                    if with_translation
                    else "",
                    category="vegetable" if with_translation else "",
                    spice_level=2,
                    vegetarian=True,
                ),
            ),
            EnrichedItem(
                id=2,
                cn="香干回锅肉饭",
                price_raw="15元",
                price_value=15.0,
                price_display="¥15",
                dish=DishInfo(
                    en="Twice-cooked Pork Rice" if with_translation else "",
                    category="rice bowl" if with_translation else "",
                    vegetarian=False,
                ),
            ),
        ],
        dropped=[{"cn": "销量96", "price": None, "reason": "sales statistic"}],
        metrics={
            "ocr_ms": 1000.0,
            "layout_ms": 20.0,
            "llm_ms": 500.0,
            "total_ms": 1520.0,
            "tokens": {"prompt_tokens": 400, "completion_tokens": 100, "total_tokens": 500},
        },
        boxes=[
            {"text": "酸辣土豆丝", "x0": 10, "y0": 10, "x1": 90, "y1": 30},
            {"text": "销量96", "x0": 10, "y0": 50, "x1": 90, "y1": 70},
            {"text": "nipic", "x0": 10, "y0": 90, "x1": 90, "y1": 110},
        ],
    )


class SummaryTests(unittest.TestCase):
    def test_counts_prices_and_tokens(self):
        stats = summary(_result())
        self.assertEqual(stats["items"], 2)
        self.assertEqual(stats["rejected"], 1)
        self.assertEqual(stats["priced"], 2)
        self.assertEqual(stats["price_min"], 11.0)
        self.assertEqual(stats["price_max"], 15.0)
        self.assertEqual(stats["price_median"], 13.0)
        self.assertEqual(stats["tokens"], 500)

    def test_no_prices_gives_none_not_zero(self):
        result = _result()
        for item in result.items:
            item.price_value = None
            item.price_display = ""
        stats = summary(result)
        self.assertEqual(stats["priced"], 0)
        self.assertIsNone(stats["price_min"])
        self.assertIsNone(stats["price_median"])


class FrameTests(unittest.TestCase):
    def test_items_frame_keeps_dish_and_price(self):
        frame = items_dataframe(_result())
        self.assertEqual(len(frame), 2)
        self.assertIn("Chinese", frame.columns)
        self.assertIn("Price (CNY)", frame.columns)
        self.assertEqual(frame.loc[0, "English"], "Hot and Sour Shredded Potato")
        self.assertEqual(frame.loc[1, "Vegetarian"], "no")

    def test_empty_translation_columns_are_dropped(self):
        """With no LLM run the table should not show four blank columns."""
        frame = items_dataframe(_result(with_translation=False))
        self.assertIn("Chinese", frame.columns)
        self.assertNotIn("English", frame.columns)
        self.assertNotIn("Pinyin", frame.columns)
        self.assertNotIn("Category", frame.columns)

    def test_rejected_frame(self):
        frame = rejected_dataframe(_result())
        self.assertEqual(frame.loc[0, "Line"], "销量96")
        self.assertEqual(frame.loc[0, "Reason"], "sales statistic")

    def test_csv_round_trip_preserves_chinese(self):
        text = to_csv_bytes(_result()).decode("utf-8-sig")
        self.assertIn("酸辣土豆丝", text)
        self.assertIn("15.0", text)

    def test_empty_result_does_not_crash(self):
        empty = MenuResult()
        self.assertTrue(items_dataframe(empty).empty)
        self.assertTrue(rejected_dataframe(empty).empty)
        self.assertEqual(summary(empty)["items"], 0)


class OverlayTests(unittest.TestCase):
    def test_box_colours_follow_the_classification(self):
        image = draw_overlay(_blank_png(), _result())
        self.assertEqual(image.size, (200, 120))
        self.assertEqual(image.getpixel((10, 10)), COLOUR_KEPT)
        self.assertEqual(image.getpixel((10, 50)), COLOUR_REJECTED)
        self.assertEqual(image.getpixel((10, 90)), COLOUR_NEUTRAL)

    def test_interiors_are_left_untouched(self):
        image = draw_overlay(_blank_png(), _result())
        self.assertEqual(image.getpixel((50, 20)), (255, 255, 255))

    def test_works_with_no_boxes(self):
        result = _result()
        result.boxes = []
        self.assertEqual(draw_overlay(_blank_png(), result).size, (200, 120))


class ThumbnailTests(unittest.TestCase):
    @staticmethod
    def _png(width: int, height: int) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (width, height), (200, 40, 40)).save(buffer, format="PNG")
        return buffer.getvalue()

    def test_portrait_and_landscape_get_the_same_canvas(self):
        box = (120, 90)
        tall = thumbnail(self._png(40, 400), box)
        wide = thumbnail(self._png(400, 40), box)
        self.assertEqual(tall.size, box)
        self.assertEqual(wide.size, box)

    def test_art_is_preserved_and_padding_is_transparent(self):
        image = thumbnail(self._png(40, 400), (120, 90))
        self.assertEqual(image.mode, "RGBA")
        # A 40x400 source fitted into 120x90 becomes ~9x90, centred.
        self.assertEqual(image.getpixel((60, 45))[:3], (200, 40, 40))
        self.assertEqual(image.getpixel((0, 0))[3], 0)


if __name__ == "__main__":
    unittest.main()
