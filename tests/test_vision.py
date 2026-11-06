"""Tests for the vision extractor: prompt shape, row validation, image prep.

No network: the model is faked, so these run in a second and fail loudly when the
prompt/schema contract drifts. The real-call behaviour is measured separately in
the work log (four boards, timings and accuracy per board).
"""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from menu_vision.config import Settings  # noqa: E402
from menu_vision.llm import LLMError, _extract_json  # noqa: E402
from menu_vision.models import OcrBox  # noqa: E402
from menu_vision.pipeline import scan_menu  # noqa: E402
from menu_vision.vision import (  # noqa: E402
    build_vision_messages,
    build_vision_prompt,
    encode_image,
    parse_vision_rows,
)


def _jpeg_bytes(size: tuple[int, int] = (64, 64), colour: str = "white") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeVisionClient:
    """Stands in for the multimodal endpoint."""

    model = "fake-vision"

    def __init__(self, rows, error: Exception | None = None, usage=None) -> None:
        self.rows = rows
        self.error = error
        self.calls: list[list[dict]] = []
        self.last_usage = usage or {}

    def chat_json_messages(self, messages: list[dict]):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.rows


class FakeEngine:
    """Two-box OCR engine, so the fallback path needs no easyocr weights."""

    name = "fake-ocr"

    def read(self, image_bgr) -> list[OcrBox]:
        return [
            OcrBox(text="酸辣土豆丝", confidence=0.95, x0=40, y0=40, x1=300, y1=90),
            OcrBox(text="11元", confidence=0.92, x0=330, y0=42, x1=400, y1=88),
        ]


ROW = {
    "cn": "红油麻辣抄手",
    "price": "16元14个",
    "section": "抄手",
    "en_printed": "Spicy Red Oil Wontons",
    "pinyin": "hóng yóu má là chāo shǒu",
    "en": "Spicy Red Oil Wontons",
    "description": "Pork wontons in chilli oil and Sichuan pepper.",
    "ingredients": ["pork", "chilli oil", "Sichuan pepper"],
    "category": "wonton",
    "spice_level": 2,
    "vegetarian": False,
    "allergens": ["wheat", "soy"],
    "confidence": 0.9,
}


class TestParseRows(unittest.TestCase):
    def test_parses_printed_units_deterministically(self):
        rows = [
            {**ROW, "price": "16元14个"},
            {**ROW, "cn": "金牌脆皮烧鹅", "price": "88例"},
            {**ROW, "cn": "白切鸡", "price": "50元半只"},
            {**ROW, "cn": "密汁排骨", "price": "18.5元"},
        ]
        items, dropped, _, warnings = parse_vision_rows(rows)
        self.assertEqual(
            [(item.cn, item.price_value, item.price_display) for item in items],
            [
                ("红油麻辣抄手", 16.0, "¥16"),
                ("金牌脆皮烧鹅", 88.0, "¥88"),
                ("白切鸡", 50.0, "¥50"),
                ("密汁排骨", 18.5, "¥18.5"),
            ],
        )
        self.assertFalse(dropped)
        self.assertFalse(warnings)

    def test_missing_price_is_kept_as_an_item(self):
        items, _, _, _ = parse_vision_rows([{**ROW, "price": None}])
        self.assertEqual(len(items), 1)
        self.assertIsNone(items[0].price_value)
        self.assertEqual(items[0].price_display, "")

    def test_unreadable_price_is_kept_but_flagged(self):
        items, _, _, warnings = parse_vision_rows([{**ROW, "price": "时价"}])
        self.assertEqual(len(items), 1)
        self.assertIsNone(items[0].price_value)
        self.assertEqual(items[0].price_raw, "时价")
        self.assertTrue(any("does not understand" in warning for warning in warnings))

    def test_nameless_row_is_dropped(self):
        """This is the guard against the cropped-column hallucination."""
        items, dropped, _, _ = parse_vision_rows([{**ROW, "cn": "", "price": "28元/份"}])
        self.assertEqual(items, [])
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["price"], "28元/份")
        self.assertIn("no readable dish name", dropped[0]["reason"])

    def test_latin_only_row_is_dropped(self):
        items, dropped, _, _ = parse_vision_rows([{**ROW, "cn": "Spicy Wontons"}])
        self.assertFalse(items)
        self.assertEqual(len(dropped), 1)

    def test_model_flagged_non_dish_is_dropped_with_reason(self):
        rows = [{**ROW, "is_dish": False, "reason": "section heading"}]
        items, dropped, _, _ = parse_vision_rows(rows)
        self.assertFalse(items)
        self.assertEqual(dropped[0]["reason"], "section heading")

    def test_sections_keep_first_seen_order_without_duplicates(self):
        rows = [
            {**ROW, "cn": "卤水鸭饭", "section": "卤味套餐"},
            {**ROW, "cn": "金牌脆皮烧鹅", "section": "烧味例牌"},
            {**ROW, "cn": "卤水鸡腿饭", "section": "卤味套餐"},
            {**ROW, "cn": "卤蛋", "section": ""},
        ]
        _, _, sections, _ = parse_vision_rows(rows)
        self.assertEqual(sections, ["卤味套餐", "烧味例牌"])

    def test_printed_english_is_preserved_verbatim(self):
        items, _, _, _ = parse_vision_rows([ROW])
        self.assertEqual(items[0].en_printed, "Spicy Red Oil Wontons")

    def test_transcribe_only_leaves_explanations_empty(self):
        items, _, _, _ = parse_vision_rows([ROW], explain=False)
        item = items[0]
        self.assertEqual(item.price_value, 16.0)
        self.assertEqual(item.dish.en, "")
        self.assertEqual(item.dish.description, "")
        self.assertEqual(item.dish.category, "")

    def test_spice_and_confidence_are_clamped(self):
        items, _, _, _ = parse_vision_rows(
            [{**ROW, "spice_level": 9, "confidence": 3}]
        )
        self.assertEqual(items[0].dish.spice_level, 3)
        self.assertEqual(items[0].dish.confidence, 1.0)

    def test_non_list_reply_is_empty_not_a_crash(self):
        items, dropped, sections, _ = parse_vision_rows({"error": "no"})
        self.assertEqual((items, dropped, sections), ([], [], []))


class TestPrompts(unittest.TestCase):
    def test_prompt_carries_target_language_and_price_examples(self):
        prompt = build_vision_prompt(Settings(target_language="Spanish"))
        self.assertIn("Spanish", prompt)
        # The unit examples are what stop the model normalising "88例" to "88".
        self.assertIn("88例", prompt)
        self.assertIn("50元半只", prompt)
        self.assertIn("en_printed", prompt)

    def test_transcribe_prompt_omits_explanations(self):
        prompt = build_vision_prompt(Settings(), explain=False)
        self.assertNotIn("description", prompt)
        self.assertIn("price", prompt)

    def test_messages_attach_the_image_before_the_text(self):
        messages = build_vision_messages("QUJD", "read this")
        self.assertEqual(len(messages), 1)
        parts = messages[0]["content"]
        self.assertEqual(parts[0]["type"], "image_url")
        self.assertEqual(parts[0]["image_url"]["url"], "data:image/jpeg;base64,QUJD")
        self.assertEqual(parts[1], {"type": "text", "text": "read this"})


class TestImagePrep(unittest.TestCase):
    def test_reencodes_to_jpeg_and_reports_size(self):
        payload, size = encode_image(_jpeg_bytes((300, 200)), Settings())
        self.assertEqual(size, (300, 200))
        decoded = __import__("base64").b64decode(payload)
        self.assertTrue(decoded.startswith(b"\xff\xd8"), "must stay a JPEG")

    def test_applies_exif_rotation(self):
        """A phone photo flagged sideways must arrive upright."""
        buffer = io.BytesIO()
        art = Image.new("RGB", (40, 20), "white")
        exif = art.getexif()
        exif[0x0112] = 6  # rotate 90 deg clockwise
        art.save(buffer, format="JPEG", exif=exif)
        _, size = encode_image(buffer.getvalue(), Settings())
        self.assertEqual(size, (20, 40))

    def test_downscales_only_above_the_cap(self):
        _, size = encode_image(_jpeg_bytes((4000, 100)), Settings(vision_max_side=1000))
        self.assertEqual(size, (1000, 25))


class TestVisionPipeline(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(llm_api_key="test-key", llm_base_url="http://invalid")
        self.image = _jpeg_bytes()

    def test_scan_returns_items_prices_and_price_comments(self):
        client = FakeVisionClient(
            [
                {**ROW, "cn": "金牌脆皮烧鹅", "price": "88例", "category": "roast meat"},
                {**ROW, "cn": "密汁排骨", "price": "68例", "category": "roast meat"},
                {**ROW, "cn": "酥皮烧腩", "price": "68例", "category": "roast meat"},
                {**ROW, "cn": "脆皮烧鸭", "price": "28例", "category": "roast meat"},
                {**ROW, "cn": "卤蛋", "price": None, "category": "snack"},
            ],
            usage={"prompt_tokens": 3000, "completion_tokens": 1200,
                   "total_tokens": 4200, "reasoning_tokens": 400},
        )
        result = scan_menu(self.image, settings=self.settings, client=client)

        self.assertEqual(result.extractor, "vision")
        self.assertEqual(result.llm_model, "fake-vision")
        self.assertEqual(len(result.items), 5)
        self.assertEqual([item.id for item in result.items], [1, 2, 3, 4, 5])
        self.assertEqual(result.items[0].price_display, "¥88")
        self.assertEqual(result.items[0].dish.en, "Spicy Red Oil Wontons")
        self.assertEqual(result.items[0].en_printed, "Spicy Red Oil Wontons")
        self.assertEqual(result.metrics["tokens"]["reasoning_tokens"], 400)
        self.assertIn("vision_ms", result.metrics)
        # Price commentary is computed, never taken from the model.
        self.assertIn("cheapest", result.items[3].dish.price_comment)
        self.assertIn("most expensive", result.items[0].dish.price_comment)

    def test_scan_sends_one_multimodal_request(self):
        client = FakeVisionClient([dict(ROW)])
        scan_menu(self.image, settings=self.settings, client=client)
        self.assertEqual(len(client.calls), 1)
        parts = client.calls[0][0]["content"]
        self.assertEqual(parts[0]["type"], "image_url")
        self.assertTrue(parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_translate_off_uses_the_transcribe_prompt(self):
        client = FakeVisionClient([dict(ROW)])
        result = scan_menu(self.image, settings=self.settings, client=client, enrich=False)
        text = client.calls[0][0]["content"][1]["text"]
        self.assertNotIn("description", text)
        self.assertEqual(result.items[0].dish.en, "")
        self.assertEqual(result.items[0].price_value, 16.0)

    def test_vector_is_the_transcription(self):
        client = FakeVisionClient([dict(ROW)])
        result = scan_menu(
            self.image, settings=self.settings, client=client, include_vector=True
        )
        self.assertEqual(
            result.vector,
            [{"id": 1, "cn": "红油麻辣抄手", "price": "16元14个",
              "price_value": 16.0, "section": "抄手"}],
        )

    def test_model_failure_is_reported_not_swallowed(self):
        client = FakeVisionClient([], error=LLMError("upstream 502"))
        result = scan_menu(self.image, settings=self.settings, client=client)
        self.assertFalse(result.items)
        self.assertTrue(any("Vision extraction failed" in w for w in result.warnings))

    def test_boxes_are_not_promised_on_this_path(self):
        client = FakeVisionClient([dict(ROW)])
        result = scan_menu(
            self.image, settings=self.settings, client=client, include_boxes=True
        )
        self.assertEqual(result.boxes, [])
        self.assertTrue(any("extractor='ocr'" in w for w in result.warnings))


class TestFallback(unittest.TestCase):
    def test_no_key_falls_back_to_ocr_with_a_warning(self):
        settings = Settings(llm_api_key="", extractor="vision")
        result = scan_menu(
            _jpeg_bytes((400, 120)), settings=settings, engine=FakeEngine(), enrich=False
        )
        self.assertEqual(result.extractor, "ocr")
        self.assertTrue(any("No MENU_LLM_API_KEY" in w for w in result.warnings))
        self.assertEqual(result.items[0].cn, "酸辣土豆丝")
        self.assertEqual(result.items[0].price_value, 11.0)

    def test_unknown_extractor_name_degrades_to_vision(self):
        client = FakeVisionClient([dict(ROW)])
        result = scan_menu(
            _jpeg_bytes(), settings=Settings(llm_api_key="k"),
            client=client, extractor="tesseract",
        )
        self.assertEqual(result.extractor, "vision")
        self.assertTrue(any("Unknown extractor" in w for w in result.warnings))


class TestTruncatedReply(unittest.TestCase):
    def test_salvages_a_reply_cut_off_mid_array(self):
        """A 20-dish board can hit the output limit; the earlier rows still count."""
        text = '[{"cn": "卤蛋"}, {"cn": "卤豆腐"}, {"cn": "叉烧", "description": "Hone'
        rows = _extract_json(text)
        self.assertEqual(rows, [{"cn": "卤蛋"}, {"cn": "卤豆腐"}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
