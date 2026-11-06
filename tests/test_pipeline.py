"""Regression tests for the menu pipeline.

Run with the interpreter that has easyocr installed:

    python -m unittest discover -s tests -v

The layout tests are pure geometry (fast, no OCR). One end-to-end test runs real
OCR on test.jpg, which takes ~20 s on CPU.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from menu_vision.config import Settings  # noqa: E402
from menu_vision.layout import group_rows, parse_menu, parse_price, split_name_and_price  # noqa: E402
from menu_vision.models import OcrBox  # noqa: E402
from menu_vision.pipeline import scan_menu  # noqa: E402


def box(text: str, x0: int, y0: int, x1: int, y1: int, confidence: float = 0.95) -> OcrBox:
    return OcrBox(text=text, confidence=confidence, x0=x0, y0=y0, x1=x1, y1=y1)


class TestPriceParsing(unittest.TestCase):
    def test_accepts_common_price_shapes(self):
        cases = {
            "11元": 11.0,
            "15 元": 15.0,
            "¥7.65": 7.65,
            "￥12": 12.0,
            "18": 18.0,
            "12.5块": 12.5,
            "RMB 20": 20.0,
            "30元起": 30.0,
            "44/份": 44.0,
            # Cantonese roast shops price by portion or per half bird, with no 元
            # at all. These used to parse as dish *names* ("88例"), so every such
            # row became a phantom dish with no price.
            "88例": 88.0,
            "68例": 68.0,
            "50元半只": 50.0,
            "28元/份": 28.0,
            # Per-unit prices that also print how many units: "12元/15个" is
            # 12 yuan for 15 pieces (small wontons on a bilingual board).
            "16元/15个": 16.0,
            "16元14个": 16.0,
            "12元/15个": 12.0,
            "8元/笼": 8.0,
            "8元/7个": 8.0,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                result = parse_price(text)
                self.assertIsNotNone(result, f"{text!r} should parse as a price")
                self.assertEqual(result[0], expected)

    def test_repairs_currency_symbol_confused_with_cjk(self):
        # Observed on a phone screenshot: "¥7.65" read as "苹7.65" at conf 0.07.
        self.assertEqual(parse_price("苹7.65")[0], 7.65)

    def test_rejects_text_that_is_not_a_price(self):
        for text in ["销量96", "1条评价", "酸辣土豆丝", "第12号", "11元12元", ""]:
            with self.subTest(text=text):
                self.assertIsNone(parse_price(text))


class TestNameAndPriceSplit(unittest.TestCase):
    def test_splits_glued_name_and_price(self):
        result = split_name_and_price("酸辣土豆丝 11元")
        self.assertIsNotNone(result)
        name, value, raw = result
        self.assertEqual(name, "酸辣土豆丝")
        self.assertEqual(value, 11.0)
        self.assertIn("11", raw)

    def test_leaves_plain_name_alone(self):
        self.assertIsNone(split_name_and_price("超浓水牛奶双皮奶"))


class TestRowGrouping(unittest.TestCase):
    def test_does_not_chain_adjacent_rows_into_one(self):
        """A tall box used to stretch the row span and absorb the next row.

        Regression: 酸菜肥肠饭 / 青椒猪肝饭 landed in one row, so one of the two
        prices was left unassigned.
        """
        boxes = [
            box("酸菜肥肠饭", 556, 300, 690, 332),
            box("15元", 702, 302, 754, 332),
            box("青椒猪肝饭", 556, 336, 690, 368),
            box("15元", 702, 338, 752, 368),
        ]
        rows = group_rows(boxes, 0.55)
        self.assertEqual(len(rows), 2, f"expected 2 rows, got {[[b.text for b in r] for r in rows]}")

    def test_pairs_each_name_with_price_on_its_right(self):
        settings = Settings()
        boxes = [
            box("酸辣土豆丝", 342, 120, 458, 150),
            box("11元", 468, 120, 522, 150),
            box("香干回锅肉饭", 556, 120, 688, 150),
            box("15元", 702, 122, 750, 148),
        ]
        items, _, _ = parse_menu(boxes, settings)
        self.assertEqual([(i.cn, i.price_value) for i in items],
                         [("酸辣土豆丝", 11.0), ("香干回锅肉饭", 15.0)])

    def test_attaches_price_printed_above_the_name(self):
        """Delivery-app product cards show the price above the dish name."""
        settings = Settings()
        boxes = [
            box("苹7.65", 45, 890, 206, 957, confidence=0.07),
            box("超浓水牛奶双皮奶", 49, 995, 468, 1063),
            box("销量96", 51, 1098, 160, 1140),
        ]
        items, _, _ = parse_menu(boxes, settings)
        by_name = {item.cn: item for item in items}
        self.assertEqual(by_name["超浓水牛奶双皮奶"].price_value, 7.65)
        self.assertIsNone(by_name["销量96"].price_value, "sold-count must not become a price")

    def test_keeps_items_without_prices(self):
        """A menu with no prices must not lose items or invent numbers."""
        settings = Settings()
        boxes = [box("羊肉串", 132, 248, 374, 350), box("鸡菌", 524, 248, 696, 350)]
        items, _, warnings = parse_menu(boxes, settings)
        self.assertEqual([i.cn for i in items], ["羊肉串", "鸡菌"])
        self.assertTrue(all(i.price_value is None for i in items))
        self.assertTrue(any("No prices found" in w for w in warnings))

    def test_ignores_section_header(self):
        settings = Settings()
        boxes = [
            box("新疆美味烧烤", 113, 79, 799, 235),
            box("新疆美味烧烤", 1211, 79, 1895, 235),
            box("羊肉串", 132, 248, 374, 350),
        ]
        items, sections, _ = parse_menu(boxes, settings)
        self.assertEqual([i.cn for i in items], ["羊肉串"])
        self.assertIn("新疆美味烧烤", sections)

    def test_section_banner_found_by_wording_not_height(self):
        """Wall boards print "烧味套餐" at dish-name size, in a coloured pill."""
        settings = Settings()
        boxes = [
            box("烧味套餐", 0, 290, 143, 329),
            box("卤味套餐", 586, 308, 727, 346),
            box("红油牛腱子饭", 412, 348, 595, 391),
            box("28元/份", 780, 360, 882, 392),
            box("金牌脆皮烧鹅", 987, 361, 1163, 397),
            box("88例", 1386, 366, 1456, 396),
        ]
        items, sections, _ = parse_menu(boxes, settings)
        self.assertEqual([i.cn for i in items], ["红油牛腱子饭", "金牌脆皮烧鹅"])
        self.assertIn("烧味套餐", sections)
        self.assertIn("卤味套餐", sections)

    def test_set_meal_with_a_price_is_still_a_dish(self):
        """"咖喱鸡套餐" is a dish; text alone cannot separate it from a banner."""
        settings = Settings()
        boxes = [
            box("卤味套餐", 586, 308, 727, 346),
            box("咖喱鸡套餐", 412, 348, 595, 391),
            box("28元", 780, 360, 882, 392),
        ]
        items, sections, _ = parse_menu(boxes, settings)
        self.assertEqual([(i.cn, i.price_value) for i in items], [("咖喱鸡套餐", 28.0)])
        self.assertEqual(sections, ["卤味套餐"])

    def test_skips_printed_footnotes(self):
        """A note fragment once inherited a price and became a fake "mixed plate"."""
        settings = Settings()
        boxes = [
            box("卤水猪脚饭", 408, 384, 561, 423),
            box("20元/份", 778, 396, 880, 428),
            box("(以上双拼按价高加5元,三拼按价高加10元)", 368, 646, 863, 689),
            box("姘按价高加10元)", 0, 801, 197, 837),
        ]
        items, _, warnings = parse_menu(boxes, settings)
        self.assertEqual([i.cn for i in items], ["卤水猪脚饭"])
        self.assertTrue(any("note line" in w for w in warnings), warnings)

    def test_drops_single_character_fragment_at_the_frame_edge(self):
        settings = Settings()
        boxes = [box("饭", 0, 342, 32, 374), box("28元/份", 190, 346, 296, 378)]
        items, _, warnings = parse_menu(boxes, settings)
        self.assertEqual(items, [])
        self.assertTrue(any("fragment" in w for w in warnings), warnings)

    def test_clipped_price_column_does_not_hijack_the_next_column(self):
        """The bug this photo exposed.

        The leftmost column of the board is cropped off-frame, leaving prices
        with no names. Each of those prices used to bind to the first dish of the
        *next* column, so 卤水猪脚饭 came out at 25 (its neighbour's price)
        instead of 20.
        """
        settings = Settings()
        boxes = [
            # The cropped column: prices only, names are outside the photo.
            box("28元/份", 190, 346, 296, 378),
            box("25元/份", 181, 381, 287, 411),
            # Column 2: names and their own prices.
            box("红油牛腱子饭", 412, 348, 595, 391),
            box("28元/份", 780, 360, 882, 392),
            box("卤水猪脚饭", 408, 384, 561, 423),
            box("20元/份", 778, 396, 880, 428),
        ]
        items, _, warnings = parse_menu(boxes, settings)
        got = {i.cn: i.price_value for i in items}
        self.assertEqual(got, {"红油牛腱子饭": 28.0, "卤水猪脚饭": 20.0})
        self.assertTrue(
            any("no dish name beside them" in w for w in warnings),
            f"stranded prices should be reported: {warnings}",
        )

    def test_four_column_board_keeps_every_column_separate(self):
        """Full replica of the roast-meat board, including the price-less column."""
        settings = Settings()
        boxes = [
            box("烧味套餐", 0, 290, 143, 329),
            box("卤味套餐", 586, 308, 727, 346),
            box("小吃", 1771, 315, 1841, 351),
            # Column 1 (cropped names)
            box("28元/份", 190, 346, 296, 378),
            # Column 2 (卤味套餐)
            box("红油牛腱子饭", 412, 348, 595, 391),
            box("28元/份", 780, 360, 882, 392),
            box("卤水猪脚饭", 408, 384, 561, 423),
            box("20元/份", 778, 396, 880, 428),
            # Column 3 (烧味例排, its banner was not read by OCR)
            box("金牌脆皮烧鹅", 987, 361, 1163, 397),
            box("88例", 1386, 366, 1456, 396),
            box("密汁排骨", 987, 409, 1105, 445),
            box("68例", 1392, 414, 1460, 446),
            # Column 4 (小吃 - no prices printed at all)
            box("烧鹅头带脖子", 1577, 362, 1759, 398),
            box("密汁鸡翅", 1583, 399, 1707, 435),
        ]
        items, _, _ = parse_menu(boxes, settings)
        got = [(i.cn, i.price_value, i.section) for i in items]
        # Order is by vertical position, so the columns interleave - that is the
        # reading order of a four-column board read across.
        self.assertEqual(
            got,
            [
                ("红油牛腱子饭", 28.0, "卤味套餐"),
                ("金牌脆皮烧鹅", 88.0, None),
                ("烧鹅头带脖子", None, "小吃"),
                ("卤水猪脚饭", 20.0, "卤味套餐"),
                ("密汁排骨", 68.0, None),
                ("密汁鸡翅", None, "小吃"),
            ],
        )


class TestBilingualBoards(unittest.TestCase):
    """Boards that print a Chinese name and an English line for every dish.

    Seen on a real "乡谷村" board: 13 dishes produced 31 "items", one per
    language, with prices landing on English fragments.
    """

    settings = Settings()

    def test_english_on_the_same_row_is_not_a_second_dish(self):
        boxes = [
            box("特色酱汁拌面", 100, 300, 340, 367),
            box("Noodles with Special Sauce", 400, 320, 800, 382),
            box("8元", 1050, 348, 1120, 392),
            box("椒麻酱汁拌面", 100, 420, 340, 487),
            box("Noodles with Pepper Mala Sauce", 400, 440, 800, 502),
            box("16元/15个", 1000, 468, 1120, 512),
            box("黑胡椒酱汁拌面", 100, 540, 360, 607),
            box("Noodles with Black Pepper Sauce", 400, 560, 820, 622),
            box("8元", 1050, 588, 1120, 632),
        ]
        items, _, warnings = parse_menu(boxes, self.settings)
        self.assertEqual(
            [item.cn for item in items],
            ["特色酱汁拌面", "椒麻酱汁拌面", "黑胡椒酱汁拌面"],
        )
        self.assertEqual([item.price_value for item in items], [8.0, 16.0, 8.0])
        self.assertEqual(items[0].en_hint, "Noodles with Special Sauce")
        self.assertTrue(any("Folded" in warning for warning in warnings))

    def test_english_under_the_name_is_folded_too(self):
        boxes = [
            box("生烫云南米线(骨汤)", 100, 300, 400, 360),
            box("Blanched Yunnan Rice Noodles (Bone Broth)", 100, 365, 700, 410),
            box("12元", 1050, 300, 1120, 344),
            box("生烫云南米线(番茄)", 100, 430, 400, 490),
            box("Blanched Yunnan Rice Noodles (Tomato)", 100, 495, 700, 540),
            box("12元", 1050, 430, 1120, 474),
            box("生烫云南米线(酸汤)", 100, 560, 400, 620),
            box("Blanched Yunnan Rice Noodles (Sour Soup)", 100, 625, 700, 670),
            box("12元", 1050, 560, 1120, 604),
        ]
        items, _, _ = parse_menu(boxes, self.settings)
        self.assertEqual(len(items), 3)
        self.assertTrue(all(item.price_value == 12.0 for item in items))
        self.assertEqual(items[0].en_hint, "Blanched Yunnan Rice Noodles (Bone Broth)")

    def test_english_only_menu_keeps_its_lines(self):
        """No Chinese on the board means nothing to fold - do not delete it."""
        boxes = [
            box("Chaozhou-Style Lean Meat Rice Noodles", 100, 300, 700, 350),
            box("15元", 1000, 300, 1070, 344),
            box("Original Hot and Sour Sweet Potato Vermicelli", 100, 400, 760, 450),
            box("15元", 1000, 400, 1070, 444),
            box("Beef Ball Bone Soup (No Noodles)", 100, 500, 700, 550),
            box("12元", 1000, 500, 1070, 544),
        ]
        items, _, _ = parse_menu(boxes, self.settings)
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0].cn, "Chaozhou-Style Lean Meat Rice Noodles")
        self.assertEqual(items[0].price_value, 15.0)

    def test_bottom_aligned_price_still_binds_to_its_name(self):
        """The price sits on the English baseline, 36px below the name centre."""
        boxes = [
            box("红烧排骨面拼抄手", 100, 300, 400, 394),
            box("Braised Beef Brisket Noodle Soup", 420, 340, 900, 420),
            box("18.5元", 1000, 375, 1120, 419),
            box("番茄鸡蛋面拼抄手", 100, 460, 400, 554),
            box("Tomato & Egg Noodle Soup with Wontons", 420, 500, 900, 580),
            box("15元", 1030, 535, 1120, 579),
            box("香菇炖鸡面拼抄手", 100, 620, 400, 714),
            box("Mushroom & Stewed Chicken Noodle Soup", 420, 660, 900, 740),
            box("16.5元", 1010, 695, 1120, 739),
        ]
        items, _, _ = parse_menu(boxes, self.settings)
        self.assertEqual(
            [(item.cn, item.price_value) for item in items],
            [
                ("红烧排骨面拼抄手", 18.5),
                ("番茄鸡蛋面拼抄手", 15.0),
                ("香菇炖鸡面拼抄手", 16.5),
            ],
        )


class TestEndToEnd(unittest.TestCase):
    def test_printed_menu_matches_expected_price_list(self):
        image = ROOT / "test.jpg"
        if not image.exists():
            self.skipTest("test.jpg not present")
        # Pinned to the OCR extractor: this test is the ground truth for the
        # geometric parser, so it must not depend on a vision model or a network.
        result = scan_menu(image, extractor="ocr", enrich=False)
        got = [(item.cn, item.price_value) for item in result.items]
        expected = [
            ("酸辣土豆丝", 11.0), ("香干回锅肉饭", 15.0), ("凉瓜鸡蛋饭", 12.0),
            ("腐竹回锅肉饭", 15.0), ("西红柿蛋饭", 12.0), ("青椒回锅肉饭", 15.0),
            ("木耳肉片饭", 14.0), ("蒜苗回锅肉饭", 15.0), ("青椒肥肠饭", 15.0),
            ("青椒肉片饭", 14.0), ("酸菜肥肠饭", 15.0), ("香菇肉片饭", 14.0),
            ("青椒猪肝饭", 15.0), ("土豆肉丝饭", 14.0), ("红烧鱼片饭", 15.0),
            ("青椒腊肠饭", 14.0), ("腐竹排骨饭", 15.0), ("花菜肉片饭", 14.0),
            ("凉瓜牛肉饭", 18.0), ("芹菜香干肉丝饭", 14.0), ("青椒牛肉饭", 18.0),
        ]
        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
