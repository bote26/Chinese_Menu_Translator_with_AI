"""Tests for the deterministic price commentary.

The model used to write this text; it produced three "cheapest rice dish"
claims on one board. These tests pin the arithmetic.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from menu_vision.models import DishInfo, EnrichedItem  # noqa: E402
from menu_vision.pricing import NO_PRICE, annotate_prices  # noqa: E402


def item(name: str, price: float | None, category: str = "rice bowl",
         raw: str | None = None, id: int = 1) -> EnrichedItem:
    return EnrichedItem(
        id=id,
        cn=name,
        price_value=price,
        price_raw=raw or (f"{price:g}元" if price is not None else None),
        dish=DishInfo(cn=name, category=category),
    )


class TestPriceComments(unittest.TestCase):
    def test_item_without_a_price_says_so(self):
        items = [item("羊肉串", None)]
        annotate_prices(items)
        self.assertEqual(items[0].dish.price_comment, NO_PRICE)

    def test_ties_are_reported_as_ties(self):
        # The real failure: 16, 16, 16 were each called "the cheapest dish".
        items = [
            item("卤水鸡腿饭", 16), item("卤水鸭腿饭", 16), item("潮汕肉卷饭", 16),
            item("卤水鸭饭", 19), item("卤水猪脚饭", 20), item("卤水猪舌饭", 22),
            item("红油牛腱子饭", 28), item("卤水粉肠饭", 22),
        ]
        annotate_prices(items)
        for entry in items[:3]:
            self.assertIn("tied for cheapest (3 at this price)", entry.dish.price_comment)
        self.assertNotIn("the cheapest of", items[0].dish.price_comment)

    def test_compares_against_the_median_of_its_own_category(self):
        # median of (10, 26, 30, 40) is 28, so 26 sits 2 below it.
        items = [item("a", 10), item("b", 30), item("c", 40), item("d", 26)]
        annotate_prices(items)
        self.assertIn("2 CNY below the median of the 4 rice bowls here",
                      items[3].dish.price_comment)
        self.assertIn("the most expensive", items[2].dish.price_comment)

    def test_within_one_yuan_is_described_as_typical(self):
        items = [item("a", 15), item("b", 19), item("c", 20.5), item("d", 25)]
        annotate_prices(items)
        self.assertIn("about the typical price", items[2].dish.price_comment)

    def test_falls_back_to_all_prices_when_a_category_is_too_small(self):
        items = [item("a", 10, "rice bowl"), item("b", 20, "rice bowl"),
                 item("c", 30, "rice bowl"), item("d", 90, "roast meat")]
        annotate_prices(items)
        self.assertIn("the most expensive of the 4 priced dishes here",
                      items[3].dish.price_comment)

    def test_per_half_bird_prices_are_not_compared_with_portions(self):
        items = [item("a", 10), item("b", 20), item("c", 30),
                 item("白切鸡", 50, raw="50元半只")]
        annotate_prices(items)
        self.assertEqual(items[3].dish.price_comment,
                         "50 CNY - the only item priced per half bird here.")

    def test_single_priced_item_does_not_invent_a_comparison(self):
        items = [item("超浓水牛奶双皮奶", 7.65), item("销量96", None, category="")]
        annotate_prices(items)
        self.assertEqual(items[0].dish.price_comment,
                         "7.65 CNY - the only priced item on this menu.")

    def test_the_price_in_the_comment_matches_the_price_field(self):
        import re

        items = [item("a", 16), item("b", 22), item("c", 28), item("d", 16)]
        annotate_prices(items)
        for entry in items:
            lead = re.match(r"^(\d+(?:\.\d+)?) CNY", entry.dish.price_comment)
            self.assertIsNotNone(lead, entry.dish.price_comment)
            self.assertEqual(float(lead.group(1)), entry.price_value)


if __name__ == "__main__":
    unittest.main()
