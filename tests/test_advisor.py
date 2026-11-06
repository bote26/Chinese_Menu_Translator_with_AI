"""Tests for the grounded Q&A.

Two properties are pinned here: the common questions never wait for a model, and
what the model is told never exceeds what the scan actually found.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from menu_vision.advisor import (  # noqa: E402
    Answer,
    ask_about_dish,
    ask_about_menu,
    dish_context,
    menu_context,
)
from menu_vision.llm import LLMError  # noqa: E402
from menu_vision.models import DishInfo, EnrichedItem, MenuResult  # noqa: E402
from menu_vision.preferences import TasteProfile  # noqa: E402
from menu_vision.recommend import build_plan  # noqa: E402


def dish(
    name: str,
    price: float | None = 16.0,
    category: str = "noodle",
    *,
    spice: int = 1,
    ingredients: tuple[str, ...] = ("wheat noodles", "sesame oil"),
    allergens: tuple[str, ...] = ("wheat", "sesame"),
    vegetarian: bool | None = None,
    description: str = "Noodles in a savoury sauce.",
    item_id: int = 1,
    described: bool = True,
) -> EnrichedItem:
    if not described:
        return EnrichedItem(id=item_id, cn=name, price_value=price)
    return EnrichedItem(
        id=item_id,
        cn=name,
        price_value=price,
        price_raw=f"{price:g}元" if price is not None else None,
        price_display=f"¥{price:g}" if price is not None else "",
        dish=DishInfo(
            en="Noodles",
            category=category,
            spice_level=spice,
            vegetarian=vegetarian,
            ingredients=list(ingredients),
            allergens=list(allergens),
            description=description,
            confidence=0.9,
            price_comment="8 CNY - tied for cheapest of the 4 noodles here (median 8 CNY).",
        ),
    )


class _FakeClient:
    """Records what it was asked and returns a scripted reply."""

    def __init__(self, reply: str = "It is mild.", fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls: list[list[dict]] = []

    def chat_text(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        if self.fail:
            raise LLMError("HTTP 503: overloaded")
        return self.reply


class OfflineAnswerTests(unittest.TestCase):
    def setUp(self):
        self.profile = TasteProfile(party_size=4, budget_total=80.0, spice_tolerance=1)
        self.item = dish("特色酱汁拌面", 8.0)

    def test_price_question_answers_from_the_facts(self):
        answer = ask_about_dish(self.item, "How much is it?", self.profile, client=None)
        self.assertEqual(answer.source, "facts")
        self.assertIn("¥8", answer.text)

    def test_spice_question_reports_the_level(self):
        answer = ask_about_dish(self.item, "Is it spicy?", self.profile, client=None)
        self.assertIn("1/3", answer.text)
        self.assertEqual(answer.source, "facts")

    def test_allergen_question_on_a_bare_dish_says_it_does_not_know(self):
        bare = dish("水煮鱼", 48.0, described=False)
        answer = ask_about_dish(bare, "Does it have peanuts?", self.profile, client=None)
        self.assertEqual(answer.source, "facts")
        self.assertIn("does not list ingredients", answer.text)
        self.assertIn("过敏", answer.text)  # the phrase to show the staff
        self.assertNotIn("no allergens", answer.text.lower())

    def test_allergen_question_names_the_listed_allergens(self):
        answer = ask_about_dish(self.item, "Any allergens?", self.profile, client=None)
        self.assertIn("wheat", answer.text)
        self.assertIn("may not be complete", answer.text)

    def test_portion_question_uses_the_party_size(self):
        answer = ask_about_dish(self.item, "How many should we order?", self.profile,
                                client=None)
        self.assertIn("4", answer.text)

    def test_vegetarian_unknown_is_stated_as_unknown(self):
        answer = ask_about_dish(self.item, "Is this vegetarian?", self.profile, client=None)
        self.assertIn("does not say", answer.text)

    def test_vegetarian_explicit_marker_is_quoted(self):
        veg = dish("清炒时蔬", 18.0, "vegetable", vegetarian=True)
        answer = ask_about_dish(veg, "Is this vegetarian?", self.profile, client=None)
        self.assertIn("marks this dish as vegetarian", answer.text)


class GroundingTests(unittest.TestCase):
    def test_dish_context_contains_nothing_outside_the_item(self):
        item = dish("麻婆豆腐", 16.0, "tofu", ingredients=("tofu", "chilli"))
        context = dish_context(item, TasteProfile())
        self.assertIn("tofu", context)
        self.assertIn("chilli", context)
        # A dish the model never described must not gain an ingredient here.
        bare = dish("水煮鱼", 48.0, described=False)
        bare_context = dish_context(bare, TasteProfile())
        self.assertIn("not stated", bare_context)
        for token in ("tofu", "chilli", "pork", "fish"):
            self.assertNotIn(token, bare_context)

    def test_menu_context_stays_small_for_a_thirty_dish_menu(self):
        items = [dish(f"菜{i}", 10.0 + i, "noodle", item_id=i) for i in range(1, 31)]
        result = MenuResult(items=items, source="menu.jpg")
        profile = TasteProfile(party_size=4, budget_total=200.0)
        recs = build_plan(items, profile)
        context = menu_context(result, profile, recs)
        self.assertLess(len(context), 6000)
        # Every dish is still represented, at least as a compact row.
        for item in items[-5:]:
            self.assertIn(item.cn, context)

    def test_context_carries_the_table(self):
        item = dish("麻婆豆腐", 16.0)
        context = dish_context(
            item,
            TasteProfile(party_size=6, budget_total=300.0, avoid_allergens=("peanut",)),
        )
        self.assertIn("6 people", context)
        self.assertIn("peanut", context)


class ModelPathTests(unittest.TestCase):
    def setUp(self):
        self.profile = TasteProfile(party_size=2, budget_total=100.0)
        self.item = dish("特色酱汁拌面", 8.0)

    def test_common_questions_never_call_the_model(self):
        client = _FakeClient()
        for question in ("How much is it?", "Is it spicy?", "Is this vegetarian?",
                         "What is in it?", "How many should we order?"):
            ask_about_dish(self.item, question, self.profile, client=client)
        self.assertEqual(client.calls, [], "facts already answer these")

    def test_the_model_gets_the_facts_and_the_question(self):
        client = _FakeClient("The menu does not list nuts, but ask the staff.")
        answer = ask_about_dish(self.item, "Would a child like it?", self.profile,
                                client=client)
        self.assertEqual(answer.source, "model")
        self.assertEqual(answer.text, "The menu does not list nuts, but ask the staff.")
        sent = client.calls[0]
        self.assertIn("DISH FACTS", sent[1]["content"])
        self.assertIn("特色酱汁拌面", sent[1]["content"])
        self.assertIn("Would a child like it?", sent[1]["content"])
        # The prompt must forbid filling gaps from general knowledge.
        self.assertIn("ONLY from the FACTS", sent[0]["content"])
        self.assertIn("Never do arithmetic", sent[0]["content"])

    def test_a_failed_call_falls_back_to_the_facts(self):
        client = _FakeClient(fail=True)
        answer = ask_about_dish(self.item, "Would a child like it?", self.profile,
                                client=client)
        self.assertEqual(answer.source, "facts")
        self.assertTrue(answer.text)
        self.assertTrue(any("unavailable" in note for note in answer.notes))

    def test_an_empty_reply_falls_back_to_the_facts(self):
        client = _FakeClient("   ")
        answer = ask_about_dish(self.item, "Would a child like it?", self.profile,
                                client=client)
        self.assertEqual(answer.source, "facts")

    def test_no_client_is_a_clean_answer_not_an_error(self):
        answer = ask_about_dish(self.item, "Would a child like it?", self.profile,
                                client=None)
        self.assertIsInstance(answer, Answer)
        self.assertEqual(answer.source, "facts")
        self.assertTrue(answer.notes)


class MenuAnswerTests(unittest.TestCase):
    def setUp(self):
        self.items = [
            dish("米饭", 3.0, "rice", item_id=1, vegetarian=True, allergens=()),
            dish("番茄蛋汤", 12.0, "soup", item_id=2, vegetarian=True,
                 allergens=("egg",)),
            dish("清炒时蔬", 18.0, "vegetable", item_id=3, vegetarian=True,
                 allergens=()),
            dish("宫保鸡丁", 32.0, "meat", item_id=4, vegetarian=False,
                 ingredients=("chicken", "peanut"), allergens=("peanut",)),
        ]
        self.result = MenuResult(items=self.items, source="menu.jpg")
        self.profile = TasteProfile(party_size=4, budget_total=100.0)
        self.recs = build_plan(self.items, self.profile)

    def test_what_should_we_order_lists_the_plan(self):
        answer = ask_about_menu("What should we order?", self.result, self.profile,
                                self.recs, client=None)
        self.assertIn("For 4:", answer.text)
        self.assertEqual(answer.source, "facts")

    def test_vegetarian_question_lists_only_marked_dishes(self):
        answer = ask_about_menu("Which dishes are vegetarian?", self.result,
                                self.profile, self.recs, client=None)
        self.assertIn("清炒时蔬", answer.text)
        self.assertNotIn("宫保鸡丁", answer.text)

    def test_total_cost_question_quotes_the_computed_total(self):
        answer = ask_about_menu("How much is this in total?", self.result,
                                self.profile, self.recs, client=None)
        self.assertIn("¥100", answer.text)
        self.assertIn("cheapest dish", answer.text)


if __name__ == "__main__":
    unittest.main()
