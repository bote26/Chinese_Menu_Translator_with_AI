"""Tests for the deterministic recommender.

The tests that matter most are the ones that stop the app from being confidently
wrong: an empty allergen list must not read as "safe", a menu with no prices must
not read as "within budget", and no sentence may quote a number the data does not
contain.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from menu_vision.models import DishInfo, EnrichedItem  # noqa: E402
from menu_vision.preferences import (  # noqa: E402
    TasteProfile,
    event_from_item,
)
from menu_vision.recommend import (  # noqa: E402
    build_plan,
    budget_line,
    slot_of,
    slot_template,
)


def dish(
    name: str,
    price: float | None = 16.0,
    category: str = "meat",
    *,
    spice: int = 1,
    ingredients: tuple[str, ...] = ("pork", "soy sauce"),
    allergens: tuple[str, ...] = (),
    vegetarian: bool | None = False,
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
            en=name,
            category=category,
            spice_level=spice,
            vegetarian=vegetarian,
            ingredients=list(ingredients),
            allergens=list(allergens),
            confidence=0.9,
        ),
    )


def mixed_menu() -> list[EnrichedItem]:
    """One dish per course, all priced, one category so peer medians are simple."""
    return [
        dish("米饭", 3.0, "rice", item_id=1, vegetarian=True,
             ingredients=("rice",), allergens=()),
        dish("番茄蛋汤", 12.0, "soup", item_id=2, vegetarian=True,
             ingredients=("tomato", "egg"), allergens=("egg",)),
        dish("清炒时蔬", 18.0, "vegetable", item_id=3, vegetarian=True,
             ingredients=("greens",), allergens=()),
        dish("宫保鸡丁", 32.0, "meat", item_id=4, spice=2,
             ingredients=("chicken", "peanut"), allergens=("peanut",)),
        dish("水煮鱼", 48.0, "meat", item_id=5, spice=3,
             ingredients=("fish", "chilli"), allergens=("fish",)),
    ]


class FilterTests(unittest.TestCase):
    def test_stated_allergen_excludes_the_dish(self):
        rec = build_plan(mixed_menu(), TasteProfile(avoid_allergens=("peanut",)))
        excluded = {d.cn: d.excluded_reason for d in rec.excluded}
        self.assertIn("宫保鸡丁", excluded)
        self.assertIn("peanut", excluded["宫保鸡丁"])
        self.assertNotIn("宫保鸡丁", {s.cn for s in rec.plan.slots})

    def test_empty_allergen_list_is_not_safe_but_is_not_excluded(self):
        # Absence of evidence is not evidence of absence: the dish stays, at a
        # capped score, and the plan says the status is unknown.
        rec = build_plan(mixed_menu(), TasteProfile(avoid_allergens=("peanut",)))
        self.assertNotIn("米饭", {d.cn for d in rec.excluded})
        self.assertTrue(rec.unknown_allergen_ids)
        self.assertTrue(any("unknown allergen" in note for note in rec.plan.notes))

    def test_vegetarian_only_excludes_only_known_meat(self):
        menu = [
            dish("清炒时蔬", 18.0, "vegetable", item_id=1, vegetarian=True),
            dish("红烧肉", 38.0, "meat", item_id=2, vegetarian=False),
            dish("豆腐煲", 22.0, "vegetable", item_id=3, vegetarian=None),
        ]
        rec = build_plan(menu, TasteProfile(vegetarian_only=True))
        self.assertIn("红烧肉", {d.cn for d in rec.excluded})
        self.assertNotIn("豆腐煲", {d.cn for d in rec.excluded})
        # Unknown vegetarian status is labelled, not silently accepted as fine.
        tofu = [d for d in rec.shortlist if d.cn == "豆腐煲"][0]
        self.assertTrue(any("Vegetarian status is not stated" in r.text for r in tofu.reasons))

    def test_spice_over_tolerance_is_penalised_not_excluded(self):
        rec = build_plan(mixed_menu(), TasteProfile(spice_tolerance=0))
        self.assertNotIn("水煮鱼", {d.cn for d in rec.excluded})
        hot = [d for d in rec.shortlist if d.cn == "水煮鱼"][0]
        mild = [d for d in rec.shortlist if d.cn == "米饭"][0]
        self.assertLess(hot.score, mild.score)
        self.assertTrue(any("above your 0/3" in r.text or "against your 0/3" in r.text
                            for r in hot.reasons))

    def test_every_dish_excluded_says_so(self):
        menu = [dish("红烧肉", 38.0, "meat", item_id=1, vegetarian=False)]
        rec = build_plan(menu, TasteProfile(vegetarian_only=True))
        self.assertEqual(rec.plan.slots, ())
        self.assertTrue(any("Every dish was excluded" in note for note in rec.notes))


class UnknownFactsTests(unittest.TestCase):
    def setUp(self):
        self.menu = [
            dish("酸辣土豆丝", 16.0, item_id=1, described=False),
            dish("水煮鱼", 48.0, item_id=2, described=False),
            dish("米饭", 3.0, item_id=3, described=False),
        ]
        self.rec = build_plan(self.menu, TasteProfile(spice_tolerance=1))

    def test_no_forbidden_claims(self):
        text = " ".join(
            reason.text
            for dish_item in list(self.rec.shortlist)
            for reason in dish_item.reasons
        )
        for word in ("vegetarian", "no allergens", "not spicy", "mild"):
            self.assertNotIn(word, text.lower())

    def test_evidence_stays_below_the_threshold_and_is_reported(self):
        # Price data is real data, so evidence is not zero even here - but the
        # dishes were never described, and that is what the note has to say.
        self.assertLess(self.rec.evidence, 0.5)
        self.assertTrue(any("not analysed" in note for note in self.rec.notes))

    def test_a_plan_is_still_built(self):
        # A tourist with the API key down still needs an answer.
        self.assertTrue(self.rec.plan.slots)
        self.assertEqual(self.rec.unknown_allergen_ids, ())

    def test_every_dish_says_the_menu_does_not_say(self):
        for dish_item in self.rec.shortlist:
            self.assertTrue(
                any("does not say what is in this dish" in r.text
                    for r in dish_item.reasons),
                dish_item.cn,
            )


class BudgetTests(unittest.TestCase):
    def test_no_price_menu_never_invents_a_total(self):
        menu = [dish(name, None, item_id=index) for index, name in
                enumerate(["米饭", "炒青菜", "红烧肉"], start=1)]
        rec = build_plan(menu, TasteProfile(party_size=4, budget_total=100.0))
        self.assertIsNone(rec.plan.total_known)
        self.assertIsNone(rec.plan.per_head)
        self.assertIsNone(rec.plan.shortfall)
        self.assertTrue(any("no prices" in note for note in rec.plan.notes))
        self.assertNotIn("¥0", budget_line(rec.plan))
        for note in rec.plan.notes:
            self.assertNotIn("¥0 ", note)

    def test_over_budget_returns_best_effort_with_matching_numbers(self):
        rec = build_plan(mixed_menu(), TasteProfile(party_size=4, budget_total=5.0))
        total = rec.plan.total_known
        self.assertIsNotNone(total)
        self.assertGreater(total, 5.0)
        note = [n for n in rec.plan.notes if "over your" in n][0]
        self.assertIn(f"¥{total:g}", note)
        self.assertIn(f"¥{total - 5.0:g}", note)
        self.assertIn("¥5", note)

    def test_under_budget_has_no_shortfall(self):
        rec = build_plan(mixed_menu(), TasteProfile(party_size=2, budget_total=500.0))
        self.assertIsNone(rec.plan.shortfall)
        self.assertLessEqual(rec.plan.total_known, 500.0)

    def test_budget_trims_the_plan(self):
        rec = build_plan(mixed_menu(), TasteProfile(party_size=4, budget_total=30.0))
        self.assertLessEqual(rec.plan.total_known, 30.0)
        self.assertTrue(any("Dropped" in note for note in rec.plan.notes))

    def test_unpriced_items_are_not_counted_as_free(self):
        menu = mixed_menu() + [dish("时价海鲜", None, "meat", item_id=9)]
        rec = build_plan(menu, TasteProfile(party_size=2, budget_total=400.0))
        priced = sum(
            1 for slot in rec.plan.slots if slot.price_value is not None
        )
        if rec.plan.unpriced_count:
            self.assertLess(priced, len(rec.plan.slots))
            for note in rec.plan.notes:
                self.assertNotIn("free", note.lower())


class PlanShapeTests(unittest.TestCase):
    def test_empty_menu_is_handled(self):
        rec = build_plan([], TasteProfile())
        self.assertEqual(rec.plan.slots, ())
        self.assertEqual(budget_line(rec.plan), "No plan yet.")

    def test_party_size_scales_the_plan(self):
        menu = [
            dish("米饭", 3.0, "rice", item_id=1, vegetarian=True),
            dish("番茄蛋汤", 12.0, "soup", item_id=2, vegetarian=True),
            dish("清炒时蔬", 18.0, "vegetable", item_id=3, vegetarian=True),
            dish("豆腐煲", 22.0, "vegetable", item_id=4, vegetarian=True),
            dish("红烧肉", 38.0, "meat", item_id=5),
            dish("白切鸡", 42.0, "meat", item_id=6, ingredients=("chicken",)),
            dish("清蒸鱼", 58.0, "meat", item_id=7, ingredients=("fish",)),
            dish("回锅肉", 36.0, "meat", item_id=8, ingredients=("pork", "leek")),
        ]
        counts = [len(build_plan(menu, TasteProfile(party_size=p)).plan.slots)
                  for p in (1, 2, 4, 6, 12)]
        self.assertEqual(counts, sorted(counts), counts)
        self.assertLessEqual(counts[-1], 10)

    def test_a_four_top_never_gets_three_vegetables(self):
        # When the menu has every course, the template decides the mix: two
        # vegetables, however many are on offer.
        menu = [
            dish("米饭", 3.0, "rice", item_id=1, vegetarian=True),
            dish("番茄蛋汤", 12.0, "soup", item_id=2, vegetarian=True),
            dish("青菜1", 18.0, "vegetable", item_id=3, vegetarian=True,
                 ingredients=("greens",)),
            dish("青菜2", 19.0, "vegetable", item_id=4, vegetarian=True,
                 ingredients=("cabbage",)),
            dish("青菜3", 20.0, "vegetable", item_id=5, vegetarian=True,
                 ingredients=("spinach",)),
            dish("青菜4", 21.0, "vegetable", item_id=6, vegetarian=True,
                 ingredients=("bok choy",)),
            dish("红烧肉", 38.0, "meat", item_id=7, ingredients=("pork",)),
            dish("白切鸡", 42.0, "meat", item_id=8, ingredients=("chicken",)),
        ]
        rec = build_plan(menu, TasteProfile(party_size=4))
        vegetables = sum(1 for slot in rec.plan.slots if slot.slot == "vegetable")
        self.assertLessEqual(vegetables, 2)

    def test_one_course_menu_serves_everyone(self):
        # A noodle shop has no protein or vegetable course. Two dishes for four
        # people would be a worse answer than four bowls.
        menu = [dish(f"面{i}", 12.0, "noodle", item_id=i, vegetarian=True)
                for i in range(1, 6)]
        rec = build_plan(menu, TasteProfile(party_size=4))
        # At least a dish each, even though the kitchen only makes noodles.
        self.assertGreaterEqual(len(rec.plan.slots), 4)
        self.assertTrue(all(slot.slot == "staple" for slot in rec.plan.slots))
        self.assertFalse(any("Every dish was excluded" in n for n in rec.notes))

    def test_missing_course_is_reported_not_faked(self):
        menu = [
            dish("米饭", 3.0, "rice", item_id=1, vegetarian=True),
            dish("红烧肉", 38.0, "meat", item_id=2),
        ]
        rec = build_plan(menu, TasteProfile(party_size=2))
        self.assertEqual({s.cn for s in rec.plan.slots[:2]}, {"米饭", "红烧肉"})
        self.assertTrue(rec.plan.slots)

    def test_drinks_are_kept_out_of_the_meal(self):
        menu = mixed_menu() + [
            dish("酸梅汁", 8.0, "drink", item_id=7, vegetarian=True,
                 ingredients=("plum",))
        ]
        rec = build_plan(menu, TasteProfile(party_size=4))
        self.assertNotIn("drink", {slot.slot for slot in rec.plan.slots})
        self.assertIn("酸梅汁", {d.cn for d in rec.shortlist})

    def test_slot_classification(self):
        self.assertEqual(slot_of(dish("番茄蛋汤", 12.0, "")), "soup")
        self.assertEqual(slot_of(dish("米饭", 3.0, "")), "staple")
        self.assertEqual(slot_of(dish("水煮鱼", 48.0, "")), "protein")
        self.assertEqual(slot_of(dish("清炒时蔬", 18.0, "")), "vegetable")
        self.assertEqual(slot_of(dish("酸梅汁", 8.0, "")), "drink")

    def test_sweet_sauce_dish_is_not_a_drink(self):
        # 汁 is a drink keyword, and 蜜汁叉烧 contains it. Protein wins.
        self.assertEqual(slot_of(dish("蜜汁叉烧", 32.0, "", ingredients=("pork",))),
                         "protein")

    def test_slot_template_sizes(self):
        self.assertEqual(len(slot_template(1)), 3)
        self.assertEqual(len(slot_template(2)), 4)
        self.assertEqual(len(slot_template(4)), 6)
        self.assertEqual(len(slot_template(6)), 8)
        self.assertLessEqual(len(slot_template(12)), 10)


class DeterminismTests(unittest.TestCase):
    def test_same_input_twice_is_byte_identical(self):
        first = build_plan(mixed_menu(), TasteProfile(party_size=4, budget_total=90.0))
        second = build_plan(mixed_menu(), TasteProfile(party_size=4, budget_total=90.0))
        self.assertEqual(first.model_dump_json(), second.model_dump_json())

    def test_input_order_does_not_change_the_plan(self):
        ordered = mixed_menu()
        shuffled = list(reversed(ordered))
        for index, item in enumerate(shuffled, start=1):
            item.id = index
        first = build_plan(ordered, TasteProfile(party_size=4))
        second = build_plan(shuffled, TasteProfile(party_size=4))
        self.assertEqual(
            [slot.cn for slot in first.plan.slots],
            [slot.cn for slot in second.plan.slots],
        )


class ReasonArithmeticTests(unittest.TestCase):
    """No sentence may quote a number the data does not contain."""

    def test_every_price_in_every_reason_is_a_real_number(self):
        menu = mixed_menu()
        profile = TasteProfile(party_size=4, budget_total=100.0)
        rec = build_plan(menu, profile)
        prices = {item.price_value for item in menu if item.price_value is not None}
        permitted = prices | {100.0, 25.0} | {round(100.0 / 4, 2)}
        import statistics

        permitted.add(statistics.median(sorted(prices)))
        for dish_item in list(rec.shortlist) + [
            slot_dish
            for slot_dish in rec.excluded
        ]:
            for reason in dish_item.reasons:
                for number in re.findall(r"¥(\d+(?:\.\d+)?)", reason.text):
                    self.assertIn(
                        float(number),
                        permitted,
                        f"{dish_item.cn}: {reason.text!r} quotes ¥{number}",
                    )

    def test_the_score_never_leaves_the_unit_interval(self):
        rec = build_plan(mixed_menu(), TasteProfile(party_size=6, budget_total=60.0))
        for dish_item in list(rec.shortlist) + list(rec.excluded):
            self.assertGreaterEqual(dish_item.score, 0.0, dish_item.cn)
            self.assertLessEqual(dish_item.score, 1.0, dish_item.cn)


class MemoryTests(unittest.TestCase):
    def test_a_liked_dish_rises(self):
        menu = mixed_menu()
        plain = build_plan(menu, TasteProfile(party_size=2))
        pork = [d for d in plain.shortlist if d.cn == "宫保鸡丁"][0]

        events = [event_from_item(menu[3], "like")]
        with_memory = build_plan(menu, TasteProfile(party_size=2), events)
        liked = [d for d in with_memory.shortlist if d.cn == "宫保鸡丁"][0]
        self.assertGreater(liked.score, pork.score)

    def test_a_disliked_dish_is_demoted_not_deleted(self):
        menu = mixed_menu()
        events = [event_from_item(menu[3], "dislike")]
        rec = build_plan(menu, TasteProfile(party_size=2), events)
        self.assertNotIn("宫保鸡丁", {slot.cn for slot in rec.plan.slots})
        self.assertIn("宫保鸡丁", {d.cn for d in rec.shortlist})

    def test_history_is_optional(self):
        rec = build_plan(mixed_menu(), TasteProfile())
        self.assertTrue(rec.shortlist)


if __name__ == "__main__":
    unittest.main()
