"""Tests for the taste profile, dish identity and the event log.

Two properties are load-bearing here and get their own tests: a dislike is not an
allergy, and a broken file on disk is a warning rather than a crash.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from menu_vision.models import DishInfo, EnrichedItem, MenuResult  # noqa: E402
from menu_vision.preferences import (  # noqa: E402
    DEFAULT_PROFILE,
    PROFILE_FILE,
    SCHEMA_VERSION,
    TasteProfile,
    append_event,
    clear_history,
    derive_profile,
    dish_key,
    event_from_item,
    load_events,
    load_profile,
    match_key,
    name_key,
    normalise_name,
    save_profile,
    venue_key,
)

DAY = 86400.0


def item(
    name: str,
    *,
    price: float | None = 16.0,
    category: str = "tofu",
    spice: int = 3,
    ingredients: tuple[str, ...] = ("tofu", "chilli"),
    allergens: tuple[str, ...] = (),
    vegetarian: bool | None = True,
    described: bool = True,
    item_id: int = 1,
) -> EnrichedItem:
    """A dish the model described, or a bare one when described=False."""
    if not described:
        return EnrichedItem(id=item_id, cn=name, price_value=price)
    return EnrichedItem(
        id=item_id,
        cn=name,
        price_value=price,
        price_raw=f"{price:g}元" if price is not None else None,
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


class DishIdentityTests(unittest.TestCase):
    def test_punctuation_and_spaces_do_not_split_the_key(self):
        keys = {
            name_key("酸辣土豆丝"),
            name_key("酸辣 土豆丝"),
            name_key("酸辣土豆丝（微辣）"),
            name_key(" 酸辣土豆丝 "),
        }
        self.assertEqual(len(keys), 1, keys)

    def test_full_width_characters_fold(self):
        self.assertEqual(name_key("卤蛋（２个）"), name_key("卤蛋"))

    def test_ocr_confusable_glyphs_merge(self):
        self.assertEqual(name_key("蜜汁叉烧"), name_key("密汁叉烧"))

    def test_a_different_ingredient_is_a_different_dish(self):
        # The 0.85 threshold exists to keep these apart: 丝 and 片 are two dishes,
        # and merging them would attribute a dislike to the wrong one.
        self.assertNotEqual(name_key("酸辣土豆丝"), name_key("酸辣土豆片"))
        self.assertNotEqual(
            match_key("酸辣土豆片", ["酸辣土豆丝"]), name_key("酸辣土豆丝")
        )

    def test_match_key_resolves_ocr_damage_but_not_a_different_dish(self):
        self.assertEqual(match_key("密汁叉烧", ["蜜汁叉烧"]), name_key("蜜汁叉烧"))
        self.assertEqual(match_key("水煮鱼", ["水煮牛肉"]), name_key("水煮鱼"))

    def test_the_same_name_at_two_restaurants_is_two_dish_keys(self):
        self.assertNotEqual(dish_key("麻婆豆腐", "u:place-a"), dish_key("麻婆豆腐", "u:place-b"))
        self.assertEqual(name_key("麻婆豆腐"), name_key("麻婆豆腐"))

    def test_empty_name_returns_empty_key(self):
        for text in ("", "   ", "（）", "()", "！？"):
            self.assertEqual(name_key(text), "", text)
        self.assertEqual(match_key("", ["麻婆豆腐"]), "")

    def test_venue_key_prefers_a_typed_name(self):
        result = MenuResult(items=[item("麻婆豆腐")])
        self.assertEqual(venue_key(result, "Chuan Ban"), "u:chuanban")

    def test_venue_key_fingerprints_the_dish_set(self):
        first = venue_key(MenuResult(items=[item("麻婆豆腐"), item("水煮鱼", item_id=2)]))
        again = venue_key(MenuResult(items=[item("水煮鱼"), item("麻婆豆腐", item_id=2)]))
        other = venue_key(MenuResult(items=[item("叉烧饭")]))
        self.assertTrue(first.startswith("p:"))
        self.assertEqual(first, again, "order must not change the fingerprint")
        self.assertNotEqual(first, other)


class DeriveProfileTests(unittest.TestCase):
    def setUp(self):
        self.now = time.time()

    def events(self, *specs, now=None):
        now = self.now if now is None else now
        return [event_from_item(spec, spec_kind, now=now) for spec, spec_kind in specs]

    def test_a_like_raises_ingredient_affinity(self):
        profile = derive_profile(self.events((item("麻婆豆腐"), "like")))
        self.assertGreater(profile.ingredient_weight("tofu"), 0.0)
        self.assertEqual(profile.avoid_allergens, ())

    def test_like_then_dislike_is_net_negative(self):
        dish = item("麻婆豆腐")
        events = self.events((dish, "like"), (dish, "dislike"))
        profile = derive_profile(events)
        self.assertLess(profile.ingredient_weight("tofu"), 0.0)

    def test_a_year_old_like_barely_counts(self):
        old = self.now - 365 * DAY
        fresh = derive_profile(self.events((item("麻婆豆腐"), "like")))
        stale = derive_profile(self.events((item("麻婆豆腐"), "like"), now=old))
        self.assertLess(
            abs(stale.ingredient_weight("tofu")), abs(fresh.ingredient_weight("tofu"))
        )

    def test_repeat_likes_become_a_named_preference(self):
        dish = item("麻婆豆腐")
        events = self.events(*[(dish, "like")] * 3)
        profile = derive_profile(events)
        self.assertIn("tofu", profile.liked_ingredients)

    def test_a_dislike_creates_a_penalty_not_a_filter(self):
        profile = derive_profile(
            self.events((item("香菜拌豆腐", ingredients=("tofu", "coriander")), "dislike"))
        )
        self.assertIn("coriander", profile.disliked_ingredients)
        # The whole point: inference may not write a hard restriction.
        self.assertEqual(profile.avoid_allergens, ())
        self.assertFalse(profile.vegetarian_only)

    def test_inference_never_adds_an_allergen(self):
        peanuts = item("宫保鸡丁", ingredients=("chicken", "peanut"), allergens=("peanut",))
        events = self.events(*[(peanuts, "dislike")] * 3)
        profile = derive_profile(events, TasteProfile(dietary_notes="no pork"))
        self.assertEqual(profile.avoid_allergens, ())
        self.assertEqual(profile.dietary_notes, "no pork")
        self.assertIn("peanut", profile.disliked_ingredients)

    def test_stated_restrictions_survive_derivation(self):
        base = TasteProfile(
            vegetarian_only=True, avoid_allergens=("peanut",), party_size=4, budget_total=200
        )
        profile = derive_profile(self.events((item("麻婆豆腐"), "like")), base)
        self.assertTrue(profile.vegetarian_only)
        self.assertEqual(profile.avoid_allergens, ("peanut",))
        self.assertEqual(profile.party_size, 4)
        self.assertEqual(profile.budget_total, 200)

    def test_spice_is_inferred_only_when_asked_for(self):
        hot = item("水煮鱼", spice=3)
        events = self.events(*[(hot, "like")] * 2)

        stated = derive_profile(events, TasteProfile(spice_tolerance=1))
        self.assertEqual(stated.spice_tolerance, 1)
        self.assertEqual(stated.inferred_spice, 3)

        learned = derive_profile(
            events, TasteProfile(spice_tolerance=1, spice_tolerance_source="inferred")
        )
        self.assertEqual(learned.spice_tolerance, 3)

    def test_spice_inference_needs_evidence(self):
        profile = derive_profile(self.events(), TasteProfile())
        self.assertIsNone(profile.inferred_spice)

    def test_spice_inference_ignores_un_enriched_events(self):
        bare = item("水煮鱼", described=False, spice=3)
        events = self.events(*[(bare, "like")] * 4)
        profile = derive_profile(
            events, TasteProfile(spice_tolerance=1, spice_tolerance_source="inferred")
        )
        self.assertIsNone(profile.inferred_spice)
        self.assertEqual(profile.spice_tolerance, 1)

    def test_un_enriched_events_teach_no_ingredients(self):
        bare = item("麻婆豆腐", described=False)
        profile = derive_profile(self.events(*[(bare, "like")] * 3))
        self.assertEqual(profile.ingredient_affinity, ())
        self.assertEqual(profile.liked_ingredients, ())
        self.assertEqual(profile.events_used, 3)

    def test_ubiquitous_ingredient_matters_less_than_a_distinctive_one(self):
        common = item("A", ingredients=("soy sauce", "pork"), item_id=1)
        also = item("B", ingredients=("soy sauce", "tofu"), item_id=2)
        events = self.events((common, "like"), (also, "like"))
        profile = derive_profile(events)
        self.assertGreater(
            profile.ingredient_weight("pork"), profile.ingredient_weight("soy sauce")
        )

    def test_no_events_clears_derived_fields_but_keeps_stated_ones(self):
        base = TasteProfile(party_size=6, avoid_allergens=("peanut",),
                            liked_ingredients=("tofu",))
        profile = derive_profile([], base)
        self.assertEqual(profile.party_size, 6)
        self.assertEqual(profile.avoid_allergens, ("peanut",))
        self.assertEqual(profile.liked_ingredients, ())
        self.assertEqual(profile.events_used, 0)

    def test_profile_is_never_mutated(self):
        before = TasteProfile()
        derive_profile(self.events((item("麻婆豆腐"), "like")), before)
        self.assertEqual(before.ingredient_affinity, ())

    def test_derivation_is_deterministic(self):
        events = self.events((item("麻婆豆腐"), "like"), (item("水煮鱼"), "dislike"))
        first = derive_profile(events, TasteProfile(), now=self.now)
        second = derive_profile(events, TasteProfile(), now=self.now)
        self.assertEqual(first.model_dump_json(), second.model_dump_json())


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_profile_is_not_an_error(self):
        profile, warnings = load_profile(self.dir)
        self.assertEqual(profile, DEFAULT_PROFILE)
        self.assertEqual(warnings, [])

    def test_truncated_json_returns_defaults_and_quarantines(self):
        (self.dir / PROFILE_FILE).write_text('{"party_size": 4, "budget', encoding="utf-8")
        profile, warnings = load_profile(self.dir)
        self.assertEqual(profile, DEFAULT_PROFILE)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(list(self.dir.glob("profile.json.corrupt-*")).__len__(), 1)

    def test_unknown_newer_version_is_kept_with_a_warning(self):
        payload = TasteProfile(party_size=5).model_dump()
        payload["version"] = SCHEMA_VERSION + 1
        (self.dir / PROFILE_FILE).write_text(json.dumps(payload), encoding="utf-8")
        profile, warnings = load_profile(self.dir)
        self.assertEqual(profile.party_size, 5)
        self.assertEqual(len(warnings), 1)

    def test_save_load_round_trip(self):
        original = TasteProfile(
            party_size=4, budget_total=180.0, avoid_allergens=("peanut", "shellfish"),
            spice_tolerance=2, dietary_notes="no pork",
        )
        self.assertIsNone(save_profile(original, self.dir))
        loaded, warnings = load_profile(self.dir)
        self.assertEqual(warnings, [])
        self.assertEqual(
            loaded.model_dump(exclude={"updated_at"}),
            original.model_dump(exclude={"updated_at"}),
        )

    def test_save_leaves_no_temp_file(self):
        save_profile(TasteProfile(party_size=3), self.dir)
        self.assertEqual(list(self.dir.glob("*.tmp*")), [])

    def test_events_round_trip_in_order(self):
        first = event_from_item(item("麻婆豆腐"), "like", now=1000.0)
        second = event_from_item(item("水煮鱼"), "dislike", now=2000.0)
        self.assertIsNone(append_event(first, self.dir))
        self.assertIsNone(append_event(second, self.dir))
        events, warnings = load_events(self.dir)
        self.assertEqual(warnings, [])
        self.assertEqual([e.cn for e in events], ["麻婆豆腐", "水煮鱼"])
        self.assertEqual(events[1].kind, "dislike")

    def test_partially_written_last_line_is_dropped_with_a_warning(self):
        good = event_from_item(item("麻婆豆腐"), "like", now=1000.0)
        (self.dir / "events.jsonl").write_text(
            good.model_dump_json() + "\n" + '{"ts": 1, "kind": "li', encoding="utf-8"
        )
        events, warnings = load_events(self.dir)
        self.assertEqual(len(events), 1)
        self.assertEqual(len(warnings), 1)

    def test_valid_tail_without_a_newline_is_kept(self):
        # The inverse of the test above: a record that was written but not synced
        # is still valid, and silently dropping it would lose a real tap.
        good = event_from_item(item("麻婆豆腐"), "like", now=1000.0)
        (self.dir / "events.jsonl").write_text(good.model_dump_json(), encoding="utf-8")
        events, warnings = load_events(self.dir)
        self.assertEqual(len(events), 1)
        self.assertEqual(warnings, [])

    def test_append_after_a_torn_tail_does_not_corrupt(self):
        good = event_from_item(item("麻婆豆腐"), "like", now=1000.0)
        (self.dir / "events.jsonl").write_text(
            good.model_dump_json() + "\n" + '{"ts": 1, "kin', encoding="utf-8"
        )
        append_event(event_from_item(item("水煮鱼"), "like", now=2000.0), self.dir)
        events, warnings = load_events(self.dir)
        self.assertEqual([e.cn for e in events], ["麻婆豆腐", "水煮鱼"])
        self.assertEqual(len(warnings), 1)

    def test_clear_history_keeps_the_stated_profile(self):
        append_event(event_from_item(item("麻婆豆腐"), "like"), self.dir)
        save_profile(TasteProfile(party_size=8), self.dir)
        self.assertIsNone(clear_history(self.dir))
        events, _ = load_events(self.dir)
        self.assertEqual(events, [])
        profile, _ = load_profile(self.dir)
        self.assertEqual(profile.party_size, 8)

    def test_clear_history_on_an_empty_directory_is_harmless(self):
        self.assertIsNone(clear_history(self.dir))

    def test_event_snapshot_is_self_contained(self):
        source = item("麻婆豆腐", price=16.0, ingredients=("tofu", "chilli"))
        event = event_from_item(source, "like", venue="u:x", menu_source="menu.jpg")
        # The dish object can disappear; the event still describes what was seen.
        del source
        self.assertEqual(event.price_value, 16.0)
        self.assertIn("tofu", event.ingredients)
        self.assertTrue(event.facts_known)
        self.assertEqual(event.venue, "u:x")


if __name__ == "__main__":
    unittest.main()
