"""Build a meal, not a list.

Two rules shape this module, both inherited from `pricing.py`:

1. **The score is computed, never described.** Eight weighted components, all
   reading fields that already exist on the dish. The user never sees the number
   - they see sentences, and every sentence is true on its own regardless of what
   the weights are.

2. **Exclusion requires evidence.** An allergen filter only excludes a dish whose
   allergen list actually names the allergen. An empty list means "nobody
   checked", not "safe", and an un-enriched scan means nobody checked anything.
   Excluding on absence would empty the page on exactly the menus where help
   matters most, and - worse - imply a dish is safe when the app has no idea.

Nothing here touches the network, so a menu scanned with the API key down still
gets a plan; it just says so.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict

from .models import EnrichedItem, has_dish_facts
from .preferences import PreferenceEvent, TasteProfile, name_key
from .pricing import format_price, unit_of

__all__ = [
    "Reason",
    "ScoredDish",
    "PlanSlot",
    "MealPlan",
    "Recommendation",
    "Weights",
    "DEFAULT_WEIGHTS",
    "build_plan",
    "hard_exclusion",
    "has_dish_facts",
    "slot_of",
    "slot_template",
    "budget_line",
]

# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #
MAX_PARTY = 12
MAX_DISHES = 10

# A dish has to be cheap relative to a person's share to score well here: one
# shared plate eating less than a third of a head's budget is comfortable.
BUDGET_HEADROOM = 0.35

# Score for a dish that is `gap` levels hotter than the stated tolerance. Graded,
# because one chilli more is a garnish and three more is a different dish.
SPICE_SCORE: dict[int, float] = {0: 1.0, 1: 0.65, 2: 0.35, 3: 0.10}

# A dish whose safety or suitability is unknown cannot outrank one that is known
# to be fine, but it is not deleted either.
UNKNOWN_CAP = 0.30
# The user explicitly said they did not like this dish. Not an exclusion (tastes
# change), just a hard demotion.
DISLIKED_CAP = 0.15

# Do not churn the plan over float noise or a 50-fen rounding difference.
BUDGET_TOLERANCE = 0.5

SHORTLIST_SIZE = 12


class Weights(BaseModel):
    """How much each component matters. Frozen so a test can pin it."""

    model_config = ConfigDict(frozen=True)

    spice: float = 0.15
    budget: float = 0.20
    ingredient: float = 0.15
    category: float = 0.10
    peers: float = 0.10
    novelty: float = 0.10
    crowding: float = 0.05
    confidence: float = 0.15


DEFAULT_WEIGHTS = Weights()


# --------------------------------------------------------------------------- #
# What kind of course is this
# --------------------------------------------------------------------------- #
# Checked in this order. A match on the model's `category` always wins; the
# printed Chinese name is consulted only when the category does not match, since
# OCR text is ground truth while a category is the model's opinion.
_SLOT_ORDER: tuple[str, ...] = ("soup", "staple", "protein", "vegetable", "drink")

_SLOT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "soup": ("soup", "broth", "汤", "羹"),
    "staple": (
        "rice", "noodle", "noodles", "congee", "porridge", "dumpling", "dumplings",
        "bun", "buns", "bread", "pancake", "wonton", "vermicelli",
        "饭", "面", "麺", "粉", "粥", "饺", "包", "饼", "馒", "馄饨",
    ),
    "protein": (
        "meat", "pork", "beef", "chicken", "duck", "lamb", "mutton", "fish", "seafood",
        "shrimp", "prawn", "skewer", "skewers", "bbq", "barbecue", "roast", "egg",
        "hotpot", "intestine", "sausage", "ribs", "rib",
        "肉", "鸡", "牛", "猪", "鱼", "虾", "蛋", "串", "烤", "烧", "叉", "鹅", "鸭",
        "羊", "排骨", "肠",
    ),
    "vegetable": (
        "vegetable", "vegetables", "greens", "mushroom", "salad", "tofu", "bean curd",
        "菜", "豆", "菇", "笋", "藕", "蔬", "瓜", "茄",
    ),
    "drink": (
        "drink", "beverage", "tea", "juice", "beer", "soda", "water", "wine", "cider",
        "酒", "茶", "汁", "饮", "汽水",
    ),
}


def _hits(text: str, keyword: str) -> bool:
    """Substring for Chinese, whole word for English.

    `tea in "steamed bun"` is true, which would turn every steamed dish into a
    beverage; a word boundary fixes that without needing a stemmer.
    """
    if keyword.isascii():
        return re.search(rf"\b{re.escape(keyword)}s?\b", text) is not None
    return keyword in text


def slot_of(item: EnrichedItem) -> str:
    """Which course slot this dish fills: soup/staple/protein/vegetable/drink/other."""
    category = (item.dish.category or "").strip().lower()
    for slot in _SLOT_ORDER:
        if category and any(_hits(category, word) for word in _SLOT_KEYWORDS[slot]):
            return slot
    label = (item.cn or "").strip().lower()
    for slot in _SLOT_ORDER:
        if label and any(_hits(label, word) for word in _SLOT_KEYWORDS[slot]):
            return slot
    return "other"


def slot_template(party_size: int) -> tuple[str, ...]:
    """How many of each course a table of this size wants.

    Chinese meals are shared: roughly one dish per person plus one, weighted
    towards protein as the table grows. Staples come first because they anchor
    the budget, soup before the mains because it is the most droppable course.
    """
    people = max(1, min(MAX_PARTY, party_size))
    vegetables = 1 + (people - 1) // 3
    proteins = 1 + (people - 1) // 2
    staples = 1 if people <= 4 else 2
    soups = 1 if people >= 2 else 0
    order = (
        ("staple",) * staples
        + ("soup",) * soups
        + ("vegetable",) * vegetables
        + ("protein",) * proteins
    )
    return order[:MAX_DISHES]


# --------------------------------------------------------------------------- #
# Result models
# --------------------------------------------------------------------------- #
class Reason(BaseModel):
    """One sentence explaining a dish, and the number it is built on."""

    model_config = ConfigDict(frozen=True)

    component: str
    text: str
    polarity: Literal["pro", "con", "neutral"]
    value: float = 0.0


class ScoredDish(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: int
    cn: str
    en: str = ""
    price_value: float | None = None
    price_display: str = ""
    category: str = ""
    slot: str = "other"
    score: float = 0.0
    # Share of the scoring weight that had real data behind it for this dish.
    evidence: float = 0.0
    facts_known: bool = False
    vegetarian: bool | None = None
    spice_level: int = 0
    ingredients: tuple[str, ...] = ()
    allergens: tuple[str, ...] = ()
    reasons: tuple[Reason, ...] = ()
    excluded_reason: str | None = None


class PlanSlot(BaseModel):
    model_config = ConfigDict(frozen=True)

    slot: str
    item_id: int
    cn: str
    en: str = ""
    price_value: float | None = None
    price_display: str = ""
    score: float = 0.0
    why: tuple[str, ...] = ()


class MealPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    slots: tuple[PlanSlot, ...] = ()
    # Sum of the picks that carry a price. None when nothing in the plan is
    # priced - not 0.0, which would read as "free".
    total_known: float | None = None
    unpriced_count: int = 0
    per_head: float | None = None
    budget: float | None = None
    shortfall: float | None = None
    missing: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


class Recommendation(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan: MealPlan = MealPlan()
    shortlist: tuple[ScoredDish, ...] = ()
    excluded: tuple[ScoredDish, ...] = ()
    # Fraction of the scoring weight that had real data behind it. Below 0.5 the
    # UI says the menu was ranked on price alone.
    evidence: float = 0.0
    unknown_allergen_ids: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Shared facts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _PeerGroup:
    median: float
    count: int
    label: str


@dataclass(frozen=True)
class _Context:
    profile: TasteProfile
    items: tuple[EnrichedItem, ...]
    priced: tuple[EnrichedItem, ...]
    ingredient_counts: dict[str, int]
    candidates: int
    interactions: dict[str, int]  # name_key -> likes + orders
    disliked: frozenset[str]      # name_keys explicitly disliked


def _peer_group(item: EnrichedItem, priced: Sequence[EnrichedItem]) -> _PeerGroup | None:
    """Comparable prices for this dish, or None when there are too few.

    Same logic as `pricing.py`: never compare a rice bowl with a plate of ribs,
    and never compare a half bird with a whole one.
    """
    if item.price_value is None:
        return None
    unit = unit_of(item.price_raw)
    category = (item.dish.category or "").strip().lower()
    same_unit = [other for other in priced if unit_of(other.price_raw) == unit]
    same_category = [
        other
        for other in same_unit
        if category and (other.dish.category or "").strip().lower() == category
    ]
    peers = same_category if len(same_category) >= 3 else same_unit
    values = [peer.price_value for peer in peers if peer.price_value is not None]
    if len(values) < 3:
        return None
    median = statistics.median(values)
    if median <= 0:
        return None
    label = category if len(same_category) >= 3 and category else ""
    return _PeerGroup(median=median, count=len(values), label=label)


def _context(
    items: Sequence[EnrichedItem],
    profile: TasteProfile,
    events: Sequence[PreferenceEvent],
) -> _Context:
    interactions: dict[str, int] = {}
    disliked: set[str] = set()
    for event in events:
        key = event.name_key or name_key(event.cn)
        if not key:
            continue
        if event.kind in ("ordered", "like"):
            interactions[key] = interactions.get(key, 0) + 1
        elif event.kind == "dislike":
            disliked.add(key)

    counts: dict[str, int] = {}
    for item in items:
        for token in {t.strip().lower() for t in item.dish.ingredients if t.strip()}:
            counts[token] = counts.get(token, 0) + 1

    return _Context(
        profile=profile,
        items=tuple(items),
        priced=tuple(item for item in items if item.price_value is not None),
        ingredient_counts=counts,
        candidates=len(items),
        interactions=interactions,
        disliked=frozenset(disliked),
    )


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def _allergen_overlap(item: EnrichedItem, avoid: Sequence[str]) -> str | None:
    """The first avoided allergen the menu actually names, or None."""
    listed = {token.strip().lower() for token in item.dish.allergens if token.strip()}
    if not listed:
        return None
    for allergen in avoid:
        needle = allergen.strip().lower()
        if not needle:
            continue
        for token in listed:
            if needle == token or needle in token or token in needle:
                return token
    return None


def hard_exclusion(item: EnrichedItem, profile: TasteProfile) -> str | None:
    """The sentence explaining why this dish is out, or None.

    Only two things exclude: an allergen the menu names, and meat when the user
    asked for vegetarian. Everything else is a score, because everything else can
    be worked around at the table.
    """
    overlap = _allergen_overlap(item, profile.avoid_allergens)
    if overlap:
        return f"Contains {overlap}, which you asked to avoid."
    if profile.vegetarian_only and item.dish.vegetarian is False:
        return "Contains meat or fish, and you asked for vegetarian."
    return None


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _components(
    item: EnrichedItem, ctx: _Context, weights: Weights
) -> dict[str, tuple[float, bool]]:
    """Component name -> (score in [0, 1], whether it had data to work with).

    An inactive component is removed from the average rather than scored 0.5: a
    menu with no prices must not be penalised against one with prices, and a dish
    nobody described is not "average", it is unknown.
    """
    dish = item.dish
    facts = has_dish_facts(item)
    profile = ctx.profile
    out: dict[str, tuple[float, bool]] = {}

    # Spice: graded, and never a reason to exclude - you can ask for less chilli.
    if facts:
        gap = max(0, min(3, int(dish.spice_level) - int(profile.spice_tolerance)))
        out["spice"] = (SPICE_SCORE[gap], True)
    else:
        out["spice"] = (0.0, False)

    # Budget: how much of one person's share this dish would eat.
    if profile.budget_total and item.price_value is not None:
        per_head = profile.budget_total / max(1, profile.party_size)
        if per_head > 0:
            ratio = item.price_value / per_head
            share = max(0.0, min(1.0, (1.0 - ratio) / (1.0 - BUDGET_HEADROOM)))
            out["budget"] = (share, True)
        else:
            out["budget"] = (0.0, False)
    else:
        out["budget"] = (0.0, False)

    # Ingredients: the soft memory of what this person actually enjoys.
    tokens = [t.strip().lower() for t in dish.ingredients if t.strip()]
    if facts and tokens:
        mean = sum(profile.ingredient_weight(token) for token in tokens) / len(tokens)
        out["ingredient"] = (max(0.0, min(1.0, 0.5 + 0.5 * mean)), True)
    else:
        out["ingredient"] = (0.5, False)

    # Category: meaningless with no history - a fresh profile is ignorant, not
    # neutral - so it only participates once there is something to go on.
    category = (dish.category or "").strip().lower()
    if facts and category and profile.category_affinity:
        weight = profile.category_weight(category)
        out["category"] = (max(0.0, min(1.0, 0.5 + 0.5 * weight)), True)
    else:
        out["category"] = (0.5, False)

    # Price against comparable dishes on this menu.
    group = _peer_group(item, ctx.priced)
    if group is not None and item.price_value is not None:
        delta = (group.median - item.price_value) / group.median
        out["peers"] = (max(0.0, min(1.0, 0.5 + 0.5 * max(-1.0, min(1.0, delta)))), True)
    else:
        out["peers"] = (0.5, False)

    # Novelty: how well the dish matches how adventurous this person said they are.
    key = name_key(item.cn)
    seen = ctx.interactions.get(key, 0)
    familiarity = 0.0 if seen >= 2 else (0.6 if seen == 1 else 1.0)
    out["novelty"] = (
        1.0 - abs(familiarity - max(0.0, min(1.0, profile.adventurousness))),
        True,
    )

    # Crowding: is this dish one of eight near-identical pork plates?
    out["crowding"] = _crowding(item, ctx)

    out["confidence"] = (float(dish.confidence) if facts else 0.0, facts)
    return out


def _crowding(item: EnrichedItem, ctx: _Context) -> tuple[float, bool]:
    """(1 - share of other candidates sharing a distinctive ingredient, active)."""
    tokens = {t.strip().lower() for t in item.dish.ingredients if t.strip()}
    if not tokens or ctx.candidates < 2:
        return (0.5, False)
    # An ingredient in more than half the menu is a house base, not a signature.
    distinctive = {
        token
        for token in tokens
        if ctx.ingredient_counts.get(token, 0) * 2 <= ctx.candidates
    }
    if not distinctive:
        return (0.5, False)
    neighbours = 0
    for other in ctx.items:
        if other is item:
            continue
        other_tokens = {t.strip().lower() for t in other.dish.ingredients if t.strip()}
        if distinctive & other_tokens:
            neighbours += 1
    return (1.0 - neighbours / (ctx.candidates - 1), True)


def _weigh(
    item: EnrichedItem, ctx: _Context, weights: Weights
) -> tuple[float, dict[str, tuple[float, bool]]]:
    """Raw score and the component breakdown that produced it."""
    components = _components(item, ctx, weights)
    active = [
        (getattr(weights, name), value)
        for name, (value, enabled) in components.items()
        if enabled
    ]
    total = sum(weight for weight, _ in active)
    if total <= 0:
        return 0.0, components
    return round(sum(weight * value for weight, value in active) / total, 4), components


def _evidence(components: dict[str, tuple[float, bool]], weights: Weights) -> float:
    """Share of the total weight that had real data behind it."""
    all_weight = sum(
        getattr(weights, name) for name in components
    )
    if all_weight <= 0:
        return 0.0
    active = sum(getattr(weights, name) for name, (_, on) in components.items() if on)
    return round(active / all_weight, 4)


# --------------------------------------------------------------------------- #
# Explanations
# --------------------------------------------------------------------------- #
def _money(value: float) -> str:
    """`16`, `18.5`, `7.65` - no trailing zeros, never a rounded lie."""
    return f"{round(value, 2):g}"


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def explain(
    item: EnrichedItem,
    ctx: _Context,
    components: dict[str, tuple[float, bool]],
) -> tuple[Reason, ...]:
    """The sentences shown under a dish.

    Every number here is a field value or a statistic computed in this module.
    No percentages, no ratios, no "much cheaper": if a sentence would need a
    number that is not already in the data, it does not get written.
    """
    dish = item.dish
    profile = ctx.profile
    facts = has_dish_facts(item)
    reasons: list[Reason] = []

    # Budget, when there is one to talk about.
    if components["budget"][1] and item.price_value is not None:
        per_head = round(profile.budget_total / max(1, profile.party_size), 2)
        score = components["budget"][0]
        reasons.append(
            Reason(
                component="budget",
                text=(
                    f"¥{_money(item.price_value)} against your "
                    f"¥{_money(profile.budget_total)} for {profile.party_size} - "
                    f"about ¥{_money(per_head)} each."
                ),
                polarity="pro" if score >= 0.66 else ("con" if score <= 0.33 else "neutral"),
                value=item.price_value,
            )
        )
    elif item.price_value is None:
        reasons.append(
            Reason(
                component="price",
                text="No price is printed for this dish.",
                polarity="neutral",
            )
        )

    # Spice, three distinct sentences because there are three distinct situations.
    if facts:
        level = int(dish.spice_level)
        tolerance = int(profile.spice_tolerance)
        gap = level - tolerance
        if gap <= 0:
            reasons.append(
                Reason(
                    component="spice",
                    text=f"Spice {level}/3 - within your {tolerance}/3.",
                    polarity="pro",
                    value=components["spice"][0],
                )
            )
        elif gap == 1:
            reasons.append(
                Reason(
                    component="spice",
                    text=(
                        f"Spice {level}/3 is above your {tolerance}/3 - "
                        "ask for 微辣 (a little chilli)."
                    ),
                    polarity="con",
                    value=components["spice"][0],
                )
            )
        else:
            reasons.append(
                Reason(
                    component="spice",
                    text=(
                        f"Spice {level}/3 against your {tolerance}/3 - ask for "
                        "不辣 (no chilli); this one is built on chilli."
                    ),
                    polarity="con",
                    value=components["spice"][0],
                )
            )

    # Ingredients: name the single strongest hit rather than listing everything.
    if components["ingredient"][1]:
        tokens = [t.strip().lower() for t in dish.ingredients if t.strip()]
        best_token, best_weight = "", 0.0
        for token in sorted(set(tokens)):
            weight = profile.ingredient_weight(token)
            if abs(weight) > abs(best_weight):
                best_token, best_weight = token, weight
        if best_token and best_weight >= 0.3:
            reasons.append(
                Reason(
                    component="ingredient",
                    text=f"Contains {best_token}, which you liked before.",
                    polarity="pro",
                    value=best_weight,
                )
            )
        elif best_token and best_weight <= -0.3:
            reasons.append(
                Reason(
                    component="ingredient",
                    text=f"Contains {best_token}, which you disliked before.",
                    polarity="con",
                    value=best_weight,
                )
            )

    # Category history.
    category = (dish.category or "").strip().lower()
    if components["category"][1] and category:
        weight = profile.category_weight(category)
        if weight >= 0.3:
            reasons.append(
                Reason(
                    component="category",
                    text=f"You have liked {plural_word(category)} dishes before.",
                    polarity="pro",
                    value=weight,
                )
            )

    # Price against comparable dishes.
    group = _peer_group(item, ctx.priced)
    if group is not None and item.price_value is not None:
        if item.price_value < group.median:
            reasons.append(
                Reason(
                    component="peers",
                    text=(
                        f"Below the median ¥{_money(group.median)} of the "
                        f"{_plural(group.count, group.label or 'priced dish')} here."
                    ),
                    polarity="pro",
                    value=group.median,
                )
            )
        elif item.price_value > group.median:
            reasons.append(
                Reason(
                    component="peers",
                    text=(
                        f"Above the median ¥{_money(group.median)} of the "
                        f"{_plural(group.count, group.label or 'priced dish')} here."
                    ),
                    polarity="con",
                    value=group.median,
                )
            )

    # Memory of this exact dish.
    key = name_key(item.cn)
    seen = ctx.interactions.get(key, 0)
    if key and key in ctx.disliked:
        reasons.append(
            Reason(
                component="memory",
                text="You disliked this before.",
                polarity="con",
            )
        )
    elif seen == 1:
        reasons.append(
            Reason(component="memory", text="You liked or ordered this once before.",
                   polarity="neutral")
        )
    elif seen >= 2:
        reasons.append(
            Reason(
                component="memory",
                text=f"You liked or ordered this {_plural(seen, 'time')} before.",
                polarity="neutral",
                value=float(seen),
            )
        )
    elif seen == 0:
        reasons.append(Reason(component="memory", text="New to you.", polarity="neutral"))

    # Unknowns, stated as unknowns.
    if not facts:
        reasons.append(
            Reason(
                component="facts",
                text="The menu does not say what is in this dish.",
                polarity="neutral",
            )
        )
    elif profile.avoid_allergens and not dish.allergens:
        reasons.append(
            Reason(
                component="allergens",
                text="Allergen status is not stated on this menu.",
                polarity="con",
            )
        )
    if profile.vegetarian_only and facts and dish.vegetarian is None:
        reasons.append(
            Reason(
                component="vegetarian",
                text="Vegetarian status is not stated on this menu.",
                polarity="con",
            )
        )
    return tuple(reasons)


def plural_word(word: str) -> str:
    """`noodle` -> `noodles`, for a sentence that cannot know the count."""
    if not word:
        return ""
    return word if word.endswith("s") else word + "s"


# --------------------------------------------------------------------------- #
# Building the plan
# --------------------------------------------------------------------------- #
def _scored(item: EnrichedItem, ctx: _Context, weights: Weights) -> ScoredDish:
    score, components = _weigh(item, ctx, weights)
    profile = ctx.profile
    facts = has_dish_facts(item)
    dish = item.dish

    # Caps, in increasing severity. None of these delete the dish - it stays
    # reachable in the shortlist - but a dish known to be fine must outrank one
    # nobody has information about.
    if profile.vegetarian_only and facts and dish.vegetarian is None:
        score = min(score, UNKNOWN_CAP)
    if profile.avoid_allergens and not dish.allergens:
        score = min(score, UNKNOWN_CAP)
    key = name_key(item.cn)
    if key and key in ctx.disliked:
        score = min(score, DISLIKED_CAP)

    return ScoredDish(
        item_id=item.id,
        cn=item.cn,
        en=dish.en or "",
        price_value=item.price_value,
        price_display=item.price_display or format_price(item.price_value, item.currency),
        category=(dish.category or "").strip().lower(),
        slot=slot_of(item),
        score=round(score, 4),
        evidence=_evidence(components, weights),
        facts_known=facts,
        vegetarian=dish.vegetarian,
        spice_level=int(dish.spice_level),
        ingredients=tuple(t.strip().lower() for t in dish.ingredients if t.strip()),
        allergens=tuple(t.strip().lower() for t in dish.allergens if t.strip()),
        reasons=explain(item, ctx, components),
        excluded_reason=hard_exclusion(item, profile),
    )


def _sort_key(dish: ScoredDish) -> tuple:
    """A total order, so the same menu always produces the same plan."""
    return (
        -round(dish.score, 4),
        dish.price_value is None,          # priced dishes first
        dish.price_value if dish.price_value is not None else 0.0,
        dish.cn,
        dish.item_id,
    )


def _total(chosen: Sequence[tuple[str, ScoredDish]]) -> float | None:
    values = [dish.price_value for _, dish in chosen if dish.price_value is not None]
    return round(sum(values), 2) if values else None


def _best_swap(
    chosen: list[tuple[str, ScoredDish]],
    pools: dict[str, list[ScoredDish]],
    used: set[int],
) -> tuple[int, ScoredDish] | None:
    """Swap one dish for a cheaper one of the same course, losing the least score.

    Always strictly cheaper, so the repair loop cannot oscillate.
    """
    best: tuple[float, int, ScoredDish] | None = None
    for index, (slot, current) in enumerate(chosen):
        if current.price_value is None:
            continue
        for candidate in pools.get(slot, []):
            if candidate.item_id in used or candidate.price_value is None:
                continue
            if candidate.price_value >= current.price_value:
                continue
            loss = candidate.score - current.score
            if best is None or loss > best[0]:
                best = (loss, index, candidate)
    return None if best is None else (best[1], best[2])


def _droppable(chosen: Sequence[tuple[str, ScoredDish]]) -> int | None:
    """The most expendable dish: soup, then spare vegetables, then spare protein.

    Never the staple - a meal without rice is not a cheaper meal, it is a
    different one - and never the last dish of any kind.
    """
    if len(chosen) <= 1:
        return None
    slots = [slot for slot, _ in chosen]
    for target, keep in (("soup", 0), ("vegetable", 1), ("protein", 1)):
        indexes = [index for index, slot in enumerate(slots) if slot == target]
        if len(indexes) > keep:
            return indexes[-1]
    return None


def _most_expensive(chosen: Sequence[tuple[str, ScoredDish]]) -> int | None:
    """The priciest dish, cheapest-scoring first among ties.

    Used only when the polite dropping order has nothing left to offer and the
    plan is still over budget: a person with ¥18 for four people is better served
    by two bowls of noodles than by five and a bill they cannot pay.
    """
    priced = [
        (index, dish.price_value)
        for index, (_, dish) in enumerate(chosen)
        if dish.price_value is not None
    ]
    if not priced:
        return None
    index, _ = max(priced, key=lambda pair: (pair[1], -chosen[pair[0]][1].score))
    return index


def _pick_slot(
    slot: str,
    pools: dict[str, list[ScoredDish]],
    used: set[int],
    chosen: Sequence[tuple[str, ScoredDish]],
    disliked: frozenset[str],
) -> ScoredDish | None:
    """The best unused candidate for this course, avoiding dishes already disliked."""
    def usable(candidates: Sequence[ScoredDish]) -> list[ScoredDish]:
        return [
            dish
            for dish in candidates
            if dish.item_id not in used and name_key(dish.cn) not in disliked
        ]

    pool = usable(pools.get(slot, ()))
    if not pool:
        # Plenty of menus have dishes the model could not classify; filling the
        # course with one of those beats leaving a hole in the meal.
        pool = usable(pools.get("other", ()))
    if not pool:
        return None

    if slot == "protein" and any(s == "protein" for s, _ in chosen):
        # Vary the main: a second protein sharing nothing with the first beats a
        # second pork dish.
        taken = {token for _, other in chosen if other.slot == "protein"
                 for token in other.ingredients}
        for candidate in pool:
            if not (set(candidate.ingredients) & taken):
                return candidate
    return pool[0]


def build_plan(
    items: Sequence[EnrichedItem],
    profile: TasteProfile,
    events: Sequence[PreferenceEvent] = (),
    *,
    weights: Weights = DEFAULT_WEIGHTS,
    shortlist_size: int = SHORTLIST_SIZE,
) -> Recommendation:
    """Score the menu, pick a meal for the table, and say what could not be checked."""
    if not items:
        note = "No dishes were read from this photo."
        return Recommendation(plan=MealPlan(notes=(note,)), notes=(note,))

    ctx = _context(items, profile, events)
    dishes = [_scored(item, ctx, weights) for item in items]
    eligible = [dish for dish in dishes if dish.excluded_reason is None]
    excluded = [dish for dish in dishes if dish.excluded_reason is not None]
    eligible.sort(key=_sort_key)

    # The menu was analysed if any dish carries the enrichment the scorer needs.
    evidence = max((dish.evidence for dish in eligible), default=0.0)
    budget = (
        profile.budget_total
        if profile.budget_total and profile.budget_total > 0
        else None
    )
    template = slot_template(profile.party_size)

    pools: dict[str, list[ScoredDish]] = {}
    for dish in eligible:
        pools.setdefault(dish.slot, []).append(dish)

    chosen: list[tuple[str, ScoredDish]] = []
    used: set[int] = set()
    missing: list[str] = []
    for slot in template:
        pick = _pick_slot(slot, pools, used, chosen, ctx.disliked)
        if pick is None:
            missing.append(slot)
            continue
        used.add(pick.item_id)
        chosen.append((slot, pick))

    # A one-course menu - a noodle shop, a skewer stall - has nothing to fill the
    # other courses with. Serving two dishes to four people is a worse answer than
    # serving four bowls of noodles, so spend the gap on the course the menu
    # actually offers until it runs out.
    if missing and pools:
        filler = max(sorted(pools), key=lambda name: len(pools[name]))
        unfilled: list[str] = []
        for slot in missing:
            pick = _pick_slot(filler, pools, used, chosen, ctx.disliked)
            if pick is None:
                unfilled.append(slot)
                continue
            used.add(pick.item_id)
            chosen.append((filler, pick))
        missing = unfilled

    dropped: list[str] = []
    if budget is not None and chosen:
        # Never shrink below half a dish per person: at that point the honest
        # answer is "this budget does not feed this table", which the shortfall
        # note says in words.
        floor = max(1, profile.party_size // 2)
        for _ in range(2 * max(1, len(chosen))):
            spent = _total(chosen)
            if spent is None or spent <= budget + BUDGET_TOLERANCE:
                break
            swap = _best_swap(chosen, pools, used)
            if swap is not None:
                index, candidate = swap
                used.discard(chosen[index][1].item_id)
                used.add(candidate.item_id)
                chosen[index] = (chosen[index][0], candidate)
                continue
            victim = _droppable(chosen)
            if victim is None:
                if len(chosen) <= floor:
                    break
                victim = _most_expensive(chosen)
                if victim is None:
                    break
            dropped.append(chosen[victim][0])
            used.discard(chosen[victim][1].item_id)
            chosen.pop(victim)

    total_known = _total(chosen)
    unpriced = sum(1 for _, dish in chosen if dish.price_value is None)
    per_head = (
        round(total_known / max(1, profile.party_size), 2)
        if total_known is not None
        else None
    )
    shortfall = (
        round(total_known - budget, 2)
        if total_known is not None and budget is not None and total_known > budget
        else None
    )

    # A dish whose allergen list is empty is unknown, not safe. Say so once.
    unknown_allergens = tuple(
        dish.item_id
        for _, dish in chosen
        if profile.avoid_allergens and not dish.allergens
    )

    notes: list[str] = []
    if shortfall is not None:
        notes.append(
            f"The cheapest plan for {profile.party_size} here is "
            f"¥{_money(total_known)}, which is ¥{_money(shortfall)} over your "
            f"¥{_money(budget)}."
        )
    elif budget is not None and total_known is None and chosen:
        notes.append("This menu prints no prices, so the budget could not be checked.")
    elif budget is not None and unpriced:
        notes.append(
            f"¥{_money(total_known)} plus {_plural(unpriced, 'dish', 'dishes')} "
            "with no printed price."
        )
    if unknown_allergens:
        notes.append(
            f"{_plural(len(unknown_allergens), 'dish', 'dishes')} in this plan have "
            "unknown allergen status - ask the staff."
        )
    if missing and len(chosen) < len(template):
        # Only worth saying when the plan really is shorter. On a one-course menu
        # the gap was already filled with more of what the kitchen makes, and
        # announcing a missing "protein" course would just be noise.
        courses = ", ".join(sorted(set(missing)))
        notes.append(
            f"This menu has no {courses}, so the plan is "
            f"{_plural(len(chosen), 'dish', 'dishes')} rather than "
            f"{len(template)}."
        )
    if dropped:
        notes.append(
            f"Dropped {_plural(len(dropped), 'dish', 'dishes')} to fit your budget."
        )
    if evidence < 0.5:
        notes.append(
            "This menu was not analysed, so dishes are ranked on price and your "
            "history only."
        )
    if not eligible:
        notes.append(
            "Every dish was excluded by your restrictions. Relax one to see the menu."
        )

    slots = tuple(
        PlanSlot(
            slot=slot,
            item_id=dish.item_id,
            cn=dish.cn,
            en=dish.en,
            price_value=dish.price_value,
            price_display=dish.price_display,
            score=dish.score,
            why=tuple(reason.text for reason in dish.reasons[:3]),
        )
        for slot, dish in chosen
    )

    return Recommendation(
        plan=MealPlan(
            slots=slots,
            total_known=total_known,
            unpriced_count=unpriced,
            per_head=per_head,
            budget=budget,
            shortfall=shortfall,
            missing=tuple(dict.fromkeys(missing)),
            notes=tuple(notes),
        ),
        shortlist=tuple(eligible[:shortlist_size]),
        excluded=tuple(excluded),
        evidence=evidence,
        unknown_allergen_ids=unknown_allergens,
        notes=tuple(notes),
    )


def budget_line(plan: MealPlan) -> str:
    """One line for the header: what this meal costs, and against what."""
    if not plan.slots:
        return "No plan yet."
    dishes = _plural(len(plan.slots), "dish", "dishes")
    if plan.total_known is None:
        return f"{dishes} - no prices printed, so the total is unknown."
    line = f"{dishes} - ¥{_money(plan.total_known)}"
    if plan.unpriced_count:
        line += f" plus {plan.unpriced_count} with no printed price"
    if plan.budget:
        line += f" of your ¥{_money(plan.budget)}"
    if plan.per_head is not None:
        line += f" (¥{_money(plan.per_head)} each)"
    return line + "."
