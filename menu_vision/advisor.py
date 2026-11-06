"""Answering questions about a dish, or about the menu.

The ordering here is deliberate and it is the opposite of the obvious one. Most
questions a person asks in front of a menu - how hot is it, does it have pork,
what does it cost, is it enough for four - are questions the scan has already
answered. Those are computed here, instantly and for free. The language model is
the escape hatch for everything else: the odd question, the comparison, the
"what would you order if I hate coriander".

That inversion matters because the model is the slow, unreliable path in this
project. It sits behind a free tier that answers 429 routinely, it takes tens of
seconds, and it does not exist at all without an API key. It should not be
standing in front of a question that a dictionary lookup can answer.

When the model is used, it is given only the facts the scan produced and is told
to say "the menu doesn't say" rather than fill gaps from what it knows about the
dish. It knows what 麻婆豆腐 usually contains; that knowledge is exactly what must
not appear in the answer.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict

from .config import Settings
from .llm import LLMError, OpenAICompatibleClient
from .models import EnrichedItem, MenuResult, has_dish_facts
from .preferences import TasteProfile
from .pricing import NO_PRICE, format_price
from .recommend import Recommendation, Reason, budget_line, slot_template

__all__ = ["Answer", "ask_about_dish", "ask_about_menu", "text_client"]


class Answer(BaseModel):
    """A reply and where it came from, so the UI never implies a model was used."""

    model_config = ConfigDict(frozen=True)

    text: str
    source: Literal["model", "facts"] = "facts"
    notes: tuple[str, ...] = ()


def text_client(settings: Settings) -> OpenAICompatibleClient | None:
    """A client for prose, or None when no key is configured.

    Deliberately not `llm.default_client`, which hands back `OfflineStubClient` -
    it is built to raise on anything it cannot do, and the advisor has a better
    answer than an exception.
    """
    if not settings.llm_api_key:
        return None
    return OpenAICompatibleClient(settings)


# --------------------------------------------------------------------------- #
# Deterministic answers
# --------------------------------------------------------------------------- #
# Ordered: the first intent that matches wins, so the specific questions come
# before the general ones.
_INTENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("allergen", ("allerg", "peanut", "nut", "dairy", "milk", "gluten", "shellfish",
                  "egg", "soy", "sesame", "safe", "过敏", "安全")),
    ("vegetarian", ("vegetarian", "vegan", "meat", "pork", "素食", "素", "肉")),
    ("spice", ("spicy", "spice", "hot", "chilli", "chili", "辣", "麻")),
    ("price", ("price", "cost", "expensive", "cheap", "how much", "¥", "钱",
               "贵", "便宜")),
    ("portion", ("portion", "enough", "feed", "share", "how many", "size",
                 "how big", "people", "person", "份量", "够")),
    ("ingredient", ("ingredient", "contain", "made of", "what's in", "what is in",
                    "材料", "什么做的")),
    ("recommend", ("recommend", "suggest", "should i", "order", "best", "推荐", "点什么")),
    ("describe", ("what is", "what are", "what's", "describe", "tell me", "explain")),
)


def _intent(question: str) -> str:
    text = (question or "").strip().lower()
    if not text:
        return "describe"
    for name, keywords in _INTENTS:
        if any(keyword in text for keyword in keywords):
            return name
    return "describe"


def _unknown_allergen_line() -> str:
    return (
        "The menu does not list ingredients for this dish, so I cannot tell you "
        "whether it is safe. Ask the staff: 我对…过敏 (I am allergic to …)."
    )


def _answer_dish(
    intent: str,
    item: EnrichedItem,
    profile: TasteProfile,
    reasons: Sequence[Reason] = (),
) -> str:
    dish = item.dish
    facts = has_dish_facts(item)
    price = format_price(item.price_value, item.currency)

    if intent == "recommend":
        # "Should I order this?" is answered by the ranking, which is computed.
        if reasons:
            return "This one is on the list because: " + " ".join(
                reason.text for reason in reasons[:2]
            )
        intent = "describe"

    if intent == "allergen":
        if not facts:
            return _unknown_allergen_line()
        listed = [token for token in dish.allergens if token.strip()]
        avoided = {token.strip().lower() for token in profile.avoid_allergens}
        hits = [token for token in listed if token.lower() in avoided]
        if hits:
            return (
                f"This dish lists {', '.join(hits)}, which you asked to avoid. "
                "Do not order it without checking with the staff."
            )
        if listed:
            return (
                f"The menu lists these allergens for it: {', '.join(listed)}. "
                "That list may not be complete - confirm with the staff."
            )
        return (
            "No allergens are listed for this dish, which does not mean there are "
            "none. Ask the staff: 有没有…？(does it contain …?)"
        )

    if intent == "vegetarian":
        if not facts:
            return "The menu does not say what is in this dish, so it may contain meat."
        if dish.vegetarian is True:
            return "The menu marks this dish as vegetarian."
        if dish.vegetarian is False:
            return "This dish contains meat or fish."
        return "The menu does not say whether this dish is vegetarian."

    if intent == "spice":
        if not facts:
            return "The menu does not say how spicy this is. Assume chilli is present."
        level = int(dish.spice_level)
        tolerance = int(profile.spice_tolerance)
        if level == 0:
            return "It is marked 0/3 for spice - not hot."
        if level <= tolerance:
            return f"It is {level}/3 for spice, within what you said you can handle."
        if level - tolerance == 1:
            return f"It is {level}/3 for spice, a little above your {tolerance}/3. Ask for 微辣 (a little chilli)."
        return f"It is {level}/3 for spice, well above your {tolerance}/3. Ask for 不辣 (no chilli) - it is built on chilli."

    if intent == "price":
        if item.price_value is None:
            return NO_PRICE
        comment = dish.price_comment.strip()
        return f"{price}. {comment}".strip()

    if intent == "portion":
        dishes = len(slot_template(profile.party_size))
        base = (
            f"The menu does not print a portion size. For {profile.party_size} "
            f"people, plan on about {dishes} shared dishes in total."
        )
        if profile.party_size == 1 and facts:
            return (
                f"The menu does not print a portion size, so this is a guess: one "
                f"main like this is usually enough for one person."
            )
        return base

    if intent == "ingredient":
        if not facts:
            return "The menu does not say what is in this dish."
        tokens = [token for token in dish.ingredients if token.strip()]
        description = (dish.description or "").strip()
        if tokens:
            text = f"It contains: {', '.join(tokens)}."
            return f"{text} {description}".strip()
        if description:
            return description
        return "The menu describes this dish but does not list its ingredients."

    # describe
    parts: list[str] = []
    name = dish.en or item.cn
    if dish.en and dish.en != item.cn:
        parts.append(f"{item.cn} - {dish.en}.")
    else:
        parts.append(f"{item.cn}.")
    if dish.pinyin:
        parts.append(f"Say: {dish.pinyin}.")
    if dish.description:
        parts.append(dish.description)
    if facts and dish.category:
        parts.append(f"Category: {dish.category}. Spice {int(dish.spice_level)}/3.")
    if price:
        parts.append(f"{price}.")
    if not facts:
        parts.append("The menu does not describe this dish.")
    return " ".join(parts).strip()


def _answer_menu(
    intent: str, result: MenuResult, profile: TasteProfile, recs: Recommendation
) -> str:
    items = list(result.items)
    dishes = len(slot_template(profile.party_size))

    if intent == "recommend" or intent == "describe":
        if not recs.plan.slots:
            return "No plan yet - scan a menu first."
        names = ", ".join(
            f"{slot.cn} ({slot.price_display})" if slot.price_display else slot.cn
            for slot in recs.plan.slots
        )
        notes = " ".join(recs.plan.notes)
        return f"For {profile.party_size}: {names}. {budget_line(recs.plan)} {notes}".strip()

    if intent == "price":
        if recs.plan.total_known is None:
            return "This menu prints no prices, so there is nothing to compare."
        priced = [item for item in items if item.price_value is not None]
        cheap = min(priced, key=lambda item: item.price_value)
        dear = max(priced, key=lambda item: item.price_value)
        return (
            f"{budget_line(recs.plan)} The cheapest dish is {cheap.cn} at "
            f"{cheap.price_display}; the most expensive is {dear.cn} at "
            f"{dear.price_display}."
        )

    if intent == "portion":
        return (
            f"For {profile.party_size} people, about {dishes} shared dishes is the "
            f"usual amount. This plan has {len(recs.plan.slots)}."
        )

    if intent == "vegetarian":
        veg = [item.cn for item in items
               if has_dish_facts(item) and item.dish.vegetarian is True]
        unknown = sum(1 for item in items if not has_dish_facts(item))
        if veg:
            text = f"Vegetarian dishes on this menu: {', '.join(veg)}."
        else:
            text = "No dish on this menu is marked vegetarian."
        if unknown:
            text += f" {unknown} dishes were not described, so their status is unknown."
        return text

    if intent == "allergen":
        avoid = [token.strip().lower() for token in profile.avoid_allergens]
        named = [
            f"{item.cn} ({', '.join(item.dish.allergens)})"
            for item in items
            if has_dish_facts(item) and item.dish.allergens
        ]
        unknown = sum(
            1 for item in items if has_dish_facts(item) and not item.dish.allergens
        )
        head = (
            f"Dishes that list allergens: {'; '.join(named)}."
            if named
            else "No dish on this menu lists its allergens."
        )
        tail = ""
        if avoid:
            tail = f" You asked to avoid {', '.join(avoid)}."
        if unknown:
            tail += (
                f" {unknown} dishes list no allergens at all, which means unknown, "
                "not safe."
            )
        return f"{head}{tail}".strip()

    if intent == "spice":
        hot = sorted(
            (item for item in items if has_dish_facts(item)),
            key=lambda item: int(item.dish.spice_level),
        )
        if not hot:
            return "The menu does not describe how spicy anything is."
        mild = hot[0]
        return (
            f"The mildest described dish is {mild.cn} at "
            f"{int(mild.dish.spice_level)}/3, and the hottest is {hot[-1].cn} at "
            f"{int(hot[-1].dish.spice_level)}/3."
        )

    if intent == "ingredient":
        described = [item for item in items if has_dish_facts(item) and item.dish.description]
        if not described:
            return "The menu does not describe any dish in detail."
        return f"{described[0].cn}: {described[0].dish.description}"

    return ""


# --------------------------------------------------------------------------- #
# Model prompts
# --------------------------------------------------------------------------- #
_RULES = """\
You answer questions about a Chinese menu that has already been scanned.

Rules, all of them mandatory:
1. Answer ONLY from the FACTS block. If a field says "not stated", say the menu
   does not say. Never fill a gap from your general knowledge of the dish - you
   know what 麻婆豆腐 usually contains, and that knowledge must not appear.
2. Never invent an ingredient, allergen, cooking method, portion size or price.
3. Never do arithmetic. Every money figure in FACTS is final: quote it, do not
   sum, average or compare it yourself.
4. Never tell someone a dish is safe for an allergy. If the allergen list is
   "not stated", tell them to ask the staff and give them the phrase 有没有…？
5. Answer in at most 80 words, in plain language, in {language}. No markdown
   tables, no bullet lists longer than four items.
6. If the question is not about this menu, say in one line what you can answer.
"""


def _system_prompt(language: str = "English") -> str:
    return _RULES.format(language=language)


def _table_line(profile: TasteProfile) -> str:
    parts = [f"{profile.party_size} people"]
    if profile.budget_total:
        parts.append(f"budget ¥{profile.budget_total:g}")
    parts.append(f"spice tolerance {profile.spice_tolerance}/3")
    parts.append(f"vegetarian only: {'yes' if profile.vegetarian_only else 'no'}")
    parts.append(
        "avoid allergens: " + (", ".join(profile.avoid_allergens) or "none")
    )
    if profile.dietary_notes.strip():
        parts.append(f"notes: {profile.dietary_notes.strip()}")
    return "; ".join(parts)


def _field(value: Any, fallback: str = "not stated") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def dish_context(
    item: EnrichedItem,
    profile: TasteProfile,
    reasons: Sequence[Reason] = (),
    *,
    brief: bool = False,
) -> str:
    """Everything the model is allowed to know about one dish.

    Kept small on purpose: a few hundred characters, so the answer cannot be
    diluted by details nobody asked about. `brief` drops the pronunciation and
    the price analysis for the menu-wide context, where those repeat for every
    dish in the plan and would crowd out the dishes the question is about.
    """
    dish = item.dish
    facts = has_dish_facts(item)
    lines = [
        "DISH FACTS (from the scanned menu; \"not stated\" means the menu does not say)",
        f"cn: {item.cn}",
    ]
    if not brief:
        lines.append(f"pinyin: {_field(dish.pinyin)}")
    lines += [
        f"english: {_field(dish.en, item.en_printed or 'not stated')}",
        f"price: {format_price(item.price_value, item.currency) or 'not stated'}",
    ]
    if not brief:
        lines.append(f"section: {_field(item.section)}")
    lines += [
        f"what it is: {_field(dish.description)}",
        f"category: {_field(dish.category)}",
        f"spice: {str(int(dish.spice_level)) + '/3' if facts else 'not stated'}",
        "vegetarian: "
        + (
            "not stated"
            if not facts or dish.vegetarian is None
            else ("yes" if dish.vegetarian else "no")
        ),
        f"allergens: {', '.join(dish.allergens) if dish.allergens else 'not stated'}",
        f"ingredients: {', '.join(dish.ingredients) if dish.ingredients else 'not stated'}",
    ]
    if not brief:
        lines.append(f"price analysis (computed, not an opinion): {_field(dish.price_comment)}")
    if reasons:
        lines.append(
            "why the planner ranked it: "
            + " ".join(reason.text for reason in reasons[:3])
        )
    lines.append(f"YOUR TABLE: {_table_line(profile)}")
    return "\n".join(lines)


def menu_context(
    result: MenuResult,
    profile: TasteProfile,
    recs: Recommendation,
    *,
    sample_size: int = 5,
) -> str:
    """A compact summary plus full detail for the dishes that matter.

    Full descriptions for all 30 dishes would be ~2.5k tokens and would spread
    attention across dishes nobody asked about; the compact row tells the model
    what exists, and a short detail block covers the plan.
    """
    lines = [
        f"MENU: {len(result.items)} dishes" + (f" from {result.source}" if result.source else ""),
        f"YOUR TABLE: {_table_line(profile)}",
        "",
        "THE PLAN (already computed - quote it, do not recompute):",
        budget_line(recs.plan),
    ]
    for note in recs.plan.notes:
        lines.append(f"- {note}")

    lines += ["", "ALL DISHES (id | cn | en | price | category | spice | vegetarian)"]
    for item in result.items:
        dish = item.dish
        facts = has_dish_facts(item)
        price = format_price(item.price_value, item.currency) or "no price"
        category = _field(dish.category, "-") if facts else "-"
        spice = f"{int(dish.spice_level)}/3" if facts else "-"
        veg = (
            "unknown"
            if not facts or dish.vegetarian is None
            else ("yes" if dish.vegetarian else "no")
        )
        lines.append(
            f"{item.id} | {item.cn} | {_field(dish.en, '-')} | {price} | "
            f"{category} | {spice} | {veg}"
        )

    detailed = [slot.item_id for slot in recs.plan.slots][:sample_size]
    by_id = {item.id: item for item in result.items}
    for item_id in detailed:
        item = by_id.get(item_id)
        if item is None:
            continue
        lines += ["", dish_context(item, profile, brief=True)]
    return "\n".join(lines)


def _ask(
    question: str,
    context: str,
    fallback: str,
    client: Any | None,
    *,
    language: str = "English",
) -> Answer:
    """Try the model, fall back to the computed answer if it cannot help."""
    if client is None or not hasattr(client, "chat_text"):
        return Answer(text=fallback, source="facts",
                      notes=("No language model configured.",))
    messages = [
        {"role": "system", "content": _system_prompt(language)},
        {"role": "user", "content": f"{context}\n\nQUESTION: {question.strip()}"},
    ]
    try:
        text = str(client.chat_text(messages) or "").strip()
    except LLMError as exc:
        # The queue is busy or the key is out of credit. The computed answer is
        # worse prose and exactly as true, so show it rather than an error.
        return Answer(text=fallback, source="facts",
                      notes=(f"The language model was unavailable ({exc}).",))
    except Exception as exc:  # a demo should explain, not dump a traceback
        return Answer(text=fallback, source="facts",
                      notes=(f"{type(exc).__name__}: {exc}",))
    if not text:
        return Answer(text=fallback, source="facts",
                      notes=("The model returned nothing.",))
    return Answer(text=text, source="model")


def ask_about_dish(
    item: EnrichedItem,
    question: str,
    profile: TasteProfile,
    *,
    client: Any | None = None,
    reasons: Sequence[Reason] = (),
    language: str = "English",
) -> Answer:
    """Answer a question about one dish."""
    intent = _intent(question)
    fallback = _answer_dish(intent, item, profile, reasons)
    # Questions the facts answer outright never pay for a model call.
    if intent in ("price", "spice", "vegetarian", "ingredient", "portion", "recommend"):
        return Answer(text=fallback, source="facts")
    return _ask(question, dish_context(item, profile, reasons), fallback, client,
                language=language)


def ask_about_menu(
    question: str,
    result: MenuResult,
    profile: TasteProfile,
    recs: Recommendation,
    *,
    client: Any | None = None,
    language: str = "English",
) -> Answer:
    """Answer a question about the whole menu."""
    intent = _intent(question)
    fallback = _answer_menu(intent, result, profile, recs)
    if intent in ("price", "portion", "vegetarian", "spice", "recommend", "describe"):
        return Answer(text=fallback, source="facts")
    return _ask(question, menu_context(result, profile, recs), fallback, client,
                language=language)
