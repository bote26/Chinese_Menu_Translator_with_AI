"""Price comparison, computed instead of described.

The model used to write `price_comment` itself. On a 16-item roast-meat board it
called three different dishes "the cheapest rice dish on the menu", and called a
22 CNY bowl "below average" when it sat above the median of its own group. Those
claims are arithmetic, not language: doing them here makes every sentence
checkable, keeps the numbers consistent with `price_value`, and removes ~15
output tokens per dish from the bill.

Nothing in this module touches the network, so it is trivially testable.
"""

from __future__ import annotations

import statistics

from .models import EnrichedItem

NO_PRICE = "No price printed for this item on the menu."


def format_price(value: float | None, currency: str = "CNY") -> str:
    """Human-facing price label: `¥18.5`, `¥7.65`, `RP 12000`."""
    if value is None:
        return ""
    symbol = "¥" if currency.upper() == "CNY" else f"{currency.upper()} "
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{symbol}{text}"


def _money(value: float) -> str:
    return f"{value:g} CNY"


def _plural(word: str, count: int) -> str:
    if count == 1:
        return word
    if word.endswith(("s", "x", "ch", "sh")):
        return word + "es"
    return word + "s"


def _group_label(category: str, count: int, same_category: bool) -> str:
    if same_category and category:
        return f"the {count} {_plural(category, count)} here"
    return f"the {count} priced dishes here"


def unit_of(price_raw: str | None) -> str:
    """How the printed price is measured.

    Cantonese roast shops sell the same bird by portion ("88例") or by half bird
    ("50元半只"), and those numbers are not comparable. Detect it from the raw
    text rather than translating it: "half bird" is a safe literal rendering of
    the printed unit.

    Public because the recommender groups peers by it too, and two modules
    disagreeing about what counts as a comparable price is exactly the bug this
    whole file exists to prevent.
    """
    raw = price_raw or ""
    if "半只" in raw:
        return "half bird"
    if "只" in raw:
        return "whole bird"
    return "portion"


# Kept as an alias so the existing call sites and tests read unchanged.
_unit = unit_of


def _comment(item: EnrichedItem, priced: list[EnrichedItem]) -> str:
    value = item.price_value
    if value is None:
        return NO_PRICE

    unit = _unit(item.price_raw)
    category = (item.dish.category or "").strip().lower()
    same_unit = [other for other in priced if _unit(other.price_raw) == unit]
    peers = [
        other
        for other in same_unit
        if category and (other.dish.category or "").strip().lower() == category
    ]
    # Comparing a 16 CNY rice bowl with a 68 CNY plate of ribs says nothing
    # useful, so prefer the same category - but only if there are enough of them
    # for a median to mean anything.
    same_category = len(peers) >= 3
    if not same_category:
        peers = same_unit

    values = [peer.price_value for peer in peers]
    if len(values) < 3:
        if len(values) <= 1:
            if unit != "portion":
                return f"{_money(value)} - the only item priced per {unit} here."
            return f"{_money(value)} - the only priced item on this menu."
        return f"{_money(value)} - too few prices on this menu to compare."

    median = statistics.median(values)
    group = _group_label(category, len(values), same_category)
    low, high = min(values), max(values)

    if value == low:
        tied = sum(1 for other in values if other == low)
        head = "the cheapest" if tied == 1 else f"tied for cheapest ({tied} at this price)"
        return f"{_money(value)} - {head} of {group} (median {_money(median)})."
    if value == high:
        tied = sum(1 for other in values if other == high)
        head = (
            "the most expensive"
            if tied == 1
            else f"tied for most expensive ({tied} at this price)"
        )
        return f"{_money(value)} - {head} of {group} (median {_money(median)})."

    difference = value - median
    # Within a yuan of the median is "typical": spelling out "0.5 CNY above the
    # median" is technically true and useless to a customer.
    if abs(difference) <= 1.0:
        return f"{_money(value)} - about the typical price of {group} (median {_money(median)})."
    side = "below" if difference < 0 else "above"
    return (
        f"{_money(value)} - {abs(difference):g} CNY {side} the median of {group} "
        f"({_money(median)})."
    )


def annotate_prices(items: list[EnrichedItem]) -> None:
    """Fill `dish.price_comment` for every item, priced or not.

    Runs after enrichment, so it can group by the model's `category`. It also
    overwrites whatever the model said about price, which is the point.
    """
    priced = [item for item in items if item.price_value is not None]
    for item in items:
        item.dish.price_comment = _comment(item, priced)
