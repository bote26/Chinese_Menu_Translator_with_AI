"""What this person likes, and how the app knows it.

The contract, in one sentence: **hard restrictions are only ever the ones the
user typed**. Everything inferred from taps and history can only ever produce a
soft preference - a nudge in the ranking, never a filter. The distinction is not
pedantry. An inferred "peanut" dislike must not be allowed to hide dishes, and it
must never be presented to anyone as an allergy.

Two identities, because one cannot do both jobs:

* `name_key` is venue-free. A tourist who likes 牛肉面 likes it in Shanghai and
  in Chengdu, so taste is learned across every restaurant.
* `dish_key` is venue-scoped. "You paid ¥28 for this" is only true where the user
  actually paid it, so price memory never crosses restaurants. Telling someone
  they paid ¥28 at a place where the dish costs ¥45 is exactly the kind of false
  claim `pricing.py` was written to eliminate.

The event log is append-only and self-contained: every event carries a *snapshot*
of the dish as the user saw it. That makes `derive_profile` a pure function over a
file - it never needs a MenuResult, and it cannot be invalidated by a re-scan
whose positional item ids have shifted.

Nothing here touches Streamlit or the network.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Iterable, Literal

from pydantic import BaseModel, ConfigDict

from .models import EnrichedItem, MenuResult, has_dish_facts

SCHEMA_VERSION = 1

PROFILE_FILE = "profile.json"
EVENTS_FILE = "events.jsonl"

# Default location: <project>/data. Overridable through Settings.data_dir so the
# tests can point at a tmp directory and production can point anywhere.
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# --------------------------------------------------------------------------- #
# Tunables, all in one place
# --------------------------------------------------------------------------- #
# A like from two months ago says less about tonight than one from last week.
# 30 days is the half-life of a food preference in any sense a stranger could
# defend; MIN_DECAY keeps an old dislike a faint nudge rather than a full forget.
HALF_LIFE_DAYS = 30.0
MIN_DECAY = 0.05

# A dislike is worth more than a like: people report bad experiences more
# reliably than good ones, and one bad dish is enough reason to stop suggesting
# a similar one.
EVENT_WEIGHT: dict[str, float] = {"like": 1.0, "dislike": -1.5, "ordered": 0.6}

# Affinity at or above this is strong enough to name in the UI.
STRONG = 0.30
MAX_TOKENS = 12

# Spice inference needs roughly one fresh like's worth of evidence before it is
# allowed to override the slider default.
MIN_SPICE_SIGNAL = 1.0

MAX_EVENTS = 2000
COMPACT_KEEP_DAYS = 90.0

_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Dish identity
# --------------------------------------------------------------------------- #
# OCR confusions actually observed on this project's boards. This is data, not
# fuzz: a generic "one character differs" rule would merge 酸辣土豆丝 with
# 酸辣土豆片, which are two different dishes. Both members of a pair map to one
# representative, so the mapping is order-independent.
_CONFUSABLE: dict[str, str] = {
    "蜜": "蜜", "密": "蜜",          # 蜜汁 / 密汁 - seen on the roast-meat board
    "汁": "汁", "计": "汁",
    "雞": "鸡", "鶏": "鸡", "鷄": "鸡",
    "燒": "烧", "焼": "烧",
    "麵": "面", "麪": "面",
    "帶": "带",
    "蕃": "番",                      # 蕃茄 / 番茄
    "囬": "回",
}

# Bracketed qualifiers on the same dish: 干锅花菜（微辣）, 小笼包(6只), 卤蛋（２个）.
# NFKC has already turned the full-width digits into ASCII by the time this runs.
_TRAILING_QUALIFIER = re.compile(r"\s*[（(][^（()）]{1,8}[)）]\s*$")
_NON_WORD = re.compile(r"[\s\W_]+", re.UNICODE)

# Deliberately conservative: under-merging splits one dish into two keys (a lost
# nudge), over-merging attributes a dislike to the wrong dish (a wrong answer).
SIMILARITY_THRESHOLD = 0.85


def normalise_name(cn: str) -> str:
    """Canonical form of a printed dish name.

    Deterministic and index-free, so the same string always yields the same key
    no matter what else the user has eaten.
    """
    text = unicodedata.normalize("NFKC", cn or "")
    # Strip trailing qualifiers first - they need their brackets, which step 2
    # is about to remove.
    previous = None
    while previous != text:
        previous = text
        text = _TRAILING_QUALIFIER.sub("", text)
    text = _NON_WORD.sub("", text).lower()
    if not text:
        return ""
    return "".join(_CONFUSABLE.get(char, char) for char in text)


def name_key(cn: str) -> str:
    """Venue-free dish identity, after normalisation."""
    return normalise_name(cn)


def match_key(cn: str, known: Iterable[str] = ()) -> str:
    """Resolve `cn` against names already in the history.

    Returns an existing key when the printed name is a near-miss of one the app
    has seen (multi-character OCR damage survives the confusable table), else the
    normalised name itself. Called at *write* time so the key is decided once and
    stays stable; aggregation afterwards is a plain group-by.
    """
    base = normalise_name(cn)
    if not base:
        return ""
    candidates = [
        key
        for key in set(known)
        if key and len(key) >= 2 and abs(len(key) - len(base)) <= 1 and key[0] == base[0]
    ]
    if not candidates:
        return base
    matches = [base]
    for candidate in candidates:
        if candidate == base:
            return base
        if SequenceMatcher(None, base, candidate).ratio() >= SIMILARITY_THRESHOLD:
            matches.append(candidate)
    # The smallest representative keeps the choice independent of insertion order.
    return min(matches)


def dish_key(cn: str, venue: str = "") -> str:
    """Venue-scoped identity: used for price memory and "you ordered this here"."""
    return f"{venue}::{normalise_name(cn)}"


def venue_key(result: MenuResult, label: str = "") -> str:
    """A stable tag for "which restaurant is this board".

    A typed name wins, because the user knows better than a hash does. Otherwise
    fingerprint the dish set: photographing the same board twice gives the same
    tag, and a different restaurant with a different bill of fare gives another.

    Known limitation: two restaurants with an identical dish list collide. That
    only affects price history, and typing a name fixes it.
    """
    cleaned = normalise_name(label)
    if cleaned:
        return "u:" + cleaned
    names = sorted(normalise_name(item.cn) for item in result.items if item.cn)
    payload = "|".join(names) + f"#{result.image_size[0]}x{result.image_size[1]}"
    return "p:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class TasteProfile(BaseModel):
    """What the app believes about this eater.

    Frozen: `derive_profile` returns a new profile rather than mutating one, so a
    stale object can never be mistaken for current state.

    The fields are split into *stated* (typed by the human, authoritative) and
    *derived* (a recomputed display cache over the event log). Only the stated
    block may contain a hard restriction.
    """

    model_config = ConfigDict(frozen=True)

    version: int = SCHEMA_VERSION
    updated_at: str = ""

    # --- stated: the four things the sidebar asks -----------------------
    party_size: int = 2
    # Total for the table, not per head. None means "no limit", which disables
    # the budget component rather than scoring against a guessed default.
    budget_total: float | None = None
    spice_tolerance: int = 1
    # "inferred" means derive_profile may overwrite spice_tolerance; a value the
    # user typed is never silently replaced.
    spice_tolerance_source: Literal["stated", "inferred"] = "stated"
    vegetarian_only: bool = False
    avoid_allergens: tuple[str, ...] = ()
    adventurousness: float = 0.5
    # Free text ("halal", "no pork"). Passed to the language model as context and
    # shown in the plan header, and *never* parsed into a filter: guessing a
    # restriction from prose is how somebody gets served pork.
    dietary_notes: str = ""

    # --- derived: recomputed from events on every load -------------------
    liked_ingredients: tuple[str, ...] = ()
    disliked_ingredients: tuple[str, ...] = ()
    liked_categories: tuple[str, ...] = ()
    disliked_categories: tuple[str, ...] = ()
    # Token -> [-1, 1]. Widened-in tuples keep the frozen model hashable-friendly
    # and the JSON stable.
    ingredient_affinity: tuple[tuple[str, float], ...] = ()
    category_affinity: tuple[tuple[str, float], ...] = ()
    events_used: int = 0
    # What inference *would* say, always reported separately so the UI can offer
    # it ("use spice 3?") instead of applying it behind the user's back.
    inferred_spice: int | None = None

    def _lookup(self, pairs: tuple[tuple[str, float], ...], token: str) -> float:
        for key, value in pairs:
            if key == token:
                return value
        return 0.0

    def ingredient_weight(self, token: str) -> float:
        return self._lookup(self.ingredient_affinity, (token or "").strip().lower())

    def category_weight(self, category: str) -> float:
        return self._lookup(self.category_affinity, (category or "").strip().lower())


DEFAULT_PROFILE = TasteProfile()


class PreferenceEvent(BaseModel):
    """One thing the user did, with the dish as it looked at that moment.

    The snapshot is the point. `EnrichedItem.id` is positional - it is
    `len(items)+1` on the vision path and `index` on the OCR path - so re-scanning
    the same board after one dropped line re-points every id after it. Storing a
    reference would silently attach "I liked this" to a different dish.
    """

    model_config = ConfigDict(frozen=True)

    ts: float
    kind: Literal["like", "dislike", "ordered", "rated"]
    dish_key: str
    name_key: str
    venue: str = ""
    cn: str = ""
    menu_source: str = ""

    price_value: float | None = None
    price_raw: str | None = None
    category: str = ""
    spice_level: int = 0
    vegetarian: bool | None = None
    ingredients: tuple[str, ...] = ()
    allergens: tuple[str, ...] = ()
    en: str = ""
    section: str | None = None
    # False for a dish the model never described: its ingredient list is empty
    # because nobody looked, not because it has none.
    facts_known: bool = False
    rating: int | None = None


def event_from_item(
    item: EnrichedItem,
    kind: Literal["like", "dislike", "ordered", "rated"],
    *,
    venue: str = "",
    menu_source: str = "",
    now: float | None = None,
    rating: int | None = None,
    known: Iterable[str] = (),
) -> PreferenceEvent:
    """Snapshot one dish as an event. Pure; the caller decides where it is stored."""
    dish = item.dish
    return PreferenceEvent(
        ts=time.time() if now is None else now,
        kind=kind,
        dish_key=dish_key(item.cn, venue),
        name_key=match_key(item.cn, known),
        venue=venue,
        cn=item.cn,
        menu_source=menu_source,
        price_value=item.price_value,
        price_raw=item.price_raw,
        category=(dish.category or "").strip().lower(),
        spice_level=int(dish.spice_level or 0),
        vegetarian=dish.vegetarian,
        ingredients=tuple(t.strip().lower() for t in dish.ingredients if t.strip()),
        allergens=tuple(t.strip().lower() for t in dish.allergens if t.strip()),
        en=dish.en or "",
        section=item.section,
        facts_known=has_dish_facts(item),
        rating=rating,
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _atomic_write(path: Path, text: str) -> None:
    """Write via a sibling temp file, then rename.

    `os.replace` is atomic on NTFS for a same-volume rename, so a crash leaves
    either the old file or the new one - never half of one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _quarantine(path: Path) -> None:
    """Move an unreadable file aside so it is not silently overwritten."""
    stamp = int(time.time())
    try:
        os.replace(path, path.with_name(f"{path.name}.corrupt-{stamp}"))
    except OSError:
        pass


def load_profile(directory: Path | str = DEFAULT_DATA_DIR) -> tuple[TasteProfile, list[str]]:
    """Read the profile. Never raises: a broken file is a warning, not a crash."""
    path = Path(directory) / PROFILE_FILE
    if not path.is_file():
        # First run. Not an error, and not worth a message.
        return DEFAULT_PROFILE, []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return DEFAULT_PROFILE, [f"Could not read {path}: {exc}"]

    try:
        payload = json.loads(raw)
        profile = TasteProfile.model_validate(payload)
    except Exception as exc:  # JSONDecodeError, ValidationError, and friends
        _quarantine(path)
        return DEFAULT_PROFILE, [
            f"{path.name} could not be read ({type(exc).__name__}); starting from "
            f"defaults. The old file was moved aside."
        ]

    # A profile from a newer build may contain fields this one cannot honour.
    # Keep what parses rather than discarding a working profile.
    if profile.version > SCHEMA_VERSION:
        return profile, [
            f"{path.name} was written by a newer version (schema "
            f"{profile.version}); unknown settings were ignored."
        ]
    return profile, []


def save_profile(
    profile: TasteProfile, directory: Path | str = DEFAULT_DATA_DIR
) -> str | None:
    """Persist the profile. Returns a warning string, or None on success."""
    path = Path(directory) / PROFILE_FILE
    try:
        _atomic_write(path, profile.model_dump_json(indent=2))
    except OSError as exc:
        # Windows raises PermissionError when antivirus or a second worker holds
        # the file. Losing one save is not worth losing the session over.
        return f"Could not save your preferences: {exc}"
    return None


def _ends_with_newline(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                return True
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) == b"\n"
    except FileNotFoundError:
        return True
    except OSError:
        return True


def append_event(
    event: PreferenceEvent, directory: Path | str = DEFAULT_DATA_DIR
) -> str | None:
    """Append one event to the log. Returns a warning string, or None."""
    directory = Path(directory)
    path = directory / EVENTS_FILE
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            # A crash mid-write leaves a tail without a newline. Repairing it here
            # means the next record cannot be glued onto the broken one.
            prefix = "" if _ends_with_newline(path) else "\n"
            with open(path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(prefix + event.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
    except OSError as exc:
        return f"Could not record that tap: {exc}"
    return None


def load_events(
    directory: Path | str = DEFAULT_DATA_DIR,
) -> tuple[list[PreferenceEvent], list[str]]:
    """Read the event log, skipping damaged lines one at a time.

    Parsing each line rather than trusting "everything before the last newline"
    is deliberate: a record that was fully written but not yet fsynced is still
    valid and is kept, while a genuinely truncated one is dropped.
    """
    path = Path(directory) / EVENTS_FILE
    if not path.is_file():
        return [], []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [], [f"Could not read {path}: {exc}"]

    events: list[PreferenceEvent] = []
    damaged = 0
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(PreferenceEvent.model_validate_json(line))
        except Exception:
            damaged += 1
    warnings = (
        [f"Skipped {damaged} damaged line(s) in {path.name}."] if damaged else []
    )
    return events, warnings


def clear_history(directory: Path | str = DEFAULT_DATA_DIR) -> str | None:
    """Forget everything: remove the event log, keep the stated profile.

    A feature that remembers you owes you a way to make it stop.
    """
    path = Path(directory) / EVENTS_FILE
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        return f"Could not clear the history: {exc}"
    return None


def compact_events(
    directory: Path | str = DEFAULT_DATA_DIR, *, now: float | None = None
) -> str | None:
    """Collapse a long log without changing what the decay would have said.

    Keeps the latest event per (kind, dish_key) plus everything inside the decay
    horizon, so the file stays bounded and nothing unrecoverable is dropped.
    """
    directory = Path(directory)
    events, _warnings = load_events(directory)
    if len(events) <= MAX_EVENTS:
        return None

    reference = time.time() if now is None else now
    horizon = reference - COMPACT_KEEP_DAYS * 86400.0
    latest: dict[tuple[str, str], PreferenceEvent] = {}
    for event in events:
        latest[(event.kind, event.dish_key)] = event  # later wins: file is in order
    keep = {id(event) for event in latest.values()} | {
        id(event) for event in events if event.ts >= horizon
    }
    kept = [event for event in events if id(event) in keep]
    kept.sort(key=lambda event: event.ts)
    try:
        _atomic_write(
            directory / EVENTS_FILE,
            "".join(event.model_dump_json() + "\n" for event in kept),
        )
    except OSError as exc:
        return f"Could not compact the history: {exc}"
    return None


# --------------------------------------------------------------------------- #
# Learning
# --------------------------------------------------------------------------- #
def _decay(age_days: float) -> float:
    return max(MIN_DECAY, 0.5 ** (max(0.0, age_days) / HALF_LIFE_DAYS))


def _event_weight(event: PreferenceEvent) -> float:
    if event.kind == "rated":
        if event.rating is None:
            return 0.0
        # 1-5 stars -> -1 .. +1, so a 3-star rating says nothing.
        return max(-1.0, min(1.0, (event.rating - 3) / 2.0))
    return EVENT_WEIGHT.get(event.kind, 0.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _affinities(
    events: list[PreferenceEvent],
    tokens_of: Callable[[PreferenceEvent], Iterable[str]],
    *,
    reference: float,
    use_idf: bool,
) -> tuple[tuple[tuple[str, float], ...], int]:
    """Weighted token scores, normalised to [-1, 1].

    The IDF term is what stops "soy sauce" - present in most dishes - from
    outranking the one ingredient that actually distinguishes what this person
    orders. Scores are normalised by `max(1, peak)` rather than the peak alone,
    so a single tap is a hint (around 0.4) instead of instant certainty at 1.0.
    """
    contributing = [event for event in events if event.facts_known]
    if not contributing:
        return (), 0

    total = len(contributing)
    document_frequency: dict[str, int] = {}
    per_event: list[tuple[float, set[str]]] = []
    for event in contributing:
        tokens = {token for token in tokens_of(event) if token}
        if not tokens:
            continue
        signal = _event_weight(event) * _decay((reference - event.ts) / 86400.0)
        per_event.append((signal, tokens))
        for token in tokens:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    if not per_event:
        return (), 0

    totals: dict[str, float] = {}
    for signal, tokens in per_event:
        for token in tokens:
            weight = signal
            if use_idf:
                # Smoothed: a token in every event decays towards zero without
                # ever reaching it, so genuine repetition still accumulates.
                weight *= math.log((total + 2) / (document_frequency[token] + 1))
            totals[token] = totals.get(token, 0.0) + weight

    peak = max((abs(value) for value in totals.values()), default=0.0)
    scale = max(1.0, peak)
    scaled = {
        token: round(_clamp(value / scale, -1.0, 1.0), 4)
        for token, value in totals.items()
    }
    ordered = tuple(sorted(scaled.items(), key=lambda pair: (-abs(pair[1]), pair[0])))
    return ordered, len(per_event)


def _named(affinity: tuple[tuple[str, float], ...], sign: int) -> tuple[str, ...]:
    tokens = [
        token
        for token, value in affinity
        if (value >= STRONG if sign > 0 else value <= -STRONG)
    ]
    return tuple(tokens[:MAX_TOKENS])


def derive_profile(
    events: list[PreferenceEvent],
    base: TasteProfile | None = None,
    *,
    now: float | None = None,
) -> TasteProfile:
    """Merge the event log into a profile.

    Only the *derived* block is written, plus `spice_tolerance` when the user has
    explicitly asked for it to be learned. `avoid_allergens`, `vegetarian_only`
    and `dietary_notes` are copied straight through: a dislike is not an allergy,
    and this function has no business deciding that it is.
    """
    profile = base or DEFAULT_PROFILE
    reference = time.time() if now is None else now
    if not events:
        return profile.model_copy(
            update={
                "ingredient_affinity": (),
                "category_affinity": (),
                "liked_ingredients": (),
                "disliked_ingredients": (),
                "liked_categories": (),
                "disliked_categories": (),
                "events_used": 0,
                "inferred_spice": None,
                "updated_at": _iso(reference),
            }
        )

    # Time decays from `now`, which the tests control.
    ingredient_scores, _ = _affinities(
        events, lambda event: event.ingredients, reference=reference, use_idf=True
    )
    category_scores, _ = _affinities(
        events,
        lambda event: (event.category,) if event.category else (),
        reference=reference,
        use_idf=False,
    )

    inferred = _infer_spice(events, reference)
    update: dict[str, object] = {
        "ingredient_affinity": ingredient_scores,
        "category_affinity": category_scores,
        "liked_ingredients": _named(ingredient_scores, +1),
        "disliked_ingredients": _named(ingredient_scores, -1),
        "liked_categories": _named(category_scores, +1),
        "disliked_categories": _named(category_scores, -1),
        "events_used": len(events),
        "inferred_spice": inferred,
        "updated_at": _iso(reference),
    }
    if profile.spice_tolerance_source == "inferred" and inferred is not None:
        update["spice_tolerance"] = inferred
    return profile.model_copy(update=update)


def _infer_spice(events: list[PreferenceEvent], reference: float) -> int | None:
    """Mean spice of the dishes this person reacted well to.

    Needs about one fresh like's worth of evidence; below that the slider default
    is better than a guess.
    """
    weight_total = 0.0
    weighted = 0.0
    for event in events:
        if not event.facts_known:
            continue
        signal = _event_weight(event) * _decay((reference - event.ts) / 86400.0)
        if signal <= 0:
            continue
        weight_total += signal
        weighted += signal * float(event.spice_level)
    if weight_total < MIN_SPICE_SIGNAL:
        return None
    return int(round(_clamp(weighted / weight_total, 0.0, 3.0)))


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds")
