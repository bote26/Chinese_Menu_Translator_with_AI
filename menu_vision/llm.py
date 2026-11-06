"""LLM layer: a compact Chinese item vector in, structured dish knowledge out.

The model is deliberately not asked to read the image. OCR already produced the
text; the model gets a small JSON array of `{id, cn, price}` and returns one
object per id. That keeps tokens low, makes the output validatable, and means a
better OCR engine immediately improves results without touching the prompt.

Any OpenAI-compatible endpoint works, so this uses plain `requests` instead of
a vendor SDK (no `openai` package is installed on every interpreter here).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Protocol

import requests

from .config import Settings
from .models import DishInfo

SYSTEM_PROMPT = """\
You are a Chinese-menu analyst. You receive JSON produced by OCR from a photo of \
a Chinese restaurant menu (a printed wall menu or a delivery-app product screen).

The OCR text is noisy. It can contain app UI (status bar, "销量96", "选规格", \
"商品单位"), store names, watermarks, section titles, and unit words such as "份". \
Some entries are real dishes; some are not.

For every entry, do all of the following:

1. TRIAGE - set `is_dish`. Use false for anything that is not a purchasable food \
or drink item (buttons, prices of other items, "详情", "评价", store name, phone \
status bar, watermark). A pure category line like "烧烤" or "主食" is NOT a dish.
A missing price is NOT a reason to reject a dish: on a wall board one whole \
column often prints no prices (or the column is cut off by the frame), and those \
are still dishes. Judge the words alone - "卤蛋" (braised egg) is a dish whether \
or not a number sits beside it, and "烧鹅头带脖子" (goose head with neck) is a \
real Cantonese snack.
2. CLEAN - repair clear OCR slips using Chinese culinary vocabulary (a dropped or \
confused character in a known dish name). Never invent characters; if the name is \
unreadable, keep it as-is and lower `confidence`.
3. TRANSLATE - `pinyin` with tone marks and word spacing, and a natural `en` name \
that a foreigner would order from ("Hot and Sour Shredded Potato", not "Sour \
Pepper Potato Silk"). If a record carries `en_printed`, that is the wording the \
restaurant printed on the board: keep its meaning and use it as your base, only \
smoothing obvious OCR damage or awkward phrasing - do not rename the dish.
4. EXPLAIN - `description`, 1-2 sentences, for someone who has never eaten Chinese \
food: what the dish is, its main ingredients, how it is cooked, and how it tastes.
5. CONTEXT - `category` (e.g. "cold dish", "rice bowl", "skewer", "drink", \
"vegetable", "noodle"); `spice_level` 0-3; `vegetarian` true/false/null when the \
ingredients make it clear or not; `allergens` from wheat, soy, egg, dairy, peanut, \
tree nut, shellfish, fish, sesame, gluten.

Do not comment on price at all: comparison against the rest of the menu is \
computed separately from the parsed numbers, so any sentence you write about \
price is discarded. Never invent, restate or convert a price.

Hard rules:
- If `price` is null, the menu did not show one. Say nothing about price.
- Prefer honesty over guessing: lower `confidence` (0-1) rather than fabricating.
- Answer in {language}.
- Return STRICT JSON only: a JSON array, one object per input `id`, same ids in the \
same order. No markdown fences, no commentary.

Each object must have exactly these keys:
id (int), is_dish (bool), cn (string), pinyin (string), en (string), \
description (string), ingredients (array of strings), category (string), \
spice_level (int 0-3), vegetarian (bool or null), allergens (array of strings), \
confidence (number 0-1), reason (string, only when is_dish is false).
"""


class LLMError(RuntimeError):
    """Raised when the model cannot produce usable JSON."""


class _JsonModeUnavailable(Exception):
    """Internal signal: the reply was fine, json mode was the problem.

    Deliberately *not* an `LLMError`, so it cannot be swallowed by the generic
    "unparseable reply" handler and spend a retry: the request has to go again
    without `response_format`, which should not cost retry budget.
    """


def _parse_text(content: str) -> str:
    """The prose path: the reply is the answer, so keep it as written."""
    return (content or "").strip()


# `thinking` is not portable: providers spell the same idea differently and
# cheap endpoints refuse extras outright. Two groups matter here.
_THINKING_OFF = {"disabled", "off", "none", "false", "0", "no"}
_THINKING_AUTO = {"", "auto", "default", "provider", "on", "true", "1", "yes"}
# glm-5.x answers HTTP 400 / code 1210 for every thinking value except a level:
# "该模型始终思考，不支持关闭思考；请使用 low、high 或 max。" Note `type` only
# accepts enabled/disabled - the effort goes in `level`, not in `type`.
_ALWAYS_THINKS_HINTS = ("始终思考", "always think", "cannot be disabled")

# `_DIRECT` is a sentinel rather than None: None has to keep meaning "ask the
# machine", so the two states stay distinguishable in one attribute.
_DIRECT = "direct"
_PROXY_AUTO = {"", "auto", "default", "system", "env", "environment"}
_PROXY_OFF = {"off", "no", "none", "false", "0", "direct", "bypass"}


def resolve_proxy(raw: str | None) -> str:
    """Interpret `MENU_LLM_PROXY` as a proxy URL, `_DIRECT`, or "auto".

    "auto" is returned as `_PROXY_AUTO`'s own spelling so the caller can keep
    `trust_env` on and let `requests` do the discovery (env vars, then the
    Windows registry, which is where Clash and friends live).
    """
    value = (raw or "").strip()
    lowered = value.lower()
    if lowered in _PROXY_AUTO:
        return "auto"
    if lowered in _PROXY_OFF:
        return _DIRECT
    if "://" not in value:
        # A bare `127.0.0.1:7890` is what people actually type; be forgiving
        # instead of handing requests an unparseable scheme.
        value = "http://" + value
    return value



class LLMClient(Protocol):
    model: str

    def chat_json(self, system: str, user: str) -> Any: ...


class VisionClient(Protocol):
    """Anything that can take a multimodal message list and return JSON."""

    model: str

    def chat_json_messages(self, messages: list[dict[str, Any]]) -> Any: ...


def build_system_prompt(settings: Settings) -> str:
    return SYSTEM_PROMPT.format(language=settings.target_language)


def build_user_prompt(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return (
        f"Menu entries from OCR ({len(records)} rows). "
        "`price` is the raw text as printed; `price_value` is the parsed number in CNY. "
        "`en_printed`, when present, is the English printed on the menu board.\n\n"
        f"{payload}"
    )


def _extract_json(text: str) -> Any:
    """Pull the JSON array/object out of a model reply.

    Models still wrap output in fences despite instructions, and occasionally
    add a sentence before the array, so this tolerates both.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    # Salvage a reply that hit the output limit mid-array: the last object is
    # incomplete, but every object before it is valid. Half a menu beats an
    # error, and the caller can see how many rows came back.
    start, end = cleaned.find("["), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1] + "]")
        except json.JSONDecodeError:
            pass
    raise LLMError(f"Model did not return JSON. First 300 chars: {cleaned[:300]!r}")


class OpenAICompatibleClient:
    """Minimal chat-completions client for any OpenAI-compatible API."""

    def __init__(
        self,
        settings: Settings,
        *,
        model: str | None = None,
        thinking: str | None = None,
        json_mode: bool | None = None,
        timeout: float | None = None,
    ) -> None:
        if not settings.llm_api_key:
            raise LLMError("MENU_LLM_API_KEY is not set")
        # Overrides exist so one class can serve both models: the vision model
        # and the text model sit behind the same endpoint but differ in what
        # they accept, so capabilities are learned per instance.
        self.model = model or settings.llm_model
        self._thinking_raw = settings.llm_thinking if thinking is None else thinking
        self._settings = settings
        self._url = settings.llm_base_url.rstrip("/") + "/chat/completions"
        self._timeout = timeout or settings.llm_timeout
        self._headers = {
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        # What this endpoint accepts, learned from 400s. 0 = not yet tested,
        # 1 = model always reasons so only a level is allowed, 2 = send nothing.
        self._thinking_stage = 0
        self._json_mode = settings.llm_json_mode if json_mode is None else json_mode
        # Token accounting for the last successful call, filled from the
        # provider's `usage` block. Lets the caller price a scan for real
        # instead of estimating.
        self.last_usage: dict[str, int] = {}
        # Transport. A Session (not a bare requests.post) keeps the TLS
        # connection alive across batches, which is worth a handshake per call
        # on a 3-batch OCR run.
        self._session = requests.Session()
        self._proxy = resolve_proxy(settings.llm_proxy)
        if self._proxy == _DIRECT:
            self._session.trust_env = False
        # Set to the ProxyError text once an unusable proxy has been bypassed,
        # so the caller can say so instead of silently changing behaviour.
        self.proxy_bypassed: str | None = None

    def _post(self, body: dict[str, Any]) -> requests.Response:
        """POST with the machine's proxy, falling back to a direct connection.

        A laptop that once ran a VPN client keeps `HTTP(S)_PROXY` pointing at a
        port that is now closed, and every request then dies with
        `ProxyError: Cannot connect to proxy` - which reads like the API being
        down and sends people hunting for the wrong bug. The first such error
        teaches this client that the proxy is unusable: it is dropped for the
        rest of the process and the same body is retried directly, once.
        """
        kwargs: dict[str, Any] = {
            "json": body,
            "headers": self._headers,
            "timeout": self._timeout,
        }
        explicit = self._proxy not in (_DIRECT, "auto")
        if explicit:
            kwargs["proxies"] = {"http": self._proxy, "https": self._proxy}
        try:
            return self._session.post(self._url, **kwargs)
        except requests.exceptions.ProxyError as exc:
            if not explicit and not self._session.trust_env:
                # Already direct: the failure is not the proxy's fault.
                raise
            self.proxy_bypassed = str(exc)
            self._proxy = _DIRECT
            self._session.trust_env = False
            kwargs.pop("proxies", None)
            return self._session.post(self._url, **kwargs)

    def _thinking_param(self) -> dict[str, Any] | None:
        """The `thinking` block for this endpoint, or None to omit it."""
        if self._thinking_stage >= 2:
            return None
        if self._thinking_stage == 1:
            # Model always reasons: ask for its lightest level rather than
            # falling back to the provider default, which may be "high".
            return {"type": "enabled", "level": "low"}
        raw = (self._thinking_raw or "").strip().lower()
        if raw in _THINKING_AUTO:
            return None
        if raw in _THINKING_OFF:
            return {"type": "disabled"}
        return {"type": "enabled", "level": raw}

    def _extra_params(self, *, allow_json: bool = True) -> dict[str, Any]:
        """Provider extras that are nice to have, not required.

        `allow_json=False` for prose calls: a provider handed
        `response_format: json_object` and a question either rejects the request
        or wraps the answer in JSON the reader would see verbatim.
        """
        params: dict[str, Any] = {}
        thinking = self._thinking_param()
        if thinking is not None:
            params["thinking"] = thinking
        if allow_json and self._json_mode:
            params["response_format"] = {"type": "json_object"}
        return params

    def _degrade(self, response: requests.Response) -> bool:
        """Relax the parameter a 400 objected to. False when nothing is left.

        Bounded: thinking degrades at most twice (stage 0 -> 1 -> 2) and
        response_format at most once, so a bad request cannot loop.
        """
        text = response.text or ""
        lowered = text.lower()
        thinking_to_blame = "思考" in text or "thinking" in lowered
        json_to_blame = "response_format" in lowered or "json_object" in lowered

        if self._thinking_stage < 2 and (thinking_to_blame or not json_to_blame):
            always_reasons = any(
                hint in text or hint in lowered for hint in _ALWAYS_THINKS_HINTS
            )
            self._thinking_stage = 1 if (self._thinking_stage == 0 and always_reasons) else 2
            return True
        if self._json_mode:
            self._json_mode = False
            return True
        return False

    def chat_json(self, system: str, user: str) -> Any:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.chat_json_messages(messages)

    def chat_json_messages(self, messages: list[dict[str, Any]]) -> Any:
        """Send a ready-made message list and parse the JSON reply.

        `content` may be a plain string or a multimodal list of parts, which is
        how a photo is attached. Everything else - retry, parameter degrade,
        JSON recovery - is shared with the text path on purpose: a vision call
        fails in exactly the same ways.
        """
        return self._chat(messages, parse=self._parse_items, allow_json=True, nudge=True)

    def chat_text(self, messages: list[dict[str, Any]]) -> str:
        """Send a message list and return the reply as prose.

        Same transport, retry and backoff as the JSON path - only the parsing
        differs. Used by the advisor, where a JSON envelope would be noise.
        """
        return self._chat(messages, parse=_parse_text, allow_json=False, nudge=False)

    def _parse_items(self, content: str) -> Any:
        """Turn a JSON reply into the item list, healing json mode if it interfered."""
        parsed = _extract_json(content)
        # json_object mode forbids a bare array, so the model wraps it. Key names
        # vary by model (`entries` is what glm-5.x picks), hence the generic
        # single-list rule.
        if isinstance(parsed, dict):
            for key in ("items", "entries", "data", "result", "dishes", "menu"):
                if isinstance(parsed.get(key), list):
                    return parsed[key]
            lists = [value for value in parsed.values() if isinstance(value, list)]
            if len(lists) == 1:
                return lists[0]
            if self._json_mode:
                # json_object mode forbids the array outright, and glm-5.3-flash
                # then emits a single record instead of the whole menu. Turn the
                # mode off and ask again; the prompt already demands a bare
                # array, so nothing else has to change.
                raise _JsonModeUnavailable()
            raise LLMError("Model returned an object without an items array.")
        return parsed

    def _chat(
        self,
        messages: list[dict[str, Any]],
        *,
        parse: Callable[[str], Any],
        allow_json: bool,
        nudge: bool,
    ) -> Any:
        """One loop for both reply shapes: transport, retry, degrade, recover."""
        failures = 0
        last_error: Exception | None = None

        while True:
            body: dict[str, Any] = {
                "model": self.model,
                "temperature": self._settings.llm_temperature,
                "messages": messages,
            }
            body.update(self._extra_params(allow_json=allow_json))
            try:
                response = self._post(body)
            except requests.RequestException as exc:
                last_error = exc
            else:
                if response.status_code == 400 and self._degrade(response):
                    # Reject an unsupported extra and retry without spending a
                    # retry: the prompt still demands strict JSON.
                    continue
                if response.status_code in (429, 500, 502, 503, 504):
                    # Free tiers are congested by design; back off and try again.
                    last_error = LLMError(
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )
                    if failures < self._settings.llm_max_retries:
                        time.sleep(self._settings.llm_retry_backoff * (2**failures))
                elif response.status_code >= 400:
                    last_error = LLMError(
                        f"HTTP {response.status_code}: {response.text[:300]}"
                    )
                else:
                    try:
                        payload = response.json()
                        content = payload["choices"][0]["message"]["content"]
                        usage = payload.get("usage") or {}
                        self.last_usage = {
                            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                            "completion_tokens": int(usage.get("completion_tokens") or 0),
                            "total_tokens": int(usage.get("total_tokens") or 0),
                            # Only some models report this; useful when a scan
                            # feels slow and you want to know why.
                            "reasoning_tokens": int(
                                (usage.get("completion_tokens_details") or {}).get(
                                    "reasoning_tokens"
                                )
                                or 0
                            ),
                        }
                        try:
                            return parse(content)
                        except _JsonModeUnavailable:
                            # Not a failure: the request was fine, the mode was
                            # wrong. Retry without spending the retry budget.
                            self._json_mode = False
                            continue
                    except (KeyError, ValueError, LLMError) as exc:
                        last_error = exc
                        if nudge:
                            messages = messages[:2] + [
                                {
                                    "role": "user",
                                    "content": (
                                        "Return ONLY the JSON array, no prose, "
                                        "no code fences."
                                    ),
                                }
                            ]
            failures += 1
            if failures > self._settings.llm_max_retries:
                raise LLMError(
                    f"LLM request failed after retries: {last_error}{self._hint()}"
                )

    def _hint(self) -> str:
        """One sentence a human can act on, appended to a transport failure.

        The raw error names the endpoint but not the reason a local relay
        failed, and "Cannot connect to proxy" with no address is a dead end.
        """
        if self.proxy_bypassed:
            return (
                "\n(The machine's proxy was unreachable, so this was retried "
                "directly - check MENU_LLM_PROXY / HTTP(S)_PROXY if direct "
                "connections are not allowed either.)"
            )
        if isinstance(self._proxy, str) and "://" in self._proxy:
            return f"\n(requests were routed through the proxy {self._proxy})"
        return ""


# UI words that are never dishes. Used only by the offline stub so the pipeline
# can be exercised without an API key - the real model does this semantically.
_UI_NOISE = re.compile(
    r"(销量|选规格|选规格|规格|已售|评价|详情|推荐|商品|购物车|立即购买|加入|库存|配送|"
    r"月售|起送|收藏|关注|分享|返回|首页|搜索|登录|注册|客服|营业|地址|电话)"
)
_UNIT_WORDS = {"份", "个", "只", "串", "碗", "盘", "起", "例", "杯", "瓶"}


class OfflineStubClient:
    """Deterministic stand-in used when no API key is configured.

    It only demonstrates plumbing: obvious UI noise is marked as not-a-dish and
    every real item comes back with an empty translation and a confidence of 0,
    so an uncon-enriched result is never mistaken for a real one.
    """

    model = "offline-stub"

    def chat_json_messages(self, messages: list[dict[str, Any]]) -> Any:
        # A photo cannot be read without a real model; failing loudly beats
        # returning "no dishes found", which reads like a bad photo.
        raise LLMError(
            "The offline stub cannot read images. Set MENU_LLM_API_KEY, or run "
            "with extractor='ocr'."
        )

    def chat_text(self, messages: list[dict[str, Any]]) -> str:
        # Same reasoning, one step further: the advisor answers from the scanned
        # facts without a model, so it must not be handed this client at all.
        raise LLMError(
            "The offline stub cannot answer free text. The advisor uses a "
            "deterministic answer when MENU_LLM_API_KEY is unset."
        )

    def chat_json(self, system: str, user: str) -> Any:
        start = user.find("[")
        records = json.loads(user[start:]) if start != -1 else []
        out: list[dict[str, Any]] = []
        for record in records:
            name = str(record.get("cn", "")).strip()
            noise = bool(_UI_NOISE.search(name)) or name in _UNIT_WORDS or len(name) <= 1
            out.append(
                {
                    "id": record.get("id"),
                    "is_dish": not noise,
                    "cn": name,
                    "pinyin": "",
                    "en": "",
                    "description": "Offline stub: set MENU_LLM_API_KEY to get real dish context.",
                    "ingredients": [],
                    "category": "",
                    "spice_level": 0,
                    "vegetarian": None,
                    "allergens": [],
                    "price_comment": "",
                    "confidence": 0.0,
                    "reason": "matched UI/unit noise pattern" if noise else "",
                }
            )
        return out


def default_client(settings: Settings) -> LLMClient:
    """Real client when a key exists, otherwise the offline stub."""
    if settings.llm_api_key:
        return OpenAICompatibleClient(settings)
    return OfflineStubClient()


def vision_client(settings: Settings) -> OpenAICompatibleClient:
    """Client configured for the vision model.

    Same endpoint, different model id *and* different capabilities: the vision
    model accepts {"type": "disabled"} while glm-5.x refuses it, so thinking
    level and JSON mode are passed in rather than read from the text-model
    settings.
    """
    return OpenAICompatibleClient(
        settings,
        model=settings.vision_model,
        thinking=settings.vision_thinking,
        json_mode=settings.vision_json_mode,
        timeout=settings.vision_timeout,
    )


def parse_dish_info(raw: dict[str, Any]) -> DishInfo:
    """Coerce one model object into DishInfo, dropping unknown keys."""
    payload: dict[str, Any] = {
        "is_dish": bool(raw.get("is_dish", True)),
        "pinyin": str(raw.get("pinyin") or ""),
        "en": str(raw.get("en") or ""),
        "description": str(raw.get("description") or ""),
        "category": str(raw.get("category") or ""),
        "price_comment": str(raw.get("price_comment") or ""),
        "reason": str(raw.get("reason") or ""),
    }
    ingredients = raw.get("ingredients")
    payload["ingredients"] = [str(item) for item in ingredients] if isinstance(ingredients, list) else []
    allergens = raw.get("allergens")
    payload["allergens"] = [str(item) for item in allergens] if isinstance(allergens, list) else []
    spice = raw.get("spice_level")
    payload["spice_level"] = min(3, max(0, int(spice))) if isinstance(spice, (int, float)) else 0
    vegetarian = raw.get("vegetarian")
    payload["vegetarian"] = vegetarian if isinstance(vegetarian, bool) else None
    try:
        payload["confidence"] = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
    except (TypeError, ValueError):
        payload["confidence"] = 0.0
    return DishInfo(**payload)
