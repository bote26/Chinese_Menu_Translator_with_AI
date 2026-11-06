"""Runtime configuration, read from environment variables.

Nothing here talks to the network; keep it a plain immutable object so the
future API layer can build one per worker at startup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from a local `.env` without overriding real env vars.

    Keeps the API key out of shell history and out of the code. Existing
    environment variables always win, so CI/production settings are unaffected.
    """
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if not candidate.is_file():
            continue
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- Extractor -------------------------------------------------------
    # Which layer reads the photo. This is the main architectural switch.
    #   "vision" - one multimodal call reads the board *and* explains every
    #              dish. 20-70 s per photo, best accuracy on real phone photos,
    #              but it returns no coordinates.
    #   "ocr"    - easyocr + geometric layout + a text-only model call.
    #              Slower (150-300 s on these boards) and it misreads dense
    #              Chinese, but every price is tied to a box you can inspect.
    extractor: str = "vision"
    vision_model: str = "glm-4.5v"
    # The photo is re-encoded before upload (EXIF rotation applied, RGB, JPEG).
    # Only genuinely huge images are shrunk: a 4-column board loses legibility
    # fast, and a wrong dish name costs more than the image tokens.
    vision_max_side: int = 2400
    vision_jpeg_quality: int = 90
    # A vision call measured 19-65 s. Shorter than the text-model timeout
    # because there is only one call and a hang should surface quickly.
    vision_timeout: float = 180.0
    # glm-4.5v accepts {"type": "disabled"}, and it is genuinely faster
    # (test5: 50 s vs 56 s), but with reasoning off it invented 13 nameless
    # rows on a board whose left column is cropped. "low" keeps it honest.
    vision_thinking: str = "low"
    # glm-4.5v answers a bare array in json_object mode, so the mode buys
    # nothing here; leave it off rather than risk the glm-5 truncation bug.
    vision_json_mode: bool = False

    # --- OCR -------------------------------------------------------------
    # ch_sim + en: menus mix Chinese with digits / latin ("Cola", "¥").
    ocr_langs: tuple[str, ...] = ("ch_sim", "en")
    ocr_gpu: bool = False
    # Boxes below this OCR score are dropped. Watermarks and decorative
    # borders consistently land at 0.00-0.25, real menu text at 0.9+.
    min_confidence: float = 0.30
    # Price glyphs are small and often anti-aliased, so easyocr scores them far
    # lower than dish names (a real "¥7.65" came back at 0.07). A box that is
    # unmistakably a price survives down to this floor, which recovers prices
    # without letting watermarks in - they are never price-shaped.
    price_conf_floor: float = 0.0
    # Upscale small photos before OCR; easyocr degrades on tiny text.
    upscale_below_width: int = 900

    # --- Layout ----------------------------------------------------------
    # Max vertical centre offset, as a fraction of glyph height, for two boxes
    # to count as the same row.
    row_tolerance: float = 0.55
    # A box this many times taller than the median is a title/section header.
    header_height_ratio: float = 1.45

    # --- LLM -------------------------------------------------------------
    # Any OpenAI-compatible endpoint works: Zhipu GLM, DeepSeek, Qwen/DashScope,
    # Moonshot, OpenAI. Swap provider with MENU_LLM_BASE_URL + MENU_LLM_MODEL.
    # Default is Zhipu because that is the account this project runs against.
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    llm_api_key: str = ""
    # glm-5.3-flash is free-tier and always reasons (see `llm_thinking`).
    llm_model: str = "glm-5.3-flash"
    # glm-5.3-flash reasons before every answer and the free tier is slow: a
    # 23-dish board measured 90 s for one call, so 90 s would abort mid-scan.
    # This is a read timeout (no bytes received), not a total budget.
    llm_timeout: float = 300.0
    llm_batch_size: int = 40
    llm_temperature: float = 0.2
    # Free tiers answer 429 ("model overloaded") routinely, so retries matter
    # more than raw price here. Backoff doubles on each attempt.
    llm_max_retries: int = 2
    llm_retry_backoff: float = 2.0
    # How to reach the endpoint.
    #   "auto"  - use whatever this machine advertises (HTTP(S)_PROXY env vars,
    #             then the Windows registry), but drop it and go direct on the
    #             first ProxyError. This is the fix for a proxy left behind by a
    #             VPN client: the port is closed, so every call used to die with
    #             "Cannot connect to proxy" and looked like an API outage.
    #   "off"   - never use a proxy (``trust_env = False``); direct always.
    #   <url>   - use exactly this proxy, e.g. http://127.0.0.1:7890.
    llm_proxy: str = "auto"
    # Reasoning effort. Reasoning tokens are billed as output and delay the
    # JSON, so extraction wants the lowest level that stays correct:
    #   "low" / "high" / "max" -> {"type": "enabled", "level": <value>}
    #   "disabled"             -> {"type": "disabled"}  (glm-4.x only)
    #   "auto"                 -> send nothing, use the provider's default
    # glm-5.3-flash cannot switch reasoning off at all - it answers HTTP 400
    # (code 1210, "该模型始终思考") for anything but a level, and the client
    # falls back to "low" automatically when that happens.
    llm_thinking: str = "low"
    # Ask for `response_format: json_object`. Helps providers that otherwise
    # ramble, but glm-5.3-flash collapses the array to a single record when the
    # mode is on, so it is off in `.env` for that model. The client also
    # self-heals: an unusable object reply turns the mode off and retries.
    llm_json_mode: bool = True

    # --- Output ----------------------------------------------------------
    currency: str = "CNY"
    target_language: str = "English"

    # --- Memory and recommendations --------------------------------------
    # Where the taste profile and the tap log live. User data, not cache: it is
    # read on every rerun and written when the user reacts to a dish.
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    # How many ranked dishes the shortlist shows under the meal plan.
    recommend_shortlist: int = 12

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        defaults = cls()
        return cls(
            extractor=os.getenv("MENU_EXTRACTOR", defaults.extractor),
            vision_model=os.getenv("MENU_VISION_MODEL", defaults.vision_model),
            vision_max_side=_env_int("MENU_VISION_MAX_SIDE", defaults.vision_max_side),
            vision_jpeg_quality=_env_int("MENU_VISION_QUALITY", defaults.vision_jpeg_quality),
            vision_timeout=_env_float("MENU_VISION_TIMEOUT", defaults.vision_timeout),
            vision_thinking=os.getenv("MENU_VISION_THINKING", defaults.vision_thinking),
            vision_json_mode=_env_bool("MENU_VISION_JSON_MODE", defaults.vision_json_mode),
            ocr_langs=tuple(
                lang.strip()
                for lang in os.getenv("MENU_OCR_LANGS", ",".join(defaults.ocr_langs)).split(",")
                if lang.strip()
            ),
            ocr_gpu=_env_bool("MENU_OCR_GPU", defaults.ocr_gpu),
            min_confidence=_env_float("MENU_MIN_CONFIDENCE", defaults.min_confidence),
            price_conf_floor=_env_float("MENU_PRICE_CONF_FLOOR", defaults.price_conf_floor),
            upscale_below_width=_env_int("MENU_UPSCALE_BELOW", defaults.upscale_below_width),
            row_tolerance=_env_float("MENU_ROW_TOLERANCE", defaults.row_tolerance),
            header_height_ratio=_env_float("MENU_HEADER_RATIO", defaults.header_height_ratio),
            llm_base_url=os.getenv("MENU_LLM_BASE_URL", defaults.llm_base_url),
            llm_api_key=os.getenv("MENU_LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            llm_model=os.getenv("MENU_LLM_MODEL", defaults.llm_model),
            llm_timeout=_env_float("MENU_LLM_TIMEOUT", defaults.llm_timeout),
            llm_batch_size=_env_int("MENU_LLM_BATCH", defaults.llm_batch_size),
            llm_max_retries=_env_int("MENU_LLM_RETRIES", defaults.llm_max_retries),
            llm_retry_backoff=_env_float("MENU_LLM_BACKOFF", defaults.llm_retry_backoff),
            llm_proxy=os.getenv("MENU_LLM_PROXY", defaults.llm_proxy),
            llm_thinking=os.getenv("MENU_LLM_THINKING", defaults.llm_thinking),
            llm_json_mode=_env_bool("MENU_LLM_JSON_MODE", defaults.llm_json_mode),
            llm_temperature=_env_float("MENU_LLM_TEMPERATURE", defaults.llm_temperature),
            currency=os.getenv("MENU_CURRENCY", defaults.currency),
            target_language=os.getenv("MENU_TARGET_LANGUAGE", defaults.target_language),
            data_dir=Path(os.getenv("MENU_DATA_DIR", str(defaults.data_dir))),
            recommend_shortlist=_env_int("MENU_RECOMMEND_SHORTLIST", defaults.recommend_shortlist),
        )
