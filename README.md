# Chinese Menu Translator with AI

Photograph a Chinese menu, get it read, translated and explained — then get a meal
recommended from it, not just a list of dishes.

Built for the *AI for Business* class. The photo goes in as pixels; what comes out is
structured JSON: every dish with its pinyin, English name, description, ingredients,
spice level, allergens, category, and a price comment that is *computed* rather than
described.

```
$ python -m menu_vision.cli test.jpg
=== test.jpg  1160x653  extractor=vision  vision=glm-4.5v
    {'extractor': 'vision', 'image_kb': 104.4, 'tokens': {...}, 'items_returned': 21, 'total_ms': 86922.7}
   1. 酸辣土豆丝  ¥11  ->  Hot and Sour Shredded Potato
      pinyin: suān là tǔ dòu sī   spice: 2/3   veg: True   cat: vegetable dish
      Thinly shredded potatoes stir-fried with chili peppers and vinegar, creating a classic
      Sichuan-style dish that is tangy, spicy, and crisp-tender.
      price: 11 CNY - the cheapest of the 21 priced dishes here (median 15 CNY).
   2. 凉瓜鸡蛋饭  ¥12  ->  Bitter Melon and Egg Rice Bowl
      ... 19 more
```

*(Real, trimmed output from the bundled `test.jpg`.)*

---

## What it does

| | |
|---|---|
| **Read** | One multimodal call reads the board and returns every dish it can see. A slower OCR path (`easyocr` + geometric layout + a text-only model call) is available as an alternative and returns inspectable bounding boxes. |
| **Translate** | Chinese name → pinyin + English name + a short description a foreigner can order from. |
| **Classify** | Spice level, vegetarian, category, ingredients, allergens. |
| **Price** | Prices are parsed, formatted and compared against the rest of the menu by arithmetic — cheapest dish, median, "above average for this section". |
| **Recommend** | A meal plan built against a stated party size, budget and hard restrictions, plus a ranked shortlist scored on eight weighted components. |
| **Remember** | Likes and dislikes are logged and folded into a soft taste profile. Inferred preferences only ever *nudge the ranking* — they never hide a dish. Only restrictions the user typed act as filters. |
| **Answer** | "How hot is this?" / "Does it have pork?" / "Is this enough for four?" are answered from the scan itself, for free. The model is called only for questions the data cannot answer. |

---

## Quick start (Windows)

The repo ships a launcher that bootstraps its own venv:

```bat
run_app.bat          :: http://localhost:8501
run_app.bat 8502     :: pick a port
```

It creates `.venv` on first run (~1 min), installs a current Streamlit, and walks
forward to a free port if 8501 is taken.

> **Edit one line first:** `run_app.bat` line 22 hardcodes `BOOTSTRAP_PY` to the
> interpreter path on the author's machine. Point it at your own Python 3.10+ before
> running.

### Manual setup

Requires Python 3.10+ (developed on 3.10.4).

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
pip install easyocr               # only if you want the OCR extractor
```

Run the app:

```bash
python -m streamlit run streamlit_app.py
```

Run the CLI (same pipeline, no browser):

```bash
python -m menu_vision.cli test.jpg test2.jpg --json out/
python -m menu_vision.cli test.jpg --no-llm          # transcribe names+prices only, fast
python -m menu_vision.cli test.jpg --extractor ocr --overlay out/debug.png
python -m menu_vision.cli test.jpg --vector          # show the payload sent to the model
```

`--no-llm` is the cheap path: it reads the photo but skips translation and
classification entirely.

---

## Configuration

Copy the template and fill in your key:

```bash
cp .env.example .env
```

`.env` is git-ignored. Real environment variables always win over the file, so CI and
deployments do not need it; `menu_vision/config.py` reads both.

The important ones:

| Variable | Default | Meaning |
|---|---|---|
| `MENU_LLM_API_KEY` | *(empty)* | Required. Falls back to `OPENAI_API_KEY` if unset. |
| `MENU_LLM_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4` | Any OpenAI-compatible endpoint. |
| `MENU_LLM_MODEL` | `glm-5.3-flash` | Text model for translation and classification. |
| `MENU_EXTRACTOR` | `vision` | `vision` (one multimodal call) or `ocr` (easyocr pipeline). |
| `MENU_VISION_MODEL` | `glm-4.5v` | Model used by the vision extractor. |
| `MENU_LLM_THINKING` | `low` | `low` / `high` / `max` / `disabled` / `auto`. |
| `MENU_LLM_PROXY` | `auto` | `auto`, `off`, or an explicit URL. See below. |
| `MENU_DATA_DIR` | `./data` | Where the taste profile and event log live. |

Everything is documented inline in [`.env.example`](.env.example) — including why
`MENU_LLM_JSON_MODE` is off for `glm-5.3-flash` and why vision reasoning stays on `low`.

**Swapping providers** needs no code change. The client speaks plain
`POST /chat/completions` over `requests`; there is no vendor SDK. Zhipu, DeepSeek, Qwen
and OpenAI are commented-out examples in `.env.example`. Vision needs a multimodal model.

**The proxy setting is not incidental.** On a machine where a VPN client left
`HTTP_PROXY` pointing at a dead port, every call fails with `Cannot connect to proxy`
and looks like an API outage. `MENU_LLM_PROXY=auto` uses the system proxy but drops it
and retries direct on the first `ProxyError`; `off` never uses one.

---

## How it works

Two extractors sit behind a single function, `menu_vision.pipeline.scan_menu()`, and
converge on the same `MenuResult`. The UI, the CLI and any future HTTP layer never
branch on which one ran.

```
                    ┌──────────────── vision (default) ────────────────┐
 photo ──► encode ──┤ one multimodal call: read the board, explain the  ├──► items
                    │ dishes. ~90 s. No coordinates.                    │
                    └──────────────────────────────────────────────────┘

                    ┌──────────────── ocr ─────────────────────────────┐
 photo ──► easyocr ─┤ boxes → geometric row grouping → price attach    │
                    │ → text-only model call on {id, cn, price}        │
                    │ 150-300 s. Every price tied to a box.            │
                    └──────────────────────────────────────────────────┘
                                        │
                          pricing · recommend · advisor
                          (arithmetic and rules, not prose)
```

The vision path is the default because it is faster and far more accurate on real phone
photos. The OCR path is kept because it can show you *where* it read something, which is
what the `--overlay` debug image draws.

| Module | Responsibility |
|---|---|
| `pipeline.py` | Orchestration and the extractor switch. The seam for an HTTP layer. |
| `vision.py` | Image encoding and the multimodal prompt/parse. |
| `ocr.py` | Thin `easyocr` wrapper. Imported lazily, so the rest of the package runs without it. |
| `layout.py` | Flat OCR boxes → menu rows with prices attached. Pure functions, no OCR import. |
| `llm.py` | OpenAI-compatible client: batching, retries with backoff, JSON repair, proxy fallback. |
| `models.py` | Pydantic models: `MenuResult`, `EnrichedItem`, `DishInfo`. |
| `pricing.py` | Price parsing, formatting and comparison. |
| `recommend.py` | Meal construction and the eight-component score. |
| `advisor.py` | Question answering: computed answers first, model second. |
| `preferences.py` | Taste profile and the like/dislike event log. |
| `report.py` | Presentation helpers. No Streamlit imports — unit-testable and reusable. |
| `config.py` | `Settings.from_env()`. Immutable, network-free. |
| `cli.py` | The command-line harness. |

Three design rules the code actually enforces:

1. **The score is computed, never described.** The model used to write price comments and
   called three different dishes "the cheapest rice dish". Claims that are arithmetic are
   now arithmetic.
2. **Hard restrictions come only from the user.** Anything inferred from taps can only
   reorder the list; it can never remove a dish.
3. **The UI is a thin shell.** `streamlit_app.py` arranges widgets. All reading lives in
   `menu_vision`, all drawing in `menu_vision.report`.

### The app

Five tabs, driven by one scan: **Plan** (the meal, the budget maths, follow-up
questions), **Dishes** (per-dish detail), **Price table**, **Classification**, and
**Behind the scenes** (timings, token counts, warnings, raw payload). The sidebar holds
"Your table" — party size, budget, restrictions — and the taste memory.

Four sample photos are bundled and selectable in the app, each chosen to break a
different assumption: `test7.jpg` (bilingual, soft focus — OCR reads no Chinese),
`test4.jpg` (four columns, left one cropped), `test.jpg` (printed canteen menu, 21
prices), `test2.jpg` (skewers, no prices printed).

---

## Output

`MenuResult` is the contract — `--json` writes `model_dump()`, and an API handler would
be `return scan_menu(await file.read()).model_dump()`:

```jsonc
{
  "source": "test.jpg",
  "image_size": [1160, 653],
  "extractor": "vision",
  "llm_model": "glm-4.5v",
  "items": [{
    "id": 1, "cn": "酸辣土豆丝", "price_value": 11.0, "price_display": "¥11",
    "dish": {
      "en": "Hot and Sour Shredded Potato", "pinyin": "suān là tǔ dòu sī",
      "category": "vegetable dish", "spice_level": 2, "vegetarian": true,
      "ingredients": ["potato", "vinegar", "dried chilli"],
      "allergens": [],
      "price_comment": "11 CNY - the cheapest of the 21 priced dishes here (median 15 CNY)."
    }
  }],
  "dropped": [{ "cn": "本店招牌", "reason": "not a purchasable item" }],
  "warnings": [],
  "metrics": { "total_ms": 86922.7, "tokens": { "total_tokens": 9864 } }
}
```

`dropped` matters as much as `items`: watermarks, headers and stray text are recorded
with a reason instead of silently vanishing, which is how extraction was tuned.

---

## Tests

Nine `unittest` modules, 197 tests, no browser and no network for the bulk of them:

```bash
python -m unittest discover -s tests -t tests
```

`-t tests` is required because `tests/` is not a package. Without `easyocr` installed you
should see **190 pass, 4 skipped, 3 error** — the three errors are precisely the tests
that drive the OCR extractor (`ModuleNotFoundError: easyocr`). The default vision path
needs no OCR dependency at all.

The Streamlit page has a headless smoke test using `AppTest`, which drives the real
widgets and inspects the element tree. It runs real OCR, so it is opt-in:

```bash
MENU_E2E=1 python -m unittest discover -s tests -t tests -p "test_streamlit_app.py"
```

---

## Layout

```
menu_vision/           the package — pipeline, extractors, models, logic
streamlit_app.py       the demo page (widgets only)
run_app.bat            Windows launcher: venv bootstrap + free-port detection
requirements.txt       core deps (easyocr deliberately left out)
tests/                 197 unit tests
.env.example           every setting, documented, key left blank
OCR_test.ipynb         the original OCR experiments and overlay debugging
test*.jpg              sample menus (also selectable in the app)
out/                   generated JSON, overlays, logs — git-ignored
data/                  taste profile + event log — git-ignored, user data
```

---

## Known limitations

- **`easyocr` is an optional, heavy dependency** (it pulls PyTorch). The default vision
  path avoids it entirely; only the OCR extractor and three of the tests need it.
- **Latency is provider-bound.** Measured against free-tier Zhipu: **87 s** for a
  single 21-dish board on the vision path, 150–300 s on the OCR path. `llm_timeout`
  defaults to 300 s of *silence*, not a total budget, because a reasoning model can
  think for 90 s before its first byte arrives.
- **Reasoning models are not deterministic.** `glm-5.3-flash` always reasons and cannot
  switch it off; setting `MENU_LLM_JSON_MODE=1` makes it collapse an array to a single
  record, which is why the default is off.
- **Requirements are floors, not a lockfile.** `requirements.txt` lists minimum versions
  verified against the `.venv` used to develop this: Python 3.10.4, Streamlit 1.65,
  pydantic 2.9.2, Pillow 10.4.0. Streamlit is pinned by *feature probe*, not version
  number, because the app uses `st.image(width="stretch")` and `st.metric(border=True)`.

## Privacy

Photos are sent to whichever model provider `.env` points at; nothing else leaves the
machine. `data/` holds your taste profile and like/dislike history as plain JSON and
JSONL — delete it and the app forgets you.
