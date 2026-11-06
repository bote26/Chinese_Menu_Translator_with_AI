"""Tests for the LLM layer: request shape, JSON recovery, and merging.

Runs a throwaway OpenAI-compatible server on localhost, so the real
`OpenAICompatibleClient` code path (HTTP + parsing + retries) is exercised
without an API key or network access.
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from menu_vision.config import Settings  # noqa: E402
from menu_vision.llm import (  # noqa: E402
    LLMError,
    OpenAICompatibleClient,
    OfflineStubClient,
    _DIRECT,
    _extract_json,
    build_system_prompt,
    build_user_prompt,
    parse_dish_info,
    resolve_proxy,
)
from menu_vision.pipeline import scan_menu  # noqa: E402


class _Recorder:
    """Collects what the client sent so the test can assert on it."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.fail_next = 0
        # Which status a scripted failure uses. 400 exercises the parameter
        # degrade; 500 exercises the retry+backoff path.
        self.fail_status = 400
        self.reject_json_mode = False
        # Emulates glm-5.x: reasons unconditionally, HTTP 400 for any thinking
        # value except a level.
        self.reject_thinking_off = False
        self.content = "[]"
        # Optional per-request script, so a test can change the reply between
        # retries (e.g. a bad first answer, a good second one).
        self.contents: list[str] | None = None


def _reply_for(recorder: _Recorder) -> str:
    """Scripted reply for the current request, else the constant one."""
    if not recorder.contents:
        return recorder.content
    index = min(len(recorder.requests) - 1, len(recorder.contents) - 1)
    return recorder.contents[index]


def _make_server(recorder: _Recorder) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence test output
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            recorder.requests.append({"path": self.path, "body": body,
                                      "auth": self.headers.get("Authorization")})
            if recorder.reject_json_mode and "response_format" in body:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":"response_format is not supported"}')
                return
            thinking = body.get("thinking") or {}
            if recorder.reject_thinking_off and thinking.get("type") != "enabled":
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": {"code": "1210", "message":
                              "该模型始终思考，不支持关闭思考；请使用 low、high 或 max。"}},
                    ensure_ascii=False).encode("utf-8"))
                return
            if recorder.fail_next > 0:
                recorder.fail_next -= 1
                self.send_response(recorder.fail_status)
                self.end_headers()
                self.wfile.write(b'{"error":"bad request"}')
                return
            payload = {"choices": [{"message": {"content": _reply_for(recorder)}}]}
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class TestJsonRecovery(unittest.TestCase):
    def test_strips_code_fences(self):
        self.assertEqual(_extract_json('```json\n[{"id": 1}]\n```'), [{"id": 1}])

    def test_ignores_leading_prose(self):
        self.assertEqual(_extract_json('Here you go:\n[{"id": 2}]'), [{"id": 2}])

    def test_raises_on_non_json(self):
        with self.assertRaises(LLMError):
            _extract_json("I cannot help with that.")


class TestDishInfoCoercion(unittest.TestCase):
    def test_clamps_and_defaults_bad_values(self):
        info = parse_dish_info(
            {"id": 1, "is_dish": True, "spice_level": 9, "confidence": "high",
             "ingredients": "potato", "vegetarian": "yes"}
        )
        self.assertEqual(info.spice_level, 3)       # clamped
        self.assertEqual(info.confidence, 0.0)      # unparseable -> 0
        self.assertEqual(info.ingredients, [])      # wrong type -> empty
        self.assertIsNone(info.vegetarian)          # non-bool -> None

    def test_ignores_unknown_keys(self):
        info = parse_dish_info({"id": 1, "is_dish": True, "banana": 5})
        self.assertFalse(hasattr(info, "banana"))


class TestOpenAICompatibleClient(unittest.TestCase):
    def setUp(self):
        self.recorder = _Recorder()
        self.server = _make_server(self.recorder)
        self.addCleanup(self.server.shutdown)
        port = self.server.server_address[1]
        self.settings = Settings(
            llm_base_url=f"http://127.0.0.1:{port}/v1",
            llm_api_key="test-key",
            llm_model="mock-model",
            llm_timeout=10,
        )
        self.client = OpenAICompatibleClient(self.settings)

    def test_sends_expected_request(self):
        self.recorder.content = '[{"id": 1, "is_dish": true}]'
        result = self.client.chat_json("sys", "user")
        self.assertEqual(result, [{"id": 1, "is_dish": True}])

        sent = self.recorder.requests[0]
        self.assertEqual(sent["path"], "/v1/chat/completions")
        self.assertEqual(sent["auth"], "Bearer test-key")
        self.assertEqual(sent["body"]["model"], "mock-model")
        self.assertEqual(sent["body"]["messages"][0]["role"], "system")
        self.assertEqual(sent["body"]["response_format"], {"type": "json_object"})

    def test_unwraps_items_object(self):
        # json_object mode sometimes forces an object wrapper.
        self.recorder.content = '{"items": [{"id": 7, "is_dish": true}]}'
        self.assertEqual(self.client.chat_json("s", "u"), [{"id": 7, "is_dish": True}])

    def test_unwraps_entries_object(self):
        """glm-5.x names the json_object wrapper "entries", not "items"."""
        self.recorder.content = '{"entries": [{"id": 11, "is_dish": true}]}'
        self.assertEqual(self.client.chat_json("s", "u"), [{"id": 11, "is_dish": True}])

    def test_unwraps_unknown_single_list_key(self):
        # Any lone top-level array is the payload, whatever it is called.
        self.recorder.content = '{"dishes_found": [{"id": 12}]}'
        self.assertEqual(self.client.chat_json("s", "u"), [{"id": 12}])

    def test_drops_json_mode_when_only_one_record_comes_back(self):
        """glm-5.3-flash answers json_object mode with a single record.

        Losing the whole menu to one dish is worse than losing the JSON
        guarantee, so the client turns the mode off and asks again.
        """
        self.recorder.contents = [
            '{"id": 1, "is_dish": true}',
            '[{"id": 1}, {"id": 2}]',
        ]
        self.assertEqual(self.client.chat_json("s", "u"), [{"id": 1}, {"id": 2}])
        self.assertIn("response_format", self.recorder.requests[0]["body"])
        self.assertNotIn("response_format", self.recorder.requests[-1]["body"])

    def test_json_mode_can_be_configured_off(self):
        self.recorder.content = '[{"id": 13}]'
        client = OpenAICompatibleClient(Settings(
            llm_base_url=self.settings.llm_base_url, llm_api_key="k",
            llm_model="glm-5.3-flash", llm_timeout=10, llm_json_mode=False,
        ))
        client.chat_json("s", "u")
        self.assertNotIn("response_format", self.recorder.requests[0]["body"])

    def test_retries_once_after_server_error(self):
        self.recorder.fail_next = 1
        self.recorder.content = '[{"id": 3}]'
        self.assertEqual(self.client.chat_json("s", "u"), [{"id": 3}])
        self.assertEqual(len(self.recorder.requests), 2)

    def test_raises_when_server_keeps_failing(self):
        self.recorder.fail_next = 99
        client = OpenAICompatibleClient(Settings(
            llm_base_url=self.settings.llm_base_url, llm_api_key="k",
            llm_model="m", llm_timeout=10, llm_max_retries=0,
        ))
        with self.assertRaises(LLMError):
            client.chat_json("s", "u")


    def test_drops_unsupported_params_and_still_succeeds(self):
        """Cheap providers 400 on extras; degrade one at a time, don't fail.

        Optional params are dropped individually, so this may cost one extra
        round trip per unsupported parameter (thinking, then response_format).
        """
        self.recorder.reject_json_mode = True
        self.recorder.content = '[{"id": 4, "is_dish": true}]'
        self.assertEqual(self.client.chat_json("s", "u"), [{"id": 4, "is_dish": True}])
        first = self.recorder.requests[0]["body"]
        self.assertIn("response_format", first)
        self.assertIn("thinking", first)
        self.assertNotIn("response_format", self.recorder.requests[-1]["body"])
        self.assertLessEqual(len(self.recorder.requests), 3)

    def test_sends_reasoning_level_by_default(self):
        """Extraction wants the lightest level: reasoning tokens bill as output."""
        self.recorder.content = '[{"id": 5}]'
        self.client.chat_json("s", "u")
        self.assertEqual(
            self.recorder.requests[0]["body"]["thinking"],
            {"type": "enabled", "level": "low"},
        )

    def test_thinking_can_be_disabled_on_models_that_allow_it(self):
        self.recorder.content = '[{"id": 5}]'
        client = OpenAICompatibleClient(Settings(
            llm_base_url=self.settings.llm_base_url, llm_api_key="k",
            llm_model="m", llm_timeout=10, llm_thinking="disabled",
        ))
        client.chat_json("s", "u")
        self.assertEqual(self.recorder.requests[0]["body"]["thinking"], {"type": "disabled"})

    def test_falls_back_to_a_level_when_model_always_reasons(self):
        """glm-5.3-flash 400s on thinking=disabled; retry with level=low."""
        self.recorder.reject_thinking_off = True
        self.recorder.content = '[{"id": 8}]'
        client = OpenAICompatibleClient(Settings(
            llm_base_url=self.settings.llm_base_url, llm_api_key="k",
            llm_model="glm-5.3-flash", llm_timeout=10, llm_thinking="disabled",
        ))
        self.assertEqual(client.chat_json("s", "u"), [{"id": 8}])
        bodies = [request["body"] for request in self.recorder.requests]
        self.assertEqual(bodies[0]["thinking"], {"type": "disabled"})
        self.assertEqual(bodies[-1]["thinking"], {"type": "enabled", "level": "low"})
        self.assertEqual(len(bodies), 2)

    def test_learns_the_always_reasoning_capability(self):
        """Discovery costs one 400 once, not once per call."""
        self.recorder.reject_thinking_off = True
        self.recorder.content = '[{"id": 9}]'
        client = OpenAICompatibleClient(Settings(
            llm_base_url=self.settings.llm_base_url, llm_api_key="k",
            llm_model="glm-5.3-flash", llm_timeout=10, llm_thinking="disabled",
        ))
        client.chat_json("s", "u")
        before = len(self.recorder.requests)
        client.chat_json("s", "u")
        self.assertEqual(len(self.recorder.requests) - before, 1)
        self.assertEqual(self.recorder.requests[-1]["body"]["thinking"],
                         {"type": "enabled", "level": "low"})

    def test_can_keep_thinking_enabled(self):
        self.recorder.content = '[{"id": 6}]'
        client = OpenAICompatibleClient(Settings(
            llm_base_url=self.settings.llm_base_url, llm_api_key="k",
            llm_model="m", llm_timeout=10, llm_thinking="auto",
        ))
        client.chat_json("s", "u")
        self.assertNotIn("thinking", self.recorder.requests[0]["body"])

    def test_records_token_usage(self):
        self.recorder.content = '[{"id": 7}]'
        self.client.chat_json("s", "u")
        self.assertIn("prompt_tokens", self.client.last_usage)

    def test_requires_api_key(self):
        with self.assertRaises(LLMError):
            OpenAICompatibleClient(Settings(llm_api_key=""))


class TestPrompt(unittest.TestCase):
    def test_payload_carries_chinese_name_and_price(self):
        prompt = build_user_prompt([{"id": 1, "cn": "酸辣土豆丝", "price": "11元",
                                     "price_value": 11.0, "section": None}])
        self.assertIn("酸辣土豆丝", prompt)
        self.assertIn("11元", prompt)
        self.assertIn('"id":1', prompt.replace(" ", ""))

    def test_system_prompt_names_target_language(self):
        self.assertIn("English", build_system_prompt(Settings()))
        self.assertIn("Spanish", build_system_prompt(Settings(target_language="Spanish")))


class TestMergeIntoResult(unittest.TestCase):
    """The pipeline must attach model output by id and honour is_dish."""

    class FakeClient:
        model = "fake-model"

        def chat_json(self, system, user):
            start = user.find("[")
            records = json.loads(user[start:])
            out = []
            for record in records:
                keeps = "份" not in record["cn"]
                out.append({
                    "id": record["id"], "is_dish": keeps, "cn": record["cn"],
                    "pinyin": "suan la tu dou si", "en": "Hot and Sour Shredded Potato",
                    "description": "Shredded potato stir-fried with vinegar and chili.",
                    "ingredients": ["potato", "dried chili", "vinegar"],
                    "category": "vegetable", "spice_level": 2, "vegetarian": True,
                    "allergens": [], "price_comment": "Cheap for a canteen dish.",
                    "confidence": 0.9, "reason": "" if keeps else "unit word",
                })
            return out

    def test_merges_translation_and_drops_non_dishes(self):
        image = ROOT / "test.jpg"
        if not image.exists():
            self.skipTest("test.jpg not present")
        result = scan_menu(image, client=self.FakeClient(), extractor="ocr")
        self.assertEqual(result.llm_model, "fake-model")
        self.assertTrue(result.items)
        first = result.items[0]
        self.assertEqual(first.cn, "酸辣土豆丝")
        self.assertEqual(first.price_display, "¥11")
        self.assertEqual(first.dish.en, "Hot and Sour Shredded Potato")
        self.assertEqual(first.dish.spice_level, 2)
        self.assertFalse(result.dropped)

    def test_llm_failure_keeps_ocr_result_with_warning(self):
        class Broken:
            model = "broken"

            def chat_json(self, system, user):
                raise LLMError("upstream down")

        result = scan_menu(ROOT / "test.jpg", client=Broken(), extractor="ocr")
        self.assertEqual(len(result.items), 21)  # OCR work is not thrown away
        self.assertTrue(any("LLM enrichment failed" in w for w in result.warnings))


class TestOfflineStub(unittest.TestCase):
    def test_stub_flags_ui_noise(self):
        records = [{"id": 1, "cn": "选规格", "price": None, "price_value": None, "section": None},
                   {"id": 2, "cn": "酸辣土豆丝", "price": "11元", "price_value": 11.0, "section": None}]
        reply = OfflineStubClient().chat_json("sys", build_user_prompt(records))
        flags = {item["cn"]: item["is_dish"] for item in reply}
        self.assertFalse(flags["选规格"])
        self.assertTrue(flags["酸辣土豆丝"])


class TestChatText(unittest.TestCase):
    """The prose path shares the transport and must not inherit the JSON envelope."""

    def setUp(self):
        self.recorder = _Recorder()
        self.server = _make_server(self.recorder)
        self.addCleanup(self.server.shutdown)
        port = self.server.server_address[1]
        self.settings = Settings(
            llm_base_url=f"http://127.0.0.1:{port}/v1",
            llm_api_key="test-key",
            llm_model="mock-model",
            llm_timeout=10,
            llm_retry_backoff=0.01,
        )
        self.client = OpenAICompatibleClient(self.settings)

    def test_returns_prose_and_omits_response_format(self):
        self.recorder.content = "This one is mild. Ask for 微辣."
        answer = self.client.chat_text([{"role": "user", "content": "Is it spicy?"}])
        self.assertEqual(answer, "This one is mild. Ask for 微辣.")
        # A prose question with json_object either 400s or comes back wrapped in
        # JSON the reader would see verbatim.
        self.assertNotIn("response_format", self.recorder.requests[0]["body"])

    def test_retries_like_chat_json(self):
        self.recorder.content = "Answer."
        self.recorder.fail_status = 500
        self.recorder.fail_next = 1
        answer = self.client.chat_text([{"role": "user", "content": "q"}])
        self.assertEqual(answer, "Answer.")
        self.assertEqual(len(self.recorder.requests), 2)

    def test_strips_surrounding_whitespace(self):
        self.recorder.content = "\n\n  Mild.  \n"
        self.assertEqual(
            self.client.chat_text([{"role": "user", "content": "q"}]), "Mild."
        )

    def test_offline_stub_refuses_rather_than_pretending(self):
        with self.assertRaises(LLMError):
            OfflineStubClient().chat_text([{"role": "user", "content": "q"}])


class TestProxyHandling(unittest.TestCase):
    """A dead local proxy must degrade to a direct connection, not fail a scan.

    Regression test for `ProxyError('Cannot connect to proxy.')` on every vision
    call, because HTTP(S)_PROXY still pointed at a port a long-gone VPN client
    used to own.
    """

    def test_resolve_proxy_parsing(self):
        self.assertEqual(resolve_proxy("auto"), "auto")
        self.assertEqual(resolve_proxy(""), "auto")
        self.assertEqual(resolve_proxy(None), "auto")
        self.assertEqual(resolve_proxy("off"), _DIRECT)
        self.assertEqual(resolve_proxy("OFF"), _DIRECT)
        # A bare host:port is what people actually type.
        self.assertEqual(resolve_proxy("127.0.0.1:7890"), "http://127.0.0.1:7890")
        self.assertEqual(resolve_proxy("http://10.0.0.2:8080"), "http://10.0.0.2:8080")

    # Port 1 is reserved and nothing listens there, so connecting fails at once
    # with the same ProxyError a closed 7890 raises.
    DEAD_PROXY = "http://127.0.0.1:1"

    def _client_for(self, recorder: "_Recorder", **overrides) -> OpenAICompatibleClient:
        server = _make_server(recorder)
        self.addCleanup(server.shutdown)
        settings = {
            "llm_base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
            "llm_api_key": "k",
            "llm_model": "m",
            "llm_timeout": 10,
            "llm_proxy": self.DEAD_PROXY,
        }
        settings.update(overrides)
        return OpenAICompatibleClient(Settings(**settings))

    def test_dead_proxy_is_bypassed_without_spending_a_retry(self):
        recorder = _Recorder()
        recorder.content = '[{"id": 1, "is_dish": true}]'
        client = self._client_for(recorder)
        self.assertEqual(client.chat_json("s", "u"), [{"id": 1, "is_dish": True}])
        self.assertIsNotNone(client.proxy_bypassed)
        # Exactly one model call reached the server: the proxy failure was
        # healed inline, it did not eat the retry budget or duplicate a request.
        self.assertEqual(len(recorder.requests), 1)

    def test_bypass_is_remembered_for_later_calls(self):
        recorder = _Recorder()
        recorder.content = '[{"id": 2}]'
        client = self._client_for(recorder)
        client.chat_json("s", "u")
        client.chat_json("s", "u")
        self.assertEqual(len(recorder.requests), 2)
        self.assertEqual(client._proxy, _DIRECT)

    def test_proxy_off_ignores_the_environment(self):
        recorder = _Recorder()
        client = self._client_for(recorder, llm_proxy="off")
        self.assertFalse(client._session.trust_env)
        self.assertEqual(client._proxy, _DIRECT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
