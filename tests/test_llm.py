# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
import unittest
from unittest import mock

import _support  # noqa: F401  (puts the package on sys.path)

from omnievolve.llm import LLMClient, LLMConfig, LLMError

KEY = "test-key-value-123"


def client(provider, **kw):
    return LLMClient(LLMConfig(provider=provider, model="m-1", **kw))


class RequestTests(unittest.TestCase):
    def test_openai(self):
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": KEY}, clear=False):
            url, headers, payload = client("openai", temperature=0.2).build_request("S", "U")
        self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(headers["authorization"], f"Bearer {KEY}")
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "S"})
        self.assertIn("max_completion_tokens", payload)
        self.assertEqual(payload["temperature"], 0.2)

    def test_openai_compatible_server(self):
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": KEY}):
            url, _, payload = client("openai", base_url="http://localhost:8000/v1/").build_request(
                "S", "U"
            )
        self.assertEqual(url, "http://localhost:8000/v1/chat/completions")
        self.assertIn("max_tokens", payload)
        self.assertNotIn("temperature", payload)

    def test_anthropic(self):
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": KEY}):
            url, headers, payload = client("anthropic").build_request("S", "U")
        self.assertEqual(url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(headers["x-api-key"], KEY)
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(payload["system"], "S")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "U"}])

    def test_gemini_key_in_header_not_url(self):
        env = {"GOOGLE_API_KEY": KEY}
        with (
            mock.patch.dict("os.environ", env),
            mock.patch.dict("os.environ", {"GEMINI_API_KEY": ""}),
        ):
            url, headers, payload = client("gemini").build_request("S", "U")
        self.assertNotIn(KEY, url)
        self.assertTrue(url.endswith("/models/m-1:generateContent"))
        self.assertEqual(headers["x-goog-api-key"], KEY)
        self.assertEqual(payload["systemInstruction"]["parts"][0]["text"], "S")

    def test_ollama_needs_no_key(self):
        with mock.patch.dict("os.environ", {"OLLAMA_HOST": "127.0.0.1:11434"}):
            url, headers, payload = client("ollama").build_request("S", "U")
        self.assertEqual(url, "http://127.0.0.1:11434/api/chat")
        self.assertNotIn("authorization", headers)
        self.assertFalse(payload["stream"])

    def test_missing_key_names_the_variable_only(self):
        with mock.patch.dict("os.environ", {"MY_KEY": ""}):
            with self.assertRaisesRegex(LLMError, "MY_KEY"):
                client("openai", api_key_env="MY_KEY").build_request("S", "U")

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            LLMConfig(provider="nope", model="m").validate()
        with self.assertRaises(ValueError):
            LLMConfig.from_dict({"provider": "openai", "model": "m", "api_key": KEY})
        described = LLMConfig(provider="openai", model="m", api_key_env="X").describe()
        self.assertEqual(described["api_key_env"], "X")


class ResponseTests(unittest.TestCase):
    def test_parse_each_provider(self):
        samples = {
            "openai": {"choices": [{"message": {"content": "hi"}}]},
            "anthropic": {
                "content": [{"type": "thinking", "thinking": "x"}, {"type": "text", "text": "hi"}]
            },
            "gemini": {"candidates": [{"content": {"parts": [{"text": "h"}, {"text": "i"}]}}]},
            "ollama": {"message": {"content": "hi"}},
        }
        for provider, data in samples.items():
            with self.subTest(provider=provider):
                self.assertEqual(LLMClient.parse_response(provider, data), "hi")
        with self.assertRaises(LLMError):
            LLMClient.parse_response("openai", {"error": "x"})
        with self.assertRaises(LLMError):
            LLMClient.parse_response("ollama", {"message": {"content": "  "}})

    def test_retries_only_retryable_errors(self):
        calls = []

        def flaky(url, headers, payload, timeout):
            calls.append(1)
            raise LLMError("HTTP 401 from host: bad key")

        llm = LLMClient(LLMConfig(provider="ollama", model="m", retries=3), transport=flaky)
        with self.assertRaises(LLMError):
            llm.complete("S", "U")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
