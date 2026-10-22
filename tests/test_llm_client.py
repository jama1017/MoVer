"""Offline checks for MoVer's wrapper and its two LiteLLM entry points."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import litellm
import yaml

from mover._optional import is_module_available
from mover.synthesizers.llm_client import LLMClient


MESSAGES = [{"role": "user", "content": "Return JavaScript."}]


def chat_response(content="const tl = gsap.timeline();", finish_reason="stop"):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "returned-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }


class LLMClientTest(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.payload = chat_response()
        self.send = patch("httpx.Client.send", side_effect=self._send)
        self.send.start()
        self.addCleanup(self.send.stop)

    def _send(self, request, **kwargs):
        self.requests.append(request)
        return httpx.Response(200, request=request, json=self.payload)

    def client(self, model="openai/gpt-4.1", **params):
        return LLMClient(
            model,
            params={"api_key": "test-key", "max_retries": 0, "num_retries": 0, **params},
        )

    def test_constructor_copies_params(self):
        params = {"temperature": 0.2}
        client = LLMClient("openai/gpt-4.1", params=params)
        client.params["temperature"] = 0.7
        self.assertEqual(params, {"temperature": 0.2})
        self.assertEqual(LLMClient("openai/gpt-4.1").params, {})

    def test_chat_sdk_integration(self):
        params = {
            "api_key": "test-key", "api_base": "http://localhost:8123/v1",
            "max_retries": 0, "num_retries": 0, "temperature": 0.2,
        }
        messages = copy.deepcopy(MESSAGES)
        original = copy.deepcopy(params)
        client = LLMClient("openai/local/model", params=params)
        self.payload = chat_response(finish_reason="length")

        response = client.complete(messages)

        self.assertIsInstance(response, litellm.ModelResponse)
        self.assertEqual(response.model, "returned-model")
        self.assertEqual(response.usage.total_tokens, 5)
        self.assertEqual(response.choices[0].finish_reason, "length")
        self.assertEqual(params, original)
        self.assertEqual(messages, MESSAGES)
        request = self.requests[0]
        self.assertEqual(str(request.url), "http://localhost:8123/v1/chat/completions")
        body = json.loads(request.content)
        self.assertEqual(body["model"], "local/model")
        self.assertEqual(body["messages"], MESSAGES)
        self.assertEqual(body["temperature"], 0.2)

    def test_responses_sdk_integration(self):
        from litellm.types.llms.openai import ResponsesAPIResponse

        self.payload = {
            "id": "resp-test", "object": "response", "created_at": 1,
            "model": "gpt-4.1", "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{
                "type": "message", "id": "msg-test", "role": "assistant", "status": "incomplete",
                "content": [{"type": "output_text", "text": "partial", "annotations": []}],
            }],
            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        }
        response = self.client(
            "private-model", custom_llm_provider="openai", max_output_tokens=16,
        ).responses(MESSAGES)
        self.assertIsInstance(response, ResponsesAPIResponse)
        self.assertEqual(response.status, "incomplete")
        self.assertEqual(response.incomplete_details.reason, "max_output_tokens")
        self.assertEqual(response.output_text, "partial")
        self.assertEqual(response.usage.total_tokens, 5)
        self.assertEqual(self.requests[-1].url.path, "/v1/responses")
        body = json.loads(self.requests[-1].content)
        self.assertEqual(body["input"], MESSAGES)
        self.assertEqual(body["model"], "private-model")
        self.assertEqual(body["max_output_tokens"], 16)

    def test_sdk_settings_and_results_are_forwarded_unchanged(self):
        for streaming in (False, True):
            params = {
                "stream": streaming, "drop_params": True, "timeout": 15,
                "num_retries": 2, "retry_policy": {"RateLimitErrorRetries": 1},
                "fallbacks": ["openai/backup"], "caching": True,
                "custom_llm_provider": "openai", "model_info": {"id": "deployment"},
            }
            original = copy.deepcopy(params)
            client = LLMClient("private-model", params=params)
            for method, sdk_method, argument, value in (
                ("complete", "completion", "messages", MESSAGES),
                ("responses", "responses", "input", MESSAGES),
                ("responses", "responses", "input", "Return JavaScript."),
            ):
                with self.subTest(method=method, streaming=streaming, input=value):
                    result = iter(()) if streaming else object()
                    with patch.object(litellm, sdk_method, return_value=result) as call:
                        self.assertIs(getattr(client, method)(value), result)
                    call.assert_called_once_with(
                        model="private-model", **{argument: value}, **params,
                    )
            self.assertEqual(params, original)

    def test_sdk_errors_propagate_without_wrapper_retries(self):
        for method, sdk_method in (("complete", "completion"), ("responses", "responses")):
            with self.subTest(method=method):
                error = litellm.RateLimitError(
                    message="Rate limited", llm_provider="openai", model="gpt-4.1",
                )
                with patch.object(litellm, sdk_method, side_effect=error) as call:
                    with self.assertRaises(litellm.RateLimitError) as caught:
                        getattr(self.client(num_retries=2), method)(MESSAGES)
                self.assertIs(caught.exception, error)
                call.assert_called_once()

    def test_text_helper_returns_text(self):
        for text in ("const tl = gsap.timeline();", ""):
            with self.subTest(text=text):
                response = litellm.ModelResponse(**chat_response(text))
                with patch.object(litellm, "completion", return_value=response) as call:
                    self.assertEqual(self.client().create(MESSAGES), text)
                call.assert_called_once()

    def test_text_helper_rejects_missing_text(self):
        for response in (
            litellm.ModelResponse(**chat_response(content=None)),
            litellm.ModelResponse(choices=[]),
            iter(()),
        ):
            with self.subTest(response=response):
                with patch.object(litellm, "completion", return_value=response):
                    with self.assertRaisesRegex(ValueError, "returned no text"):
                        self.client().create(MESSAGES)

    def test_text_helper_rejects_streams_and_batches_before_request(self):
        for params in ({"stream": True}, {"model_list": []}):
            with self.subTest(params=params), self.assertRaisesRegex(ValueError, "use complete"):
                self.client(**params).create(MESSAGES)
        self.assertEqual(self.requests, [])

    def test_duplicate_arguments_fail_without_sending_requests(self):
        for method, name in (("complete", "model"), ("complete", "messages"), ("responses", "input")):
            with self.subTest(argument=name):
                client = LLMClient("openai/gpt-4.1", params={name: "duplicate"})
                with self.assertRaises(TypeError):
                    getattr(client, method)(MESSAGES)
        self.assertEqual(self.requests, [])

    def test_synthesizers_use_native_model_configuration(self):
        from mover.synthesizers.animation_synthesizer import AnimationSynthesizer
        from mover.synthesizers.mover_synthesizer import MoverSynthesizer
        from mover.synthesizers.prompt_rewriter import PromptRewriter

        for cls in (AnimationSynthesizer, MoverSynthesizer, PromptRewriter):
            with self.subTest(synthesizer=cls.__name__):
                default = cls()
                self.assertEqual(default.llm_client.model, "openai/gpt-4.1")
                self.assertEqual(default.llm_client.params, {})
                custom = cls(model="groq/llama-3.3-70b-versatile", params={"num_retries": 1})
                self.assertEqual(custom.llm_client.model, "groq/llama-3.3-70b-versatile")
                self.assertEqual(custom.llm_client.params, {"num_retries": 1})

    @unittest.skipUnless(
        all(is_module_available(name) for name in ("jacinle", "lark", "pyrealb", "torch", "treelib")),
        "requires mover[full]",
    )
    def test_pipeline_configs_use_native_models_and_reject_old_fields(self):
        from mover.pipeline import AnimationPipeline, load_config_from_file
        from test_mover_synthesizer import init_synthesizer

        configs = Path(__file__).resolve().parent.parent / "examples/configs"
        for name in ("config_starter.yaml", "config_teaser.yaml"):
            with self.subTest(config=name):
                config = load_config_from_file(str(configs / name))
                pipeline = AnimationPipeline(config)
                self.assertEqual(pipeline.animation_synthesizer.llm_client.model, config["animation_model"]["model"])
                self.assertEqual(pipeline.mover_synthesizer.llm_client.model, config["mover_model"]["model"])
                self.assertEqual(init_synthesizer(config["mover_model"]).llm_client.model, config["mover_model"]["model"])

        local_settings = {
            "animation_model": ("openai/mover-smoke", {
                "api_base": "http://127.0.0.1:18080/v1", "api_key": "EMPTY",
            }),
            "mover_model": ("ollama_chat/llama3.2:latest", {
                "api_base": "http://127.0.0.1:11434", "num_ctx": 4096,
            }),
        }
        for model_key, (model, params) in local_settings.items():
            config[model_key].update(model=model, params={**params, "temperature": None})
        pipeline = AnimationPipeline(config)
        for model_key, synthesizer in (
            ("animation_model", pipeline.animation_synthesizer),
            ("mover_model", pipeline.mover_synthesizer),
        ):
            model, params = local_settings[model_key]
            self.assertEqual(synthesizer.llm_client.model, model)
            self.assertEqual(synthesizer.llm_client.params, params)
        self.assertEqual(
            init_synthesizer(config["mover_model"]).llm_client.params,
            local_settings["mover_model"][1],
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            for model_key in local_settings:
                for field in ("name", "provider", "num_ctx", "vllm_serve_port"):
                    with self.subTest(section=model_key, obsolete=field):
                        legacy = copy.deepcopy(config)
                        legacy[model_key][field] = "old-setting"
                        path.write_text(yaml.safe_dump(legacy), encoding="utf-8")
                        with self.assertRaisesRegex(ValueError, f"Obsolete {model_key} settings"):
                            load_config_from_file(str(path))
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
