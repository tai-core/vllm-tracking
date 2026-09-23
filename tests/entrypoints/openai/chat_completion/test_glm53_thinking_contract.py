# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash thinking-contract tests (ASGI-level).

Fixes reviewed under thinking-contract-review-20260922-r1:
- self-contained fixtures: a synthetic tokenizer is built in a tmp dir by the
  test itself (tokenizers library), no /model dependency, no whole-module skip;
- real HTTP through the actual /v1/chat/completions router (TestClient),
  with a mock engine wired to the SAME HfRenderer the serving layer uses and
  producing synthetic RequestOutput — rejected requests must be HTTP 400 with
  zero engine calls, valid requests HTTP 200 with expected reasoning/content
  split, finish_reason and usage, streaming and nonstream;
- GLM53 scope is by resolved model_type=glm5_next + template variables; other
  models (ordinary non-reasoning template, switchable template, custom
  request template) must be unchanged;
- thinking/enable_thinking aliases, reasoning_effort='none', reverse
  overrides, same-level conflicts all covered;
- raw /v1/completions with text and token prompts exercised through HTTP.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from http import HTTPStatus

from vllm.config import MultiModalConfig
from vllm.entrypoints.openai.chat_completion.api_router import router as chat_router
from vllm.entrypoints.openai.completion.api_router import router as completion_router
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.hf import HfRenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.tokenizers.registry import cached_tokenizer_from_config

OPEN_M = "<think>"
CLOSE_M = "</think>"

# Faithful GLM-5.3-style always-thinking template: budget control only,
# generation prompt appends the open-think marker.
GLM53_TEMPLATE = (
    "{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort"
    " is defined and reasoning_effort in ['low', 'high'] else 'max' -%}"
    "{%- if effective_reasoning_effort is not none -%}"
    "Reasoning Effort: {{ effective_reasoning_effort | capitalize }}\n"
    "{%- endif -%}"
    "{%- set clear_thinking = clear_thinking if clear_thinking is defined"
    " else false -%}"
    "{% for m in messages %}{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}" + OPEN_M + "{% endif %}"
)
# Ordinary template: no reasoning variables at all.
PLAIN_TEMPLATE = (
    "{% for m in messages %}{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}Assistant:{% endif %}"
)
# Switchable template (GLM4.5/4.7-style): declares enable_thinking.
SWITCHABLE_TEMPLATE = (
    "{%- set enable_thinking = enable_thinking if enable_thinking is defined"
    " else true -%}"
    "{% for m in messages %}{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt and enable_thinking %}" + OPEN_M + "{% endif %}"
)

MODEL_NAME = "synthetic/glm53-test"


def _make_tokenizer_dir(tmp_path, chat_template: str | None = None) -> str:
    """Build a tiny self-contained tokenizer (tokenizers JSON) in tmp_path.
    When chat_template is given it is persisted as the tokenizer's own default
    (chat_template.jinja), modeling the DEFAULT deployment where no
    --chat-template is passed and the model provides its template."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders

    tok = Tokenizer(models.BPE())
    # a handful of pieces so encode/decode round-trips ASCII words
    pieces = {
        "<unk>": 0, "hello": 1, "world": 2, "hi": 3, "answer": 4,
        "thinking": 5, "reasoning": 6, "here": 7, "a": 8, "b": 9, "c": 10,
    }
    tok.model = models.WordLevel(vocab=pieces, unk_token="<unk>")
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.WordPiece(prefix="")
    d = tmp_path / "tokdir"
    d.mkdir()
    tok.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}))
    # config.json with a recognized model_type so config parsing succeeds
    (d / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
    }))
    if chat_template is not None:
        # transformers loads the model default from tokenizer_config.json
        (d / "tokenizer_config.json").write_text(json.dumps({
            "tokenizer_class": "PreTrainedTokenizerFast",
            "chat_template": chat_template,
        }))
    return str(d)


@dataclass
class MockHFConfig:
    model_type: str = "glm5_next"


@dataclass
class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    model = MODEL_NAME
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 256
    tokenizer_revision = None
    revision = None
    code_revision = None
    multimodal_config = MultiModalConfig()
    hf_config: MockHFConfig = field(default_factory=MockHFConfig)
    hf_text_config: MockHFConfig = field(default_factory=MockHFConfig)
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    override_generation_config: dict[str, Any] = field(default_factory=dict)
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    renderer_num_workers: int = 1
    enable_prompt_embeds: bool = False

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


class MockEngine:
    """Mock engine whose renderer is the SAME HfRenderer the serving layer
    uses, and which yields a synthetic RequestOutput per generate() call."""

    def __init__(self, model_config, renderer, output_text=None,
                 stream_chunks=None, finish_reason="stop"):
        self.model_config = model_config
        self.renderer = renderer  # REAL renderer — same instance as serving
        from unittest.mock import MagicMock
        self.input_processor = MagicMock()
        self.errored = False
        async def _no_trace():
            return False
        self.is_tracing_enabled = _no_trace
        self.generate_calls = 0
        self.output_text = output_text if output_text is not None else (
            "thinking here" + CLOSE_M + "hello world")
        self.stream_chunks = stream_chunks
        self.finish_reason = finish_reason

    def check_admission(self, n: int = 1, request_id: str | None = None) -> None:
        pass

    async def generate(self, *a, **kw):
        self.generate_calls += 1
        req = a[0] if a else kw.get("prompt")
        n_prompt = getattr(req, "prompt_token_ids", None)
        n = len(n_prompt) if n_prompt else 1
        # Stream in realistic increments: text chunks with matching token
        # counts, finish_reason only on the final increment.
        chunks = self.stream_chunks if self.stream_chunks else [self.output_text]
        n_tok_total = 0
        for i, chunk in enumerate(chunks):
            last = i == len(chunks) - 1
            n_tok_total += max(1, len(chunk.split()))
            out = RequestOutput(
                request_id="test-req",
                prompt="hi",
                prompt_token_ids=list(range(n)),
                outputs=[CompletionOutput(
                    index=0, text=chunk, token_ids=[4] * max(1, len(chunk.split())),
                    cumulative_logprob=0.0, logprobs=None,
                    finish_reason=(self.finish_reason if last else None))],
                finished=last, prompt_logprobs=None,
            )
            yield out


def _build_app(tmp_path, *, template=GLM53_TEMPLATE,
               tokenizer_template=None,
               model_type="glm5_next",
               default_chat_template_kwargs=None,
               output_text=None, stream_chunks=None,
               finish_reason="stop",
               trust_request_templates=True):
    # tokenizer_template: what the tokenizer dir persists as the model's own
    # default (defaults to `template`). The constructor `template` is the
    # explicit server value (None => default deployment without
    # --chat-template).
    tok_dir = _make_tokenizer_dir(
        tmp_path, chat_template=(template if tokenizer_template is None
                                 else tokenizer_template))
    model_config = MockModelConfig()
    model_config.hf_config = MockHFConfig(model_type=model_type)
    model_config.model = tok_dir
    model_config.tokenizer = tok_dir
    model_config.tokenizer_mode = "hf"  # skip mistral auto-detect on paths
    renderer = HfRenderer(
        MockVllmConfig(model_config, parallel_config=MockParallelConfig()),
        cached_tokenizer_from_config(model_config),
    )
    engine = MockEngine(model_config, renderer, output_text=output_text,
                        stream_chunks=stream_chunks,
                        finish_reason=finish_reason)
    models = OpenAIServingModels(
        engine, [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)])
    online_renderer = OnlineRenderer(
        model_config=model_config,
        renderer=renderer,
        request_logger=None,
        chat_template=template,
        chat_template_content_format="auto",
        trust_request_chat_template=trust_request_templates,
        default_chat_template_kwargs=default_chat_template_kwargs,
    )
    serving_chat = OpenAIServingChat(
        engine, models, response_role="assistant",
        online_renderer=online_renderer,
        chat_template=template,
        chat_template_content_format="auto",
        request_logger=None,
        reasoning_parser="glm45",
        trust_request_chat_template=trust_request_templates,
        default_chat_template_kwargs=default_chat_template_kwargs,
    )
    serving_completion = OpenAIServingCompletion(
        engine, models,
        online_renderer=online_renderer,
        request_logger=None,
    )
    app = FastAPI()
    app.state.openai_serving_chat = serving_chat
    app.state.openai_serving_completion = serving_completion
    app.include_router(chat_router)
    app.include_router(completion_router)
    return app, engine


def _chat_body(**over):
    body = {"model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32, "temperature": 0.0}
    body.update(over)
    return body


# ---------------- GLM53 scope: rejections (HTTP 400, engine untouched) -------

def test_glm53_enable_thinking_false_http400_no_engine(tmp_path):
    app, engine = _build_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions",
                   json=_chat_body(chat_template_kwargs={"enable_thinking": False}))
    assert r.status_code == HTTPStatus.BAD_REQUEST, r.text
    assert r.json()["error"]["param"] == "chat_template_kwargs.enable_thinking"
    assert engine.generate_calls == 0


def test_glm53_thinking_alias_false_http400(tmp_path):
    app, engine = _build_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions",
                   json=_chat_body(chat_template_kwargs={"thinking": False}))
    assert r.status_code == HTTPStatus.BAD_REQUEST, r.text
    assert r.json()["error"]["param"] == "chat_template_kwargs.thinking"
    assert engine.generate_calls == 0


def test_glm53_reasoning_effort_none_http400(tmp_path):
    app, engine = _build_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions",
                   json=_chat_body(reasoning_effort="none"))
    assert r.status_code == HTTPStatus.BAD_REQUEST, r.text
    assert r.json()["error"]["param"] == "reasoning_effort"
    assert engine.generate_calls == 0


def test_glm53_same_level_conflict_false_false_http400(tmp_path):
    app, engine = _build_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions", json=_chat_body(
            chat_template_kwargs={"thinking": False, "enable_thinking": False}))
    assert r.status_code == HTTPStatus.BAD_REQUEST
    assert engine.generate_calls == 0


def test_glm53_conflict_thinking_false_enable_true_http400(tmp_path):
    app, engine = _build_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions", json=_chat_body(
            chat_template_kwargs={"thinking": False,
                                  "enable_thinking": True}))
    assert r.status_code == HTTPStatus.BAD_REQUEST
    err = r.json()["error"]
    # the FALSE field is named as the one disabling thinking, not the true one
    assert err["param"] == "chat_template_kwargs.thinking"
    assert "contradicts" in err["message"]
    assert engine.generate_calls == 0


def test_glm53_conflict_enable_false_thinking_true_http400(tmp_path):
    app, engine = _build_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions", json=_chat_body(
            chat_template_kwargs={"enable_thinking": False,
                                  "thinking": True}))
    assert r.status_code == HTTPStatus.BAD_REQUEST
    err = r.json()["error"]
    assert err["param"] == "chat_template_kwargs.enable_thinking"
    assert engine.generate_calls == 0


def test_glm53_effort_none_with_enable_true_http400(tmp_path):
    # none is its own unsupported semantic; a sibling true alias is a
    # contradiction, named as such, and none itself is the param.
    app, engine = _build_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions", json=_chat_body(
            reasoning_effort="none",
            chat_template_kwargs={"enable_thinking": True}))
    assert r.status_code == HTTPStatus.BAD_REQUEST
    err = r.json()["error"]
    assert err["param"] == "reasoning_effort"
    assert "enable_thinking=true" in err["message"]
    assert engine.generate_calls == 0


def test_glm53_effort_none_with_thinking_true_http400(tmp_path):
    app, engine = _build_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions", json=_chat_body(
            reasoning_effort="none",
            chat_template_kwargs={"thinking": True}))
    assert r.status_code == HTTPStatus.BAD_REQUEST
    err = r.json()["error"]
    assert err["param"] == "reasoning_effort"
    assert "thinking=true" in err["message"]
    assert engine.generate_calls == 0


def test_glm53_server_false_request_true_allowed_override(tmp_path):
    # request-level true overrides server-default false for this profile:
    # allowed, reaches the engine, and the parser stays in thinking mode.
    app, engine = _build_app(
        tmp_path, default_chat_template_kwargs={"enable_thinking": False})
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions",
                   json=_chat_body(chat_template_kwargs={"enable_thinking": True}))
    assert r.status_code == HTTPStatus.OK, r.text
    msg = r.json()["choices"][0]["message"]
    assert msg["reasoning"] == "thinking here"
    assert msg["content"] == "hello world"
    assert engine.generate_calls == 1


def test_glm53_reverse_override_server_default_true_request_false(tmp_path):
    app, engine = _build_app(
        tmp_path, default_chat_template_kwargs={"enable_thinking": True})
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions",
                   json=_chat_body(chat_template_kwargs={"enable_thinking": False}))
    assert r.status_code == HTTPStatus.BAD_REQUEST
    assert engine.generate_calls == 0


def test_glm53_server_default_false_http400(tmp_path):
    app, engine = _build_app(
        tmp_path, default_chat_template_kwargs={"enable_thinking": False})
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions", json=_chat_body())
    assert r.status_code == HTTPStatus.BAD_REQUEST
    assert engine.generate_calls == 0


def test_glm53_non_bool_enable_thinking_scoped(tmp_path):
    app, engine = _build_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions", json=_chat_body(
            chat_template_kwargs={"enable_thinking": "false"}))
    assert r.status_code == HTTPStatus.BAD_REQUEST
    err = r.json()["error"]
    assert err["param"] == "chat_template_kwargs.enable_thinking"
    assert "boolean" in err["message"]
    assert engine.generate_calls == 0


def test_glm53_non_bool_thinking_scoped(tmp_path):
    app, engine = _build_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions", json=_chat_body(
            chat_template_kwargs={"thinking": 0}))
    assert r.status_code == HTTPStatus.BAD_REQUEST
    err = r.json()["error"]
    assert err["param"] == "chat_template_kwargs.thinking"
    assert "boolean" in err["message"]
    assert engine.generate_calls == 0


# ---------------- GLM53 scope: valid requests (HTTP 200 + real parser) ------

def test_glm53_valid_chat_http200_reasoning_split(tmp_path):
    app, engine = _build_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions",
                   json=_chat_body(reasoning_effort="low"))
    assert r.status_code == HTTPStatus.OK, r.text
    body = r.json()
    msg = body["choices"][0]["message"]
    assert msg["reasoning"] == "thinking here"
    assert msg["content"] == "hello world"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] > 0
    assert engine.generate_calls == 1


def test_glm53_multi_turn_clear_thinking_http200(tmp_path):
    app, engine = _build_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions", json=_chat_body(
            messages=[{"role": "user", "content": "hi"},
                      {"role": "assistant", "content": "answer"},
                      {"role": "user", "content": "hello"}],
            reasoning_effort="low",
            chat_template_kwargs={"clear_thinking": True}))
    assert r.status_code == HTTPStatus.OK, r.text
    assert r.json()["choices"][0]["message"]["content"] == "hello world"
    assert engine.generate_calls == 1


def test_glm53_streaming_http200_reconstruction(tmp_path):
    # Multi-increment streaming: reasoning and answer in separate engine
    # outputs with the closing marker spanning chunks; final event carries
    # finish_reason and usage.
    app, engine = _build_app(tmp_path, stream_chunks=[
        "thinking ", "here" + CLOSE_M, "hello ", "world"],
        finish_reason="stop")
    with TestClient(app, raise_server_exceptions=False) as c:
        with c.stream("POST", "/v1/chat/completions",
                      json=_chat_body(stream=True)) as r:
            assert r.status_code == HTTPStatus.OK
            events = []
            for line in r.iter_lines():
                if line.startswith("data: "):
                    d = line[6:]
                    if d == "[DONE]":
                        break
                    events.append(json.loads(d))
    reasoning = "".join(
        e["choices"][0]["delta"].get("reasoning", "")
        for e in events if e.get("choices"))
    content = "".join(
        e["choices"][0]["delta"].get("content", "")
        for e in events if e.get("choices"))
    assert reasoning == "thinking here"
    assert content == "hello world"
    finishes = [e["choices"][0].get("finish_reason") for e in events
                if e.get("choices") and e["choices"][0].get("finish_reason")]
    assert finishes == ["stop"]
    assert engine.generate_calls == 1


def test_glm53_truncated_reasoning_finish_length(tmp_path):
    # Budget exhausted mid-reasoning: no closing marker, empty content, and
    # the honest finish_reason=length from the engine is preserved.
    # (nonstream consumes the final RequestOutput; single chunk fixture)
    app, engine = _build_app(
        tmp_path, output_text="still thinking and no close marker",
        finish_reason="length")
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions", json=_chat_body())
    assert r.status_code == HTTPStatus.OK, r.text
    body = r.json()
    msg = body["choices"][0]["message"]
    assert msg.get("reasoning") == "still thinking and no close marker"
    assert not msg.get("content")
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] > 0


def test_untrusted_custom_template_rejected_http400(tmp_path):
    # Negative control (risk check on error-path integrity): an invalid
    # custom template under the DEFAULT untrusted policy (server started
    # without --trust-request-chat-template) yields a proper 400, not a 500.
    app, engine = _build_app(tmp_path, trust_request_templates=False)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions", json=_chat_body(
            chat_template="{% invalid jinja",
            chat_template_kwargs={"enable_thinking": False}))
    assert r.status_code == HTTPStatus.BAD_REQUEST, r.text
    assert "error" in r.json()
    assert engine.generate_calls == 0


def test_trusted_invalid_custom_template_currently_500(tmp_path):
    # Documents current behavior under trust_request_chat_template=True: an
    # invalid template raises inside rendering and surfaces as 500. The new
    # validator does not intercept it (custom templates opt out of the
    # GLM53 profile). Recorded as a known risk for the PR, not asserted as
    # correct behavior.
    app, engine = _build_app(tmp_path)  # trust_request_templates=True default
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions", json=_chat_body(
            chat_template="{% invalid jinja"))
    assert r.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert engine.generate_calls == 0


# ---------------- other models / templates unchanged (HTTP 200) --------------

def test_plain_model_enable_thinking_false_http200(tmp_path):
    app, engine = _build_app(tmp_path, template=PLAIN_TEMPLATE,
                             model_type="other_model")
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions",
                   json=_chat_body(chat_template_kwargs={"enable_thinking": False}))
    assert r.status_code == HTTPStatus.OK, r.text
    assert engine.generate_calls == 1


def test_switchable_model_enable_thinking_false_http200(tmp_path):
    app, engine = _build_app(tmp_path, template=SWITCHABLE_TEMPLATE,
                             model_type="other_model")
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions",
                   json=_chat_body(chat_template_kwargs={"enable_thinking": False}))
    assert r.status_code == HTTPStatus.OK, r.text
    assert engine.generate_calls == 1


def test_glm53_custom_request_template_not_rejected(tmp_path):
    # A request-level custom template overrides the server default; validation
    # refers to the ACTUALLY resolved template (switchable one) -> allowed.
    app, engine = _build_app(tmp_path, template=GLM53_TEMPLATE)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions", json=_chat_body(
            chat_template=SWITCHABLE_TEMPLATE,
            chat_template_kwargs={"enable_thinking": False}))
    assert r.status_code == HTTPStatus.OK, r.text
    assert engine.generate_calls == 1


def test_glm53_custom_always_thinking_template_opts_out(tmp_path):
    # Documented boundary: a request-level custom template override OPTS OUT
    # of the GLM53 default-profile validation entirely (its capability is
    # not inferred from variable names), preserving generic behavior even
    # when the custom template looks always-thinking.
    custom = GLM53_TEMPLATE  # always-thinking by construction
    app, engine = _build_app(tmp_path, template=PLAIN_TEMPLATE)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions", json=_chat_body(
            chat_template=custom,
            chat_template_kwargs={"enable_thinking": False}))
    assert r.status_code == HTTPStatus.OK, r.text
    assert engine.generate_calls == 1


# ---------------- raw completions unchanged (HTTP, text + tokens) ------------

def test_raw_completions_text_prompt_http200(tmp_path):
    app, engine = _build_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/completions",
                   json={"model": MODEL_NAME, "prompt": "hello",
                         "max_tokens": 8, "temperature": 0.0})
    assert r.status_code == HTTPStatus.OK, r.text
    assert engine.generate_calls == 1


def test_raw_completions_token_prompt_http200(tmp_path):
    app, engine = _build_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/completions",
                   json={"model": MODEL_NAME, "prompt": [0, 1, 2],
                         "max_tokens": 8, "temperature": 0.0})
    assert r.status_code == HTTPStatus.OK, r.text
    assert engine.generate_calls == 1


# ---------------- DEFAULT DEPLOYMENT (model-provided template) ---------------
# Fixture: both constructor chat_template args are None; the GLM53 template
# lives in the tokenizer dir as the model's own default. This mirrors the
# real service which passes no --chat-template.

def _build_default_app(tmp_path, **kw):
    # DEFAULT deployment: the tokenizer carries the GLM53 template; BOTH the
    # OpenAIServingChat and OnlineRenderer constructor chat_template values
    # stay None (no --chat-template, no request override).
    kw.pop("template", None)
    kw.setdefault("model_type", "glm5_next")
    kw.setdefault("default_chat_template_kwargs", None)
    kw.setdefault("output_text", None)
    kw.setdefault("stream_chunks", None)
    kw.setdefault("finish_reason", "stop")
    kw.setdefault("trust_request_templates", True)
    kw["template"] = None  # constructor value: no explicit server template
    return _build_app(tmp_path, tokenizer_template=GLM53_TEMPLATE, **kw)


def test_default_deployment_enable_thinking_false_http400(tmp_path):
    app, engine = _build_default_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions",
                   json=_chat_body(chat_template_kwargs={"enable_thinking": False}))
    assert r.status_code == HTTPStatus.BAD_REQUEST, r.text
    assert r.json()["error"]["param"] == "chat_template_kwargs.enable_thinking"
    assert engine.generate_calls == 0


def test_default_deployment_thinking_false_http400(tmp_path):
    app, engine = _build_default_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions",
                   json=_chat_body(chat_template_kwargs={"thinking": False}))
    assert r.status_code == HTTPStatus.BAD_REQUEST, r.text
    assert r.json()["error"]["param"] == "chat_template_kwargs.thinking"
    assert engine.generate_calls == 0


def test_default_deployment_reasoning_effort_none_http400(tmp_path):
    app, engine = _build_default_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions",
                   json=_chat_body(reasoning_effort="none"))
    assert r.status_code == HTTPStatus.BAD_REQUEST, r.text
    assert r.json()["error"]["param"] == "reasoning_effort"
    assert engine.generate_calls == 0


@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_default_deployment_valid_effort_http200_parser_split(tmp_path, effort):
    app, engine = _build_default_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions",
                   json=_chat_body(reasoning_effort=effort))
    assert r.status_code == HTTPStatus.OK, r.text
    msg = r.json()["choices"][0]["message"]
    assert msg["reasoning"] == "thinking here"
    assert msg["content"] == "hello world"
    assert engine.generate_calls == 1


def test_default_deployment_valid_default_http200(tmp_path):
    app, engine = _build_default_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions", json=_chat_body())
    assert r.status_code == HTTPStatus.OK, r.text
    assert r.json()["choices"][0]["finish_reason"] == "stop"
    assert engine.generate_calls == 1


def test_default_deployment_stream_http200(tmp_path):
    app, engine = _build_default_app(tmp_path, stream_chunks=[
        "thinking ", "here" + CLOSE_M, "hello ", "world"],
        finish_reason="stop")
    with TestClient(app, raise_server_exceptions=False) as c:
        with c.stream("POST", "/v1/chat/completions",
                      json=_chat_body(stream=True)) as r:
            assert r.status_code == HTTPStatus.OK
            events = []
            for line in r.iter_lines():
                if line.startswith("data: "):
                    d = line[6:]
                    if d == "[DONE]":
                        break
                    events.append(json.loads(d))
    reasoning = "".join(
        e["choices"][0]["delta"].get("reasoning", "")
        for e in events if e.get("choices"))
    content = "".join(
        e["choices"][0]["delta"].get("content", "")
        for e in events if e.get("choices"))
    assert reasoning == "thinking here"
    assert content == "hello world"
    assert engine.generate_calls == 1


def test_default_deployment_custom_request_template_opts_out(tmp_path):
    # Request-level override still opts out even in default deployment.
    app, engine = _build_default_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions", json=_chat_body(
            chat_template=SWITCHABLE_TEMPLATE,
            chat_template_kwargs={"enable_thinking": False}))
    assert r.status_code == HTTPStatus.OK, r.text
    assert engine.generate_calls == 1
