"""HTTP adapters for live model endpoints.

Ollama, OpenAI-compatible servers (OpenAI, LiteLLM, LM Studio, vLLM, Ollama's
``/v1`` route), Anthropic, and Gemini are called with ``urllib``. API keys are
read from the environment and are never included in ``repr``, logs, or error text.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import socket
from typing import Any, Mapping
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import quote

from regression_tracker import ResponseTokenMetadata
from response_classifier import TurnRecord

OLLAMA_DEFAULT_BASE = "http://127.0.0.1:11434"
OPENAI_DEFAULT_BASE = "https://api.openai.com"
LITELLM_DEFAULT_BASE = "http://127.0.0.1:4000"
LMSTUDIO_DEFAULT_BASE = "http://127.0.0.1:1234"
ANTHROPIC_DEFAULT_BASE = "https://api.anthropic.com"
GEMINI_DEFAULT_BASE = "https://generativelanguage.googleapis.com"
ANTHROPIC_VERSION = "2023-06-01"

PROVIDER_CHOICES = (
    "ollama",
    "openai",
    "litellm",
    "lmstudio",
    "vllm",
    "ollama-openai",
    "anthropic",
    "gemini",
)


class ProviderKind:
    """Wire family used to build the request and read the response."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class LiveProviderError(RuntimeError):
    """A live provider call failed. The message is safe to store and classify."""

    def __init__(self, message: str, *, signal: str, status: int | None = None) -> None:
        super().__init__(message)
        self.signal = signal
        self.status = status

    def __repr__(self) -> str:
        return f"LiveProviderError(signal={self.signal!r}, status={self.status!r})"


@dataclass(frozen=True)
class LiveAgentSpec:
    """A configured provider endpoint. Holds the env-var name, never the key."""

    provider: str
    kind: str
    base_url: str
    model: str
    key_env: str | None
    architecture_family: str
    reachable: bool | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "kind": self.kind,
            "base_url": self.base_url,
            "model": self.model,
            "key_env": self.key_env,
            "architecture_family": self.architecture_family,
            "reachable": self.reachable,
        }


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int


def discover_live_agents(*, timeout: float = 0.4) -> tuple[LiveAgentSpec, ...]:
    """Return providers that are configured. Ollama is included only when ``/api/tags`` answers."""
    found: list[LiveAgentSpec] = []

    ollama_base = _base_from_env("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE)
    ollama_models = _ollama_model_names(ollama_base, timeout)
    if ollama_models is not None:
        preferred = os.environ.get("OLLAMA_MODEL", "").strip()
        model = preferred or (ollama_models[0] if ollama_models else "")
        found.append(
            LiveAgentSpec(
                provider="ollama",
                kind=ProviderKind.OLLAMA,
                base_url=ollama_base,
                model=model,
                key_env=None,
                architecture_family="ollama-native",
                reachable=True,
            )
        )
        found.append(
            LiveAgentSpec(
                provider="ollama-openai",
                kind=ProviderKind.OPENAI,
                base_url=ollama_base,
                model=model,
                key_env=None,
                architecture_family="openai-compatible",
                reachable=True,
            )
        )

    if _env("OPENAI_API_KEY"):
        found.append(
            LiveAgentSpec(
                provider="openai",
                kind=ProviderKind.OPENAI,
                base_url=_base_from_env("OPENAI_BASE_URL", OPENAI_DEFAULT_BASE),
                model=_env("OPENAI_MODEL") or "gpt-4o-mini",
                key_env="OPENAI_API_KEY",
                architecture_family="openai-compatible",
                reachable=None,
            )
        )

    if _env("LITELLM_API_KEY") or _env("LITELLM_BASE_URL"):
        found.append(
            LiveAgentSpec(
                provider="litellm",
                kind=ProviderKind.OPENAI,
                base_url=_base_from_env("LITELLM_BASE_URL", LITELLM_DEFAULT_BASE),
                model=_env("LITELLM_MODEL") or "gpt-4o-mini",
                key_env="LITELLM_API_KEY" if _env("LITELLM_API_KEY") else None,
                architecture_family="openai-compatible",
                reachable=None,
            )
        )

    lm_base = _base_from_env("LMSTUDIO_BASE_URL", LMSTUDIO_DEFAULT_BASE)
    lm_models = _openai_model_ids(lm_base, timeout)
    lm_model = _env("LMSTUDIO_MODEL") or (lm_models[0] if lm_models else "")
    found.append(
        LiveAgentSpec(
            provider="lmstudio",
            kind=ProviderKind.OPENAI,
            base_url=lm_base,
            model=lm_model,
            key_env="LMSTUDIO_API_KEY" if _env("LMSTUDIO_API_KEY") else None,
            architecture_family="openai-compatible",
            reachable=lm_models is not None,
        )
    )

    vllm_base = _env("VLLM_BASE_URL")
    if vllm_base:
        vllm_models = _openai_model_ids(vllm_base, timeout)
        found.append(
            LiveAgentSpec(
                provider="vllm",
                kind=ProviderKind.OPENAI,
                base_url=_strip_base(vllm_base),
                model=_env("VLLM_MODEL") or (vllm_models[0] if vllm_models else ""),
                key_env="VLLM_API_KEY" if _env("VLLM_API_KEY") else ("OPENAI_API_KEY" if _env("OPENAI_API_KEY") else None),
                architecture_family="openai-compatible",
                reachable=vllm_models is not None,
            )
        )

    if _env("ANTHROPIC_API_KEY"):
        found.append(
            LiveAgentSpec(
                provider="anthropic",
                kind=ProviderKind.ANTHROPIC,
                base_url=_base_from_env("ANTHROPIC_BASE_URL", ANTHROPIC_DEFAULT_BASE),
                model=_env("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest",
                key_env="ANTHROPIC_API_KEY",
                architecture_family="anthropic-messages",
                reachable=None,
            )
        )

    if _env("GEMINI_API_KEY"):
        found.append(
            LiveAgentSpec(
                provider="gemini",
                kind=ProviderKind.GEMINI,
                base_url=_base_from_env("GEMINI_BASE_URL", GEMINI_DEFAULT_BASE),
                model=_env("GEMINI_MODEL") or "gemini-2.0-flash",
                key_env="GEMINI_API_KEY",
                architecture_family="gemini-generate",
                reachable=None,
            )
        )

    return tuple(sorted(found, key=lambda spec: spec.provider))


def resolve_spec(provider: str, *, model: str | None = None, base_url: str | None = None) -> LiveAgentSpec:
    """Build one spec from a provider id, with optional model and base overrides."""
    if provider not in PROVIDER_CHOICES:
        raise LiveProviderError(
            f"Unknown provider '{provider}'. Expected one of: {', '.join(PROVIDER_CHOICES)}.",
            signal="provider_not_configured",
        )
    matches = [spec for spec in discover_live_agents() if spec.provider == provider]
    if matches:
        current = matches[0]
    else:
        current = _fallback_spec(provider)
    return LiveAgentSpec(
        provider=current.provider,
        kind=current.kind,
        base_url=_strip_base(base_url) if base_url else current.base_url,
        model=(model or current.model).strip(),
        key_env=current.key_env,
        architecture_family=current.architecture_family,
        reachable=current.reachable,
    )


class LiveHttpAgent:
    """TargetAgent that completes each user turn with a real HTTP provider call."""

    def __init__(
        self,
        spec: LiveAgentSpec,
        *,
        timeout: float = 60.0,
        max_output_tokens: int = 1024,
        api_key: str | None = None,
    ) -> None:
        if not spec.model.strip():
            raise LiveProviderError(
                f"Provider {spec.provider} needs a model name.",
                signal="provider_not_configured",
            )
        if timeout <= 0:
            raise LiveProviderError("timeout must be positive.", signal="provider_not_configured")
        if max_output_tokens < 1:
            raise LiveProviderError("max_output_tokens must be >= 1.", signal="provider_not_configured")
        self._spec = spec
        self._timeout = float(timeout)
        self._max_output_tokens = int(max_output_tokens)
        self._key_env = spec.key_env
        if api_key is not None:
            self._api_key: str | None = api_key
        elif spec.key_env:
            self._api_key = _env(spec.key_env)
        else:
            self._api_key = None
        if spec.key_env and not self._api_key:
            raise LiveProviderError(
                f"Set {spec.key_env} before calling provider {spec.provider}.",
                signal="provider_not_configured",
            )

    @property
    def model_id(self) -> str:
        return self._spec.model

    @property
    def model_version(self) -> str:
        return self._spec.provider

    @property
    def architecture_family(self) -> str:
        return self._spec.architecture_family

    @property
    def provider(self) -> str:
        return self._spec.provider

    @property
    def base_url(self) -> str:
        return self._spec.base_url

    def describe_transport(self) -> dict[str, object]:
        """JSON-safe description. The API key is not included."""
        return {
            "distribution": "live-http",
            "provider": self._spec.provider,
            "kind": self._spec.kind,
            "base_url": self._spec.base_url,
            "key_env": self._key_env,
            "timeout_s": self._timeout,
            "max_output_tokens": self._max_output_tokens,
        }

    def __repr__(self) -> str:
        return (
            f"LiveHttpAgent(provider={self._spec.provider!r}, model_id={self.model_id!r}, "
            f"base_url={self.base_url!r}, key_env={self._key_env!r})"
        )

    def run_case(self, test_case: Any) -> Any:
        from eval_runner import AgentResponse

        if not test_case.user_turns:
            raise LiveProviderError(
                f"Test case {test_case.test_case_id} has no user turns.",
                signal="provider_bad_response",
            )
        assistant_texts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        saw_usage = False
        for index, user_turn in enumerate(test_case.user_turns):
            turns = _case_turns(
                system_boundary=test_case.system_boundary,
                user_turns=test_case.user_turns[: index + 1],
                assistant_texts=assistant_texts,
            )
            text, usage = self.complete(turns)
            assistant_texts.append(text)
            if usage is not None:
                saw_usage = True
                prompt_tokens += usage.prompt_tokens
                completion_tokens += usage.completion_tokens
        visible_tokens = tuple(
            token for text in assistant_texts for token in text.split()
        )
        if saw_usage:
            token_meta = ResponseTokenMetadata(
                tokens=visible_tokens,
                prompt_token_count=prompt_tokens,
                completion_token_count=completion_tokens,
                tokenizer_name=f"{self._spec.provider}-usage",
            )
        else:
            token_meta = ResponseTokenMetadata(
                tokens=visible_tokens,
                prompt_token_count=None,
                completion_token_count=len(visible_tokens),
                tokenizer_name="whitespace",
            )
        assistant_turns = tuple(
            TurnRecord(turn_index=index, role="assistant", content=text)
            for index, text in enumerate(assistant_texts)
        )
        return AgentResponse(assistant_turns=assistant_turns, response_tokens=token_meta)

    def complete(self, turns: list[TurnRecord]) -> tuple[str, TokenUsage | None]:
        """Send one non-streaming chat request and return assistant text plus usage."""
        url = self._endpoint()
        payload = self._payload(turns)
        body = _request_json(
            "POST",
            url,
            payload,
            self._headers(),
            self._timeout,
            self._api_key,
        )
        if not isinstance(body, dict):
            raise LiveProviderError(
                f"Provider {self._spec.provider} returned a non-object response.",
                signal="provider_bad_response",
            )
        return self._parse_response(body)

    def _endpoint(self) -> str:
        base = self._spec.base_url.rstrip("/")
        kind = self._spec.kind
        if kind == ProviderKind.OLLAMA:
            return f"{base}/api/chat"
        if kind == ProviderKind.OPENAI:
            return f"{base}/v1/chat/completions"
        if kind == ProviderKind.ANTHROPIC:
            return f"{base}/v1/messages"
        if kind == ProviderKind.GEMINI:
            model = quote(self._spec.model, safe="")
            return f"{base}/v1beta/models/{model}:generateContent"
        raise LiveProviderError(
            f"Unsupported provider kind {kind!r}.",
            signal="provider_not_configured",
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        key = self._api_key
        if not key:
            return headers
        if self._spec.kind == ProviderKind.ANTHROPIC:
            headers["x-api-key"] = key
            headers["anthropic-version"] = ANTHROPIC_VERSION
        elif self._spec.kind == ProviderKind.GEMINI:
            headers["x-goog-api-key"] = key
        else:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _payload(self, turns: list[TurnRecord]) -> dict[str, Any]:
        kind = self._spec.kind
        if kind == ProviderKind.OLLAMA:
            return {
                "model": self._spec.model,
                "messages": _ollama_messages(turns),
                "stream": False,
                "options": {"num_predict": self._max_output_tokens},
            }
        if kind == ProviderKind.OPENAI:
            return {
                "model": self._spec.model,
                "messages": _openai_messages(turns),
                "stream": False,
                "max_tokens": self._max_output_tokens,
            }
        if kind == ProviderKind.ANTHROPIC:
            system_text, messages = _anthropic_messages(turns)
            payload: dict[str, Any] = {
                "model": self._spec.model,
                "max_tokens": self._max_output_tokens,
                "messages": messages,
            }
            if system_text:
                payload["system"] = system_text
            return payload
        if kind == ProviderKind.GEMINI:
            system_text, contents = _gemini_contents(turns)
            payload = {
                "contents": contents,
                "generationConfig": {"maxOutputTokens": self._max_output_tokens},
            }
            if system_text:
                payload["systemInstruction"] = {"parts": [{"text": system_text}]}
            return payload
        raise LiveProviderError(
            f"Unsupported provider kind {kind!r}.",
            signal="provider_not_configured",
        )

    def _parse_response(self, body: dict[str, Any]) -> tuple[str, TokenUsage | None]:
        kind = self._spec.kind
        if kind == ProviderKind.OLLAMA:
            return _parse_ollama(body, self._spec.provider)
        if kind == ProviderKind.OPENAI:
            return _parse_openai(body, self._spec.provider)
        if kind == ProviderKind.ANTHROPIC:
            return _parse_anthropic(body, self._spec.provider)
        if kind == ProviderKind.GEMINI:
            return _parse_gemini(body, self._spec.provider)
        raise LiveProviderError(
            f"Unsupported provider kind {kind!r}.",
            signal="provider_not_configured",
        )


def _case_turns(
    *,
    system_boundary: str,
    user_turns: tuple[str, ...] | list[str],
    assistant_texts: list[str],
) -> list[TurnRecord]:
    turns = [TurnRecord(0, "system", system_boundary)]
    for index, user_turn in enumerate(user_turns):
        turns.append(TurnRecord(len(turns), "user", user_turn))
        if index < len(assistant_texts):
            turns.append(TurnRecord(len(turns), "assistant", assistant_texts[index]))
    return turns


def _ollama_messages(turns: list[TurnRecord]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in turns:
        role = _require_role(turn.role)
        messages.append({"role": role, "content": turn.content})
    return messages


def _openai_messages(turns: list[TurnRecord]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for turn in turns:
        role = _require_role(turn.role)
        if role == "tool" and turn.tool_call_id:
            _ensure_openai_tool_call(messages, turn)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": turn.tool_call_id,
                    "content": turn.content,
                }
            )
        elif role == "tool":
            messages.append({"role": "user", "content": f"Tool observation:\n{turn.content}"})
        elif role == "assistant" and turn.tool_call_id and turn.tool_name:
            messages.append(
                {
                    "role": "assistant",
                    "content": turn.content or None,
                    "tool_calls": [_openai_tool_call(turn)],
                }
            )
        else:
            messages.append({"role": role, "content": turn.content})
    return messages


def _openai_tool_call(turn: TurnRecord) -> dict[str, Any]:
    return {
        "id": turn.tool_call_id,
        "type": "function",
        "function": {"name": turn.tool_name or "tool", "arguments": "{}"},
    }


def _ensure_openai_tool_call(messages: list[dict[str, Any]], turn: TurnRecord) -> None:
    call = _openai_tool_call(turn)
    if messages and messages[-1].get("role") == "assistant":
        calls = messages[-1].setdefault("tool_calls", [])
        if not any(item.get("id") == turn.tool_call_id for item in calls):
            calls.append(call)
        return
    messages.append({"role": "assistant", "content": None, "tool_calls": [call]})


def _anthropic_messages(turns: list[TurnRecord]) -> tuple[str, list[dict[str, Any]]]:
    system_parts = [turn.content.strip() for turn in turns if turn.role == "system" and turn.content.strip()]
    messages: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        role = _require_role(turn.role)
        if role == "system":
            continue
        if role == "tool" and turn.tool_call_id:
            _ensure_anthropic_tool_use(messages, turn)
            block = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": turn.tool_call_id,
                        "content": turn.content,
                    }
                ],
            }
            _append_anthropic(messages, block)
            continue
        if role == "tool":
            block = {
                "role": "user",
                "content": [{"type": "text", "text": f"Tool observation:\n{turn.content}"}],
            }
            _append_anthropic(messages, block)
            continue
        anthropic_role = "assistant" if role == "assistant" else "user"
        block = {"role": anthropic_role, "content": [{"type": "text", "text": turn.content}]}
        _append_anthropic(messages, block)
    if messages and messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": [{"type": "text", "text": "(continue)"}]})
    if not messages:
        raise LiveProviderError(
            "Anthropic request has no user or assistant turns.",
            signal="provider_bad_response",
        )
    return "\n\n".join(system_parts), messages


def _ensure_anthropic_tool_use(messages: list[dict[str, Any]], turn: TurnRecord) -> None:
    use = {
        "type": "tool_use",
        "id": turn.tool_call_id,
        "name": turn.tool_name or "tool",
        "input": {},
    }
    if messages and messages[-1]["role"] == "assistant":
        content = messages[-1]["content"]
        if not any(isinstance(block, dict) and block.get("id") == turn.tool_call_id for block in content):
            content.append(use)
        return
    messages.append({"role": "assistant", "content": [use]})


def _append_anthropic(messages: list[dict[str, Any]], block: dict[str, Any]) -> None:
    if messages and messages[-1]["role"] == block["role"]:
        messages[-1]["content"].extend(block["content"])
        return
    messages.append(block)


def _gemini_contents(turns: list[TurnRecord]) -> tuple[str, list[dict[str, Any]]]:
    system_parts = [turn.content.strip() for turn in turns if turn.role == "system" and turn.content.strip()]
    contents: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        role = _require_role(turn.role)
        if role == "system":
            continue
        if role == "assistant":
            part = {"text": turn.content}
            gemini_role = "model"
        elif role == "tool" and (turn.tool_call_id or turn.tool_name):
            part = {
                "functionResponse": {
                    "name": turn.tool_name or "tool",
                    "response": {"content": turn.content},
                }
            }
            gemini_role = "user"
        elif role == "tool":
            part = {"text": f"Tool observation:\n{turn.content}"}
            gemini_role = "user"
        else:
            part = {"text": turn.content}
            gemini_role = "user"
        if contents and contents[-1]["role"] == gemini_role:
            contents[-1]["parts"].append(part)
        else:
            contents.append({"role": gemini_role, "parts": [part]})
    if not contents:
        raise LiveProviderError(
            "Gemini request has no user or assistant turns.",
            signal="provider_bad_response",
        )
    return "\n\n".join(system_parts), contents


def _require_role(role: str) -> str:
    if role not in {"system", "user", "assistant", "tool"}:
        raise LiveProviderError(
            f"Unsupported turn role {role!r}. Expected system, user, assistant, or tool.",
            signal="provider_bad_response",
        )
    return role


def _parse_ollama(body: dict[str, Any], provider: str) -> tuple[str, TokenUsage | None]:
    message = body.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        text = message["content"]
    else:
        raise LiveProviderError(
            f"Provider {provider} response is missing message.content.",
            signal="provider_bad_response",
        )
    usage = _usage_pair(body.get("prompt_eval_count"), body.get("eval_count"))
    return text, usage


def _parse_openai(body: dict[str, Any], provider: str) -> tuple[str, TokenUsage | None]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LiveProviderError(
            f"Provider {provider} response is missing choices.",
            signal="provider_bad_response",
        )
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise LiveProviderError(
            f"Provider {provider} response is missing choices[0].message.",
            signal="provider_bad_response",
        )
    text = _openai_message_text(message)
    usage_block = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    usage = _usage_pair(usage_block.get("prompt_tokens"), usage_block.get("completion_tokens"))
    return text, usage


def _openai_message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    parts: list[str] = []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = function.get("name")
            raw_args = function.get("arguments")
            arguments: Any = {}
            if isinstance(raw_args, str) and raw_args.strip():
                try:
                    arguments = json.loads(raw_args)
                except json.JSONDecodeError:
                    arguments = {"raw": raw_args}
            elif isinstance(raw_args, dict):
                arguments = raw_args
            if isinstance(name, str):
                parts.append(json.dumps({"tool": name, "arguments": arguments}, sort_keys=True))
    text = "\n".join(part for part in parts if part).strip()
    if not text:
        raise LiveProviderError(
            "OpenAI-compatible response did not include assistant text or tool calls.",
            signal="provider_bad_response",
        )
    return text


def _parse_anthropic(body: dict[str, Any], provider: str) -> tuple[str, TokenUsage | None]:
    blocks = body.get("content")
    if not isinstance(blocks, list):
        raise LiveProviderError(
            f"Provider {provider} response is missing content blocks.",
            signal="provider_bad_response",
        )
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif block.get("type") == "tool_use" and isinstance(block.get("name"), str):
            arguments = block.get("input") if isinstance(block.get("input"), dict) else {}
            parts.append(json.dumps({"tool": block["name"], "arguments": arguments}, sort_keys=True))
    text = "\n".join(part for part in parts if part).strip()
    if not text:
        raise LiveProviderError(
            f"Provider {provider} response did not include text or tool_use blocks.",
            signal="provider_bad_response",
        )
    usage_block = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    usage = _usage_pair(usage_block.get("input_tokens"), usage_block.get("output_tokens"))
    return text, usage


def _parse_gemini(body: dict[str, Any], provider: str) -> tuple[str, TokenUsage | None]:
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        raise LiveProviderError(
            f"Provider {provider} response is missing candidates.",
            signal="provider_bad_response",
        )
    content = candidates[0].get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        raise LiveProviderError(
            f"Provider {provider} response is missing candidate content parts.",
            signal="provider_bad_response",
        )
    texts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("text"), str):
            texts.append(part["text"])
        function_call = part.get("functionCall")
        if isinstance(function_call, dict) and isinstance(function_call.get("name"), str):
            arguments = function_call.get("args") if isinstance(function_call.get("args"), dict) else {}
            texts.append(json.dumps({"tool": function_call["name"], "arguments": arguments}, sort_keys=True))
    text = "\n".join(item for item in texts if item).strip()
    if not text:
        raise LiveProviderError(
            f"Provider {provider} response did not include text or functionCall parts.",
            signal="provider_bad_response",
        )
    usage_block = body.get("usageMetadata") if isinstance(body.get("usageMetadata"), dict) else {}
    usage = _usage_pair(usage_block.get("promptTokenCount"), usage_block.get("candidatesTokenCount"))
    return text, usage


def _usage_pair(prompt: Any, completion: Any) -> TokenUsage | None:
    if isinstance(prompt, bool) or isinstance(completion, bool):
        return None
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return None
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion)


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    headers: Mapping[str, str],
    timeout: float,
    secret: str | None,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=data, headers=dict(headers), method=method)
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:
            raw = response.read()
    except urlerror.HTTPError as exc:
        detail = _read_error_body(exc, secret)
        raise LiveProviderError(
            f"HTTP {exc.code} from {method} {_safe_url(url)}: {detail}",
            signal="provider_http_error",
            status=int(exc.code),
        ) from None
    except (TimeoutError, socket.timeout):
        raise LiveProviderError(
            f"Timed out during {method} {_safe_url(url)}.",
            signal="provider_timeout",
        ) from None
    except urlerror.URLError as exc:
        reason = exc.reason
        signal = "provider_timeout" if isinstance(reason, (TimeoutError, socket.timeout)) else "provider_http_error"
        reason_name = type(reason).__name__ if reason is not None else "URLError"
        raise LiveProviderError(
            f"Request failed during {method} {_safe_url(url)}: {reason_name}.",
            signal=signal,
        ) from None
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LiveProviderError(
            f"Provider response from {_safe_url(url)} was not UTF-8.",
            signal="provider_bad_response",
        ) from exc
    try:
        return json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise LiveProviderError(
            f"Provider response from {_safe_url(url)} was not JSON.",
            signal="provider_bad_response",
        ) from exc


def _read_error_body(exc: urlerror.HTTPError, secret: str | None) -> str:
    try:
        raw = exc.read()
    except Exception:
        return ""
    text = raw.decode("utf-8", errors="replace")
    return _scrub(text, secret)[:300]


def _scrub(text: str, secret: str | None) -> str:
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return text


def _safe_url(url: str) -> str:
    return url.split("?", 1)[0]


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _base_from_env(name: str, default: str) -> str:
    return _strip_base(os.environ.get(name, "").strip() or default)


def _strip_base(url: str) -> str:
    text = url.strip().rstrip("/")
    if text.endswith("/v1"):
        text = text[: -len("/v1")]
    return text


def _probe_json(url: str, timeout: float) -> Any | None:
    req = urlrequest.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:
            if response.status < 200 or response.status >= 300:
                return None
            raw = response.read()
    except Exception:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _ollama_model_names(base: str, timeout: float) -> list[str] | None:
    payload = _probe_json(f"{base.rstrip('/')}/api/tags", timeout)
    if not isinstance(payload, dict):
        return None
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    names: list[str] = []
    for item in models:
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"].strip():
            names.append(item["name"].strip())
    return names


def _openai_model_ids(base: str, timeout: float) -> list[str] | None:
    payload = _probe_json(f"{base.rstrip('/')}/v1/models", timeout)
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    names: list[str] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip():
            names.append(item["id"].strip())
    return names


def _fallback_spec(provider: str) -> LiveAgentSpec:
    if provider == "ollama":
        return LiveAgentSpec(
            "ollama",
            ProviderKind.OLLAMA,
            _base_from_env("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE),
            _env("OLLAMA_MODEL") or "",
            None,
            "ollama-native",
            False,
        )
    if provider == "ollama-openai":
        return LiveAgentSpec(
            "ollama-openai",
            ProviderKind.OPENAI,
            _base_from_env("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE),
            _env("OLLAMA_MODEL") or "",
            None,
            "openai-compatible",
            False,
        )
    if provider == "openai":
        return LiveAgentSpec(
            "openai",
            ProviderKind.OPENAI,
            _base_from_env("OPENAI_BASE_URL", OPENAI_DEFAULT_BASE),
            _env("OPENAI_MODEL") or "gpt-4o-mini",
            "OPENAI_API_KEY",
            "openai-compatible",
            None,
        )
    if provider == "litellm":
        return LiveAgentSpec(
            "litellm",
            ProviderKind.OPENAI,
            _base_from_env("LITELLM_BASE_URL", LITELLM_DEFAULT_BASE),
            _env("LITELLM_MODEL") or "gpt-4o-mini",
            "LITELLM_API_KEY" if _env("LITELLM_API_KEY") else None,
            "openai-compatible",
            None,
        )
    if provider == "lmstudio":
        return LiveAgentSpec(
            "lmstudio",
            ProviderKind.OPENAI,
            _base_from_env("LMSTUDIO_BASE_URL", LMSTUDIO_DEFAULT_BASE),
            _env("LMSTUDIO_MODEL") or "",
            "LMSTUDIO_API_KEY" if _env("LMSTUDIO_API_KEY") else None,
            "openai-compatible",
            False,
        )
    if provider == "vllm":
        base = _env("VLLM_BASE_URL") or ""
        return LiveAgentSpec(
            "vllm",
            ProviderKind.OPENAI,
            _strip_base(base) if base else "",
            _env("VLLM_MODEL") or "",
            "VLLM_API_KEY" if _env("VLLM_API_KEY") else ("OPENAI_API_KEY" if _env("OPENAI_API_KEY") else None),
            "openai-compatible",
            None,
        )
    if provider == "anthropic":
        return LiveAgentSpec(
            "anthropic",
            ProviderKind.ANTHROPIC,
            _base_from_env("ANTHROPIC_BASE_URL", ANTHROPIC_DEFAULT_BASE),
            _env("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest",
            "ANTHROPIC_API_KEY",
            "anthropic-messages",
            None,
        )
    return LiveAgentSpec(
        "gemini",
        ProviderKind.GEMINI,
        _base_from_env("GEMINI_BASE_URL", GEMINI_DEFAULT_BASE),
        _env("GEMINI_MODEL") or "gemini-2.0-flash",
        "GEMINI_API_KEY",
        "gemini-generate",
        None,
    )
