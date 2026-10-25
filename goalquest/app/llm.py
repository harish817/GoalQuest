"""LLM clients and protocol.

Supports:
 - ClaudeLLM (Anthropic SDK)
 - GroqLLM (Groq / OpenAI-compatible REST API via httpx)
 - create_llm() factory that selects based on settings and available API keys.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Protocol, TypeVar

import anthropic
import httpx
from pydantic import BaseModel, ValidationError

from .config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    def structured(self, system: str, user: str, schema: type[T], retries: int = 2) -> T: ...
    def text(self, system: str, user: str, max_tokens: int = 300) -> str: ...
    def run_tool_loop(
        self,
        system: str,
        user: str,
        tools: list[dict],
        handlers: dict[str, Callable[..., Any]],
        max_turns: int = 8,
    ) -> str: ...


class ClaudeLLM:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str | None = None):
        if client:
            self.client = client
        elif settings.anthropic_api_key:
            self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        else:
            self.client = anthropic.Anthropic()
        self.model = model or settings.model

    # -- structured output via forced tool use --
    def structured(self, system: str, user: str, schema: type[T], retries: int = 2) -> T:
        tool = {
            "name": "emit",
            "description": f"Return the result as a {schema.__name__}.",
            "input_schema": schema.model_json_schema(),
        }
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        last_err: Exception | None = None
        for _ in range(retries + 1):
            resp = self.client.messages.create(
                model=self.model, max_tokens=8000, system=system, messages=messages,
                tools=[tool], tool_choice={"type": "tool", "name": "emit"},
            )
            block = next(b for b in resp.content if b.type == "tool_use")
            try:
                return schema.model_validate(block.input)
            except ValidationError as e:
                last_err = e
                messages += [
                    {"role": "assistant", "content": resp.content},
                    {"role": "user", "content": [{
                        "type": "tool_result", "tool_use_id": block.id, "is_error": True,
                        "content": f"Validation failed, fix and re-emit:\n{e}"}]},
                ]
        raise RuntimeError(f"LLM output failed validation after retries: {last_err}")

    def text(self, system: str, user: str, max_tokens: int = 300) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    # -- agentic tool loop --
    def run_tool_loop(self, system: str, user: str, tools: list[dict],
                      handlers: dict[str, Callable[..., Any]], max_turns: int = 8) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        for _ in range(max_turns):
            resp = self.client.messages.create(
                model=self.model, max_tokens=1500, system=system, messages=messages, tools=tools)
            if resp.stop_reason != "tool_use":
                return "".join(b.text for b in resp.content if b.type == "text").strip()
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for b in resp.content:
                if b.type != "tool_use":
                    continue
                try:
                    out = handlers[b.name](**b.input)
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": json.dumps(out, default=str)})
                except Exception as e:
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "is_error": True, "content": str(e)})
            messages.append({"role": "user", "content": results})
        return "Stopped: tool loop exceeded max turns."


class GroqLLM:
    """LLM client for Groq or OpenAI-compatible chat endpoints using httpx."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: float = 60.0,
    ):
        self.api_key = api_key or settings.groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or settings.model
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    FALLBACK_MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, payload: dict) -> httpx.Response:
        resp = self.client.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=payload)
        if resp.status_code == 404 and "model_not_found" in resp.text:
            for fallback in self.FALLBACK_MODELS:
                if fallback != payload.get("model"):
                    logger.warning(
                        "Model '%s' not found on Groq. Automatically switching to '%s'.",
                        payload.get("model"),
                        fallback,
                    )
                    payload["model"] = fallback
                    self.model = fallback
                    resp = self.client.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=payload)
                    if resp.status_code == 200:
                        break
        if not resp.is_success:
            logger.error("Groq API error (%s): %s", resp.status_code, resp.text)
        resp.raise_for_status()
        return resp

    def text(self, system: str, user: str, max_tokens: int = 300) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }
        resp = self._post(payload)
        choice = resp.json()["choices"][0]
        return (choice["message"].get("content") or "").strip()

    def structured(self, system: str, user: str, schema: type[T], retries: int = 2) -> T:
        tool = {
            "type": "function",
            "function": {
                "name": "emit",
                "description": f"Return the result as a {schema.__name__}.",
                "parameters": schema.model_json_schema(),
            },
        }
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_err: Exception | None = None
        for _ in range(retries + 1):
            payload = {
                "model": self.model,
                "messages": messages,
                "tools": [tool],
                "tool_choice": {"type": "function", "function": {"name": "emit"}},
                "max_tokens": 4096,
            }
            resp = self._post(payload)
            choice = resp.json()["choices"][0]
            msg = choice["message"]
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                call = tool_calls[0]
                raw_args = call["function"]["arguments"]
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    return schema.model_validate(args)
                except (json.JSONDecodeError, ValidationError) as e:
                    last_err = e
                    messages.append(msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": f"Validation failed, fix and re-emit:\n{e}",
                    })
            else:
                content = (msg.get("content") or "").strip()
                try:
                    args = json.loads(content)
                    return schema.model_validate(args)
                except Exception as e:
                    last_err = e
                    messages.append(msg)
                    messages.append({
                        "role": "user",
                        "content": f"Failed to output valid schema {schema.__name__}: {e}",
                    })
        raise RuntimeError(f"LLM output failed validation after retries: {last_err}")

    def run_tool_loop(
        self,
        system: str,
        user: str,
        tools: list[dict],
        handlers: dict[str, Callable[..., Any]],
        max_turns: int = 8,
    ) -> str:
        openai_tools = []
        for t in tools:
            if "input_schema" in t:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t["input_schema"],
                    },
                })
            else:
                openai_tools.append(t)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        for _ in range(max_turns):
            payload = {
                "model": self.model,
                "messages": messages,
                "tools": openai_tools,
                "max_tokens": 1500,
            }
            resp = self._post(payload)
            choice = resp.json()["choices"][0]
            msg = choice["message"]
            messages.append(msg)
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                return (msg.get("content") or "").strip()
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                raw_args = tc["function"].get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    out = handlers[fn_name](**args)
                    content = json.dumps(out, default=str)
                except Exception as e:
                    content = str(e)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": content,
                })
        return "Stopped: tool loop exceeded max turns."


def create_llm(model: str | None = None) -> LLM:
    """Factory to instantiate the appropriate LLM client based on configuration."""
    target_model = model or settings.model
    groq_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
    anthropic_key = settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

    if target_model.startswith("claude") and anthropic_key:
        return ClaudeLLM(model=target_model)
    if groq_key:
        return GroqLLM(model=target_model, api_key=groq_key)
    if anthropic_key:
        return ClaudeLLM(model=target_model)
    if "llama" in target_model.lower() or "mixtral" in target_model.lower():
        return GroqLLM(model=target_model, api_key=groq_key)
    return ClaudeLLM(model=target_model)
