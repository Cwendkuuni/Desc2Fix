#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from desc2fix.llm.registry import ModelSpec
from desc2fix.utils.config import get_required_env


class LLMClientError(RuntimeError):
    pass


class BaseLLMClient:
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def generate(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(BaseLLMClient):
    """
    Client for OpenAI and OpenAI-compatible APIs.

    Supports:
      - OpenAI GPT-4o
      - DeepSeek deepseek-chat
      - future OpenAI-compatible providers
    """

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)

        api_key = get_required_env(spec.api_key_env)

        if spec.base_url:
            self.client = OpenAI(api_key=api_key, base_url=spec.base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.spec.model_name,
                messages=messages,
                temperature=self.spec.temperature if temperature is None else temperature,
                top_p=self.spec.top_p if top_p is None else top_p,
                max_tokens=max_tokens or self.spec.reserved_response_tokens,
            )

            content = response.choices[0].message.content
            if content is None:
                raise LLMClientError(
                    f"Empty response from model {self.spec.key}/{self.spec.model_name}"
                )

            return content

        except Exception as e:
            raise LLMClientError(
                f"LLM call failed for {self.spec.key}/{self.spec.model_name}: {e}"
            ) from e


class RetryingLLMClient(BaseLLMClient):
    """
    Retry wrapper around any BaseLLMClient.
    """

    def __init__(
        self,
        inner: BaseLLMClient,
        *,
        max_retries: int = 5,
        initial_sleep: float = 2.0,
        backoff: float = 2.0,
    ):
        super().__init__(inner.spec)
        self.inner = inner
        self.max_retries = max_retries
        self.initial_sleep = initial_sleep
        self.backoff = backoff

    def generate(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        sleep_s = self.initial_sleep
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return self.inner.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                last_error = e
                if attempt == self.max_retries:
                    break

                print(
                    f"[WARN] LLM call failed "
                    f"({self.spec.key}, attempt {attempt}/{self.max_retries}): {e}"
                )
                print(f"[WARN] Sleeping {sleep_s:.1f}s before retry...")
                time.sleep(sleep_s)
                sleep_s *= self.backoff

        raise LLMClientError(
            f"LLM call failed after {self.max_retries} attempts "
            f"for {self.spec.key}: {last_error}"
        )


def make_llm_client(
    spec: ModelSpec,
    *,
    retry: bool = True,
    max_retries: int = 5,
) -> BaseLLMClient:
    """
    Factory for LLM clients.

    Supported provider values:
      - openai
      - openai_compatible
    """

    provider = spec.provider.lower().strip()

    if provider in {"openai", "openai_compatible", "deepseek"}:
        base_client: BaseLLMClient = OpenAICompatibleClient(spec)
    else:
        raise ValueError(
            f"Unsupported provider '{spec.provider}' for model '{spec.key}'. "
            f"Use provider=openai or provider=openai_compatible."
        )

    if retry:
        return RetryingLLMClient(base_client, max_retries=max_retries)

    return base_client
