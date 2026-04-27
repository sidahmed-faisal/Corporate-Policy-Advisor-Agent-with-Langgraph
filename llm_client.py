"""
llm_client.py — Factory that returns a LangChain ChatModel based on LLM_PROVIDER env var.
Swap providers by changing a single environment variable.
"""
from __future__ import annotations

from functools import lru_cache
from langchain_core.language_models import BaseChatModel

import config


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.0) -> BaseChatModel:
    provider = config.LLM_PROVIDER

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=config.GEMINI_API_KEY,
            temperature=temperature,
            convert_system_message_to_human=True,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o",
            openai_api_key=config.OPENAI_API_KEY,
            temperature=temperature,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-3-5-haiku-20241022",
            anthropic_api_key=config.ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER='{provider}'. Choose: gemini | openai | anthropic"
    )
