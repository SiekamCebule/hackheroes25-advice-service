from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from openai import AsyncOpenAI

DEFAULT_MODEL: Final[str] = "text-embedding-3-small"


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str
    organization: str | None = None
    project: str | None = None
    embeddings_model: str = DEFAULT_MODEL

    @classmethod
    def from_env(cls) -> "OpenAISettings":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing required environment variable OPENAI_API_KEY.")

        organization = os.getenv("OPENAI_ORGANIZATION")
        project = os.getenv("OPENAI_PROJECT")
        embeddings_model = os.getenv(
            "OPENAI_EMBEDDINGS_MODEL") or DEFAULT_MODEL

        return cls(
            api_key=api_key,
            organization=organization,
            project=project,
            embeddings_model=embeddings_model,
        )


@lru_cache(maxsize=1)
def get_openai_settings() -> OpenAISettings:
    return OpenAISettings.from_env()


def create_async_openai_client(settings: OpenAISettings | None = None) -> AsyncOpenAI:
    settings = settings or get_openai_settings()
    return AsyncOpenAI(
        api_key=settings.api_key,
        organization=settings.organization,
        project=settings.project,
    )
