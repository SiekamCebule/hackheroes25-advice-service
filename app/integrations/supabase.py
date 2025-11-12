from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final, cast

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    # type: ignore[import]
    from supabase.client import AsyncClient as AsyncClientProtocol
else:
    AsyncClientProtocol = Any  # type: ignore[misc]

try:  # pragma: no cover - optional dependency hint
    # type: ignore[import]
    from supabase.client import AsyncClient as RuntimeAsyncClient
    # type: ignore[import]
    from supabase.lib.client_options import AsyncClientOptions
except ImportError as import_error:  # pragma: no cover - allow informative failure
    RuntimeAsyncClient = None  # type: ignore[assignment]
    AsyncClientOptions = None  # type: ignore[assignment]
    _IMPORT_ERROR: Exception | None = import_error
else:
    _IMPORT_ERROR = None

_CLIENT_INFO_HEADER: Final[dict[str, str]] = {
    "X-Client-Info": "hackheroes25-advice-service"
}


@dataclass(frozen=True)
class SupabaseSettings:
    url: str
    service_role_key: str

    @classmethod
    def from_env(cls) -> "SupabaseSettings":
        missing = []
        url = os.getenv("SUPABASE_URL")
        if not url:
            missing.append("SUPABASE_URL")

        service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not service_role_key:
            missing.append("SUPABASE_SERVICE_ROLE_KEY")

        if missing:
            env_list = ", ".join(missing)
            raise RuntimeError(
                f"Missing required Supabase environment variables: {env_list}"
            )

        return cls(
            url=cast(str, url),
            service_role_key=cast(str, service_role_key),
        )


@lru_cache(maxsize=1)
def get_supabase_settings() -> SupabaseSettings:
    return SupabaseSettings.from_env()


def create_supabase_async_client(
    settings: SupabaseSettings | None = None,
) -> AsyncClientProtocol:
    if RuntimeAsyncClient is None or AsyncClientOptions is None:
        raise RuntimeError(
            "Supabase client library is not installed. "
            "Install it with `pip install supabase`."
        ) from _IMPORT_ERROR

    settings = settings or get_supabase_settings()
    options = AsyncClientOptions(
        headers=_CLIENT_INFO_HEADER,
        auto_refresh_token=False,
        persist_session=False,
    )
    client = RuntimeAsyncClient(
        supabase_url=settings.url,
        supabase_key=settings.service_role_key,
        options=options,
    )
    return cast(AsyncClientProtocol, client)
