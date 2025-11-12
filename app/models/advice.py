from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

from pydantic import BaseModel


class AdviceKind(str, Enum):
    BOOK = "book"
    MOVIE = "movie"
    MUSIC = "music"
    YOUTUBE_VIDEO = "youtube_video"
    ARTICLE = "article"
    HABIT = "habit"
    ADVICE = "advice"
    CONCEPT = "concept"
    PSYCHOTHERAPY = "psychotherapy"
    PODCAST = "podcast"
    QUOTE = "quote",
    PERSON = "person"


@dataclass(frozen=True)
class UserIdentifier:
    user_id: Optional[str] = None
    auth_token: Optional[str] = None

    def is_empty(self) -> bool:
        return not (self.user_id or self.auth_token)


@dataclass(frozen=True)
class Advice:
    name: str
    kind: AdviceKind
    description: str
    link_url: Optional[str] = None
    image_url: Optional[str] = None
    author: Optional[str] = None
    categories: Sequence[str] = ()


@dataclass(frozen=True)
class AdviceRequestContext:
    user_identifier: UserIdentifier
    user_message: str


@dataclass(frozen=True)
class AdviceRecommendation:
    advice: Advice
    chat_response: str


class AdviceDetailsResponse(BaseModel):
    name: str
    kind: AdviceKind
    description: str
    link_url: Optional[str] = None
    image_url: Optional[str] = None
    author: Optional[str] = None

    @classmethod
    def from_domain(cls, advice: Advice) -> "AdviceDetailsResponse":
        return cls(
            name=advice.name,
            kind=advice.kind,
            description=advice.description,
            link_url=advice.link_url,
            image_url=advice.image_url,
            author=advice.author,
        )


class AdviceResponsePayload(BaseModel):
    advice: AdviceDetailsResponse
    chat_response: str

    @classmethod
    def from_recommendation(
        cls,
        recommendation: AdviceRecommendation,
    ) -> "AdviceResponsePayload":
        return cls(
            advice=AdviceDetailsResponse.from_domain(recommendation.advice),
            chat_response=recommendation.chat_response,
        )
