from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field, constr, field_validator


UserId = constr(strip_whitespace=True, min_length=1)


PSYCHO_TRAITS = (
    "ekstrawersja",
    "ugodowość",
    "sumienność",
    "stabilność_emocjonalna",
    "kreatywność",
    "myślenie_logiczne",
    "koncentracja",
    "przywództwo",
)

VOCATION_TRAITS = (
    "majstrowanie",
    "kontakt_z_natura",
    "obsluga_komputera",
    "zarzadzanie_projektem",
    "programowanie",
    "biologia_medycyna",
    "sztuka_design",
    "analiza_danych",
    "jezyki_obce",
    "praca_terenowa",
    "praca_zdalna",
    "wystapienia_publiczne",
    "pisanie",
    "priorytet_pieniadze",
    "priorytet_rozwoj",
    "priorytet_stabilnosc",
    "praca_z_ludzmi",
)


class PsychologyTestRequest(BaseModel):
    user_id: str
    closed_answers: list[int] = Field(
        ..., description="Likert answers for 17 statements (1-7)."
    )
    open_answers: list[str] = Field(
        ..., description="Four open answers in order."
    )

    @field_validator("closed_answers")
    @classmethod
    def validate_closed_answers(cls, value: list[int]) -> list[int]:
        if len(value) != 17:
            raise ValueError("Expected 17 closed answers for psychology test.")
        for v in value:
            if not 1 <= v <= 7:
                raise ValueError("Closed answers must be between 1 and 7.")
        return value

    @field_validator("open_answers")
    @classmethod
    def validate_open_answers(cls, value: list[str]) -> list[str]:
        if len(value) != 4:
            raise ValueError("Expected 4 open answers for psychology test.")
        cleaned = []
        for answer in value:
            stripped = answer.strip()
            if not stripped:
                raise ValueError("Open answers cannot be empty.")
            cleaned.append(stripped[:2000])
        return cleaned

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("user_id cannot be empty.")
        return stripped


class VocationalTestRequest(BaseModel):
    user_id: str
    closed_answers: list[int] = Field(
        ..., description="Likert answers for 23 vocational statements (1-7)."
    )
    open_answers: list[str] = Field(
        ..., description="Four open answers for vocational context."
    )

    @field_validator("closed_answers")
    @classmethod
    def validate_voc_closed_answers(cls, value: list[int]) -> list[int]:
        if len(value) != 23:
            raise ValueError("Expected 23 closed answers for vocational test.")
        for v in value:
            if not 1 <= v <= 7:
                raise ValueError("Closed answers must be between 1 and 7.")
        return value

    @field_validator("open_answers")
    @classmethod
    def validate_voc_open_answers(cls, value: list[str]) -> list[str]:
        if len(value) != 4:
            raise ValueError("Expected 4 open answers for vocational test.")
        cleaned = []
        for answer in value:
            stripped = answer.strip()
            if not stripped:
                raise ValueError("Open answers cannot be empty.")
            cleaned.append(stripped[:2000])
        return cleaned

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("user_id cannot be empty.")
        return stripped


class TestSubmissionResponse(BaseModel):
    message: str
    trait_scores: Mapping[str, float]
    persona_generated: bool = False
    persona_text: str | None = None
    closed_answers_scoring: Mapping[str, Any] = Field(default_factory=dict)
    open_answers_scoring: Mapping[str, float] = Field(default_factory=dict)
    open_answers_details: list[dict] = Field(default_factory=list)
    scoring_logs: list[str] = Field(default_factory=list)
    question_details: list[dict] = Field(default_factory=list)


@dataclass(frozen=True)
class TraitImpact:
    trait: str
    weight: float
    reverse: bool = False
