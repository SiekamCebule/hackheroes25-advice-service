from fastapi import APIRouter, Depends, Header, HTTPException, Query

from app.models.advice import AdviceRequestContext, AdviceResponsePayload, UserIdentifier
from app.services.advice_service import AdviceService, get_advice_service

router = APIRouter(prefix="/advice", tags=["advice"])


@router.get("", response_model=AdviceResponsePayload)
async def get_advice(
    user_id: str | None = Query(
        default=None, description="Internal user identifier."),
    user_message: str = Query(
        ...,
        description="Latest chat message from the user UI.",
        alias="message",
    ),
    auth_token: str | None = Header(
        default=None,
        alias="Authorization",
        description="Bearer or custom token identifying the user session.",
    ),
    advice_service: AdviceService = Depends(get_advice_service),
) -> AdviceResponsePayload:
    user_identifier = UserIdentifier(user_id=user_id, auth_token=auth_token)
    if user_identifier.is_empty():
        raise HTTPException(
            status_code=400,
            detail="Either user_id query parameter or Authorization header must be provided.",
        )
    request_context = AdviceRequestContext(
        user_identifier=user_identifier,
        user_message=user_message,
    )
    return await advice_service.get_advice_response(request_context)
