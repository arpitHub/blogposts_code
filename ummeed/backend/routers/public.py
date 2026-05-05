"""Public API endpoints. Step 3 wires only OTP send/verify; cases and
sighting submission are added in step 4."""
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from auth import SCOPE_SIGHTING, create_token
from config import settings
from services.otp_service import (
    generate_otp,
    send_otp_sms,
    store_otp,
    verify_otp,
)

router = APIRouter(prefix="/api", tags=["public"])

# Indian mobile: 10 digits, starts with 6-9.
_PHONE_RE = re.compile(r"^[6-9]\d{9}$")


def _normalize_phone(phone: str) -> str:
    digits = re.sub(r"\D", "", phone)
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    if not _PHONE_RE.match(digits):
        raise HTTPException(
            status_code=400, detail="invalid Indian mobile number"
        )
    return digits


class OTPSendRequest(BaseModel):
    phone: str = Field(..., description="10-digit Indian mobile number")


class OTPVerifyRequest(BaseModel):
    phone: str
    otp: str = Field(..., min_length=6, max_length=6)


class OTPVerifyResponse(BaseModel):
    session_token: str
    expires_in: int


@router.post("/otp/send", status_code=204)
async def send_otp(req: OTPSendRequest):
    phone = _normalize_phone(req.phone)
    otp = generate_otp()
    store_otp(phone, otp)
    await send_otp_sms(phone, otp)


@router.post("/otp/verify", response_model=OTPVerifyResponse)
async def verify_and_issue_session(req: OTPVerifyRequest):
    phone = _normalize_phone(req.phone)
    if not verify_otp(phone, req.otp):
        raise HTTPException(status_code=400, detail="invalid or expired OTP")
    token = create_token(
        sub=phone,
        scope=SCOPE_SIGHTING,
        ttl_seconds=settings.OTP_SESSION_TTL_SECONDS,
    )
    return OTPVerifyResponse(
        session_token=token,
        expires_in=settings.OTP_SESSION_TTL_SECONDS,
    )
