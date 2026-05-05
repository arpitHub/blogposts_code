"""OTP generation, in-memory storage, and Fast2SMS delivery.

For MVP the OTP store is a process-local dict — fine for a single-worker
dev server. Swap to Redis when running multiple workers.
"""
import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from config import settings

log = logging.getLogger(__name__)


@dataclass
class _OTPRecord:
    hashed_otp: str
    expires_at: datetime


_OTP_STORE: dict[str, _OTPRecord] = {}


def _hash_otp(phone: str, otp: str) -> str:
    # JWT_SECRET acts as a server-side pepper.
    digest = hashlib.sha256()
    digest.update(f"{phone}:{otp}:{settings.JWT_SECRET}".encode())
    return digest.hexdigest()


def generate_otp() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def store_otp(phone: str, otp: str) -> None:
    _OTP_STORE[phone] = _OTPRecord(
        hashed_otp=_hash_otp(phone, otp),
        expires_at=datetime.now(timezone.utc)
        + timedelta(seconds=settings.OTP_TTL_SECONDS),
    )


def verify_otp(phone: str, otp: str) -> bool:
    record = _OTP_STORE.get(phone)
    if record is None:
        return False
    if datetime.now(timezone.utc) > record.expires_at:
        _OTP_STORE.pop(phone, None)
        return False
    if _hash_otp(phone, otp) != record.hashed_otp:
        return False
    # Single-use: drop on success so the same OTP can't be replayed.
    _OTP_STORE.pop(phone, None)
    return True


async def send_otp_sms(phone: str, otp: str) -> None:
    """Send an OTP SMS via Fast2SMS. Logs only when no API key is configured."""
    if not settings.FAST2SMS_API_KEY:
        log.warning("FAST2SMS_API_KEY not set — dev OTP for %s: %s", phone, otp)
        return

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            "https://www.fast2sms.com/dev/bulkV2",
            headers={"authorization": settings.FAST2SMS_API_KEY},
            data={
                "variables_values": otp,
                "route": "otp",
                "numbers": phone,
            },
        )

    if resp.status_code != 200:
        log.error("Fast2SMS failed: %s %s", resp.status_code, resp.text)
        raise RuntimeError("OTP send failed")
