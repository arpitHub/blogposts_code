"""Public API endpoints: OTP, case browsing, and sighting submission.

Cases are filtered to ``active`` and the responses strip private fields
(``contact_email``, ``whatsapp_number``, reporter info on sightings).
"""
import re
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import or_
from sqlalchemy.orm import Session

from auth import SCOPE_SIGHTING, create_token, get_otp_session_phone
from config import settings
from database import get_db
from models import Case, CaseStatus, Gender, SightingReport, SightingStatus
from services.otp_service import (
    generate_otp,
    send_otp_sms,
    store_otp,
    verify_otp,
)

router = APIRouter(prefix="/api", tags=["public"])


# --------------------------------------------------------------------------
# OTP
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------


class CaseListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    age: int
    gender: Gender
    photo_url: str
    last_seen_location: str
    last_seen_date: date
    status: CaseStatus


class CaseDetail(CaseListItem):
    description: Optional[str] = None
    last_seen_lat: Optional[float] = None
    last_seen_lng: Optional[float] = None
    source: Optional[str] = None
    created_at: datetime


class CaseListResponse(BaseModel):
    items: list[CaseListItem]
    total: int
    page: int
    page_size: int


class CaseMapPin(BaseModel):
    id: int
    name: str
    photo_url: str
    last_seen_location: str
    last_seen_lat: float
    last_seen_lng: float


def _photo_url(photo_path: Optional[str]) -> str:
    if not photo_path:
        return ""
    return f"/storage/{photo_path.lstrip('/')}"


def _to_list_item(case: Case) -> CaseListItem:
    return CaseListItem(
        id=case.id,
        name=case.name,
        age=case.age,
        gender=case.gender,
        photo_url=_photo_url(case.photo_path),
        last_seen_location=case.last_seen_location,
        last_seen_date=case.last_seen_date,
        status=case.status,
    )


def _to_detail(case: Case) -> CaseDetail:
    return CaseDetail(
        **_to_list_item(case).model_dump(),
        description=case.description,
        last_seen_lat=case.last_seen_lat,
        last_seen_lng=case.last_seen_lng,
        source=case.source,
        created_at=case.created_at,
    )


@router.get("/cases", response_model=CaseListResponse)
def list_cases(
    db: Session = Depends(get_db),
    q: Optional[str] = Query(
        None, description="search name or last-seen location"
    ),
    gender: Optional[Gender] = None,
    age_min: Optional[int] = Query(None, ge=0, le=120),
    age_max: Optional[int] = Query(None, ge=0, le=120),
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    query = db.query(Case).filter(Case.status == CaseStatus.active)

    if q:
        like = f"%{q}%"
        query = query.filter(
            or_(Case.name.ilike(like), Case.last_seen_location.ilike(like))
        )
    if gender is not None:
        query = query.filter(Case.gender == gender)
    if age_min is not None:
        query = query.filter(Case.age >= age_min)
    if age_max is not None:
        query = query.filter(Case.age <= age_max)
    if date_from is not None:
        query = query.filter(Case.last_seen_date >= date_from)
    if date_to is not None:
        query = query.filter(Case.last_seen_date <= date_to)

    total = query.count()
    cases = (
        query.order_by(Case.last_seen_date.desc(), Case.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return CaseListResponse(
        items=[_to_list_item(c) for c in cases],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/cases/map", response_model=list[CaseMapPin])
def cases_for_map(db: Session = Depends(get_db)):
    cases = (
        db.query(Case)
        .filter(
            Case.status == CaseStatus.active,
            Case.last_seen_lat.isnot(None),
            Case.last_seen_lng.isnot(None),
        )
        .all()
    )
    return [
        CaseMapPin(
            id=c.id,
            name=c.name,
            photo_url=_photo_url(c.photo_path),
            last_seen_location=c.last_seen_location,
            last_seen_lat=c.last_seen_lat,
            last_seen_lng=c.last_seen_lng,
        )
        for c in cases
    ]


@router.get("/cases/{case_id}", response_model=CaseDetail)
def get_case(case_id: int, db: Session = Depends(get_db)):
    case = db.query(Case).filter(Case.id == case_id).first()
    # Closed cases are hidden from public; "found" cases stay visible so
    # families/reporters get closure on outcomes.
    if not case or case.status == CaseStatus.closed:
        raise HTTPException(status_code=404, detail="case not found")
    return _to_detail(case)


# --------------------------------------------------------------------------
# Sightings
# --------------------------------------------------------------------------

_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_MAX_PHOTO_BYTES = 10 * 1024 * 1024  # 10 MB


def _save_sighting_photo(file: UploadFile) -> str:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail="unsupported image type")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="file is not an image")

    storage_root = Path(settings.PHOTO_STORAGE_PATH).resolve()
    sightings_dir = storage_root / "sightings"
    sightings_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}{ext}"
    full_path = sightings_dir / filename

    size = 0
    with full_path.open("wb") as out:
        while chunk := file.file.read(64 * 1024):
            size += len(chunk)
            if size > _MAX_PHOTO_BYTES:
                out.close()
                full_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail="photo too large (10 MB max)",
                )
            out.write(chunk)

    # Return path relative to PHOTO_STORAGE_PATH so it matches the
    # /storage StaticFiles mount in main.py.
    return f"sightings/{filename}"


class SightingResponse(BaseModel):
    id: int
    status: SightingStatus
    message: str


@router.post("/sightings", response_model=SightingResponse, status_code=201)
async def submit_sighting(
    sighting_location: str = Form(..., min_length=1),
    sighting_datetime: datetime = Form(...),
    photo: UploadFile = File(...),
    case_id: Optional[int] = Form(None),
    sighting_lat: Optional[float] = Form(None),
    sighting_lng: Optional[float] = Form(None),
    notes: Optional[str] = Form(None),
    reporter_name: Optional[str] = Form(None),
    phone: str = Depends(get_otp_session_phone),
    db: Session = Depends(get_db),
):
    if case_id is not None:
        if not db.query(Case).filter(Case.id == case_id).first():
            raise HTTPException(status_code=404, detail="case not found")

    photo_path = _save_sighting_photo(photo)

    sighting = SightingReport(
        case_id=case_id,
        reporter_phone=phone,
        reporter_name=reporter_name,
        sighting_photo_path=photo_path,
        sighting_location=sighting_location,
        sighting_lat=sighting_lat,
        sighting_lng=sighting_lng,
        sighting_datetime=sighting_datetime,
        notes=notes,
        status=SightingStatus.pending,
    )
    db.add(sighting)
    db.commit()
    db.refresh(sighting)

    # TODO step 5: queue DeepFace background match for this sighting.
    # background_tasks.add_task(run_match_for_sighting, sighting.id)

    return SightingResponse(
        id=sighting.id,
        status=sighting.status,
        message="Thank you. Our team will review your report.",
    )
