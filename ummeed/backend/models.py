"""SQLAlchemy ORM models for Ummeed.

The declarative ``Base`` lives here and is imported by ``database.py`` (added
in step 3) so that ``Base.metadata.create_all()`` picks up every model.
"""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Gender(str, enum.Enum):
    child_female = "child_female"
    child_male = "child_male"
    woman = "woman"


class CaseStatus(str, enum.Enum):
    active = "active"
    found = "found"
    closed = "closed"


class SightingStatus(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    notified = "notified"


class ModeratorRole(str, enum.Enum):
    moderator = "moderator"
    admin = "admin"


class TwitterScanStatus(str, enum.Enum):
    pending = "pending"
    reviewed = "reviewed"
    dismissed = "dismissed"


class Case(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    name = Column(String(200), nullable=False, index=True)
    age = Column(Integer, nullable=False)
    gender = Column(Enum(Gender), nullable=False, index=True)

    photo_path = Column(String(500), nullable=False)

    last_seen_location = Column(String(300), nullable=False, index=True)
    last_seen_lat = Column(Float, nullable=True)
    last_seen_lng = Column(Float, nullable=True)
    last_seen_date = Column(Date, nullable=False)

    description = Column(Text, nullable=True)
    status = Column(
        Enum(CaseStatus),
        nullable=False,
        default=CaseStatus.active,
        index=True,
    )
    source = Column(String(200), nullable=True)

    # Private contact info — moderator/admin only, never exposed in public API.
    contact_email = Column(String(255), nullable=True)
    whatsapp_number = Column(String(20), nullable=True)

    sightings = relationship(
        "SightingReport",
        foreign_keys="SightingReport.case_id",
        back_populates="case",
        cascade="all, delete-orphan",
    )


class SightingReport(Base):
    __tablename__ = "sighting_reports"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Optional: set when reporter started from a specific case page.
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=True, index=True)

    # Private — never shown to the public or family.
    reporter_phone = Column(String(20), nullable=False)
    reporter_name = Column(String(200), nullable=True)

    sighting_photo_path = Column(String(500), nullable=False)
    sighting_location = Column(String(300), nullable=False)
    sighting_lat = Column(Float, nullable=True)
    sighting_lng = Column(Float, nullable=True)
    sighting_datetime = Column(DateTime, nullable=False)
    notes = Column(Text, nullable=True)

    # Filled in by the DeepFace background task.
    ai_match_score = Column(Float, nullable=True)
    ai_match_case_id = Column(
        Integer, ForeignKey("cases.id"), nullable=True, index=True
    )

    status = Column(
        Enum(SightingStatus),
        nullable=False,
        default=SightingStatus.pending,
        index=True,
    )
    moderator_notes = Column(Text, nullable=True)

    case = relationship(
        "Case", foreign_keys=[case_id], back_populates="sightings"
    )
    ai_match_case = relationship("Case", foreign_keys=[ai_match_case_id])


class Moderator(Base):
    __tablename__ = "moderators"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    email = Column(String(255), nullable=False, unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(
        Enum(ModeratorRole),
        nullable=False,
        default=ModeratorRole.moderator,
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class TwitterScan(Base):
    __tablename__ = "twitter_scans"

    id = Column(Integer, primary_key=True, index=True)
    scanned_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    tweet_id = Column(String(50), nullable=False, unique=True, index=True)
    tweet_url = Column(String(500), nullable=False)
    tweet_text = Column(Text, nullable=True)
    image_url = Column(String(500), nullable=True)

    ai_match_score = Column(Float, nullable=True)
    matched_case_id = Column(
        Integer, ForeignKey("cases.id"), nullable=True, index=True
    )

    status = Column(
        Enum(TwitterScanStatus),
        nullable=False,
        default=TwitterScanStatus.pending,
        index=True,
    )

    matched_case = relationship("Case")
