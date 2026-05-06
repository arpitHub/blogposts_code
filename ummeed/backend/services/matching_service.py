"""DeepFace face-matching service.

DeepFace.verify() returns a *cosine distance* (lower = more similar).
We convert to similarity = 1 - distance so higher is better, which lines
up with the spec's MATCH_THRESHOLD=0.4 and HIGH_CONFIDENCE_THRESHOLD=0.75.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from config import settings
from database import SessionLocal
from models import Case, CaseStatus, SightingReport

log = logging.getLogger(__name__)

MATCH_THRESHOLD = 0.40
HIGH_CONFIDENCE_THRESHOLD = 0.75

_MODEL_NAME = "Facenet512"
_DETECTOR_BACKEND = "retinaface"
_DISTANCE_METRIC = "cosine"


@dataclass
class MatchCandidate:
    case_id: int
    similarity_score: float


def _absolute(photo_path: str) -> str:
    """Resolve a DB-stored relative path under PHOTO_STORAGE_PATH."""
    p = Path(photo_path)
    if p.is_absolute():
        return str(p)
    return str(Path(settings.PHOTO_STORAGE_PATH) / photo_path)


def _verify(sighting_photo: str, case_photo: str) -> Optional[float]:
    """Return similarity (0..1) or None if face detection fails."""
    # Lazy import: DeepFace pulls TensorFlow which is slow to load.
    # Importing here means /api/health and OTP routes don't pay the cost.
    from deepface import DeepFace

    try:
        result = DeepFace.verify(
            img1_path=sighting_photo,
            img2_path=case_photo,
            model_name=_MODEL_NAME,
            detector_backend=_DETECTOR_BACKEND,
            distance_metric=_DISTANCE_METRIC,
            enforce_detection=True,
        )
    except ValueError as e:
        # DeepFace raises ValueError when no face is detected.
        log.info("face detection skipped: %s", e)
        return None
    except Exception:
        log.exception(
            "DeepFace verify failed for %s vs %s", sighting_photo, case_photo
        )
        return None

    return max(0.0, 1.0 - float(result["distance"]))


def match_sighting_to_cases(
    sighting_photo_path: str,
    case_photos: list[tuple[int, str]],
) -> list[MatchCandidate]:
    """Compare a sighting against (case_id, photo_path) pairs.

    Returns matches with similarity > MATCH_THRESHOLD, sorted by score desc.
    """
    sighting_abs = _absolute(sighting_photo_path)
    candidates: list[MatchCandidate] = []
    for case_id, case_photo_path in case_photos:
        score = _verify(sighting_abs, _absolute(case_photo_path))
        if score is None or score <= MATCH_THRESHOLD:
            continue
        candidates.append(
            MatchCandidate(case_id=case_id, similarity_score=score)
        )
    candidates.sort(key=lambda c: c.similarity_score, reverse=True)
    return candidates


def confidence_level(score: Optional[float]) -> str:
    """Bucket a similarity score into 'high' / 'possible' / 'low'."""
    if score is None:
        return "low"
    if score > HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if score > MATCH_THRESHOLD:
        return "possible"
    return "low"


def run_match_for_sighting(sighting_id: int) -> None:
    """Background-task entry point: match a sighting against active cases
    and persist the top result on the SightingReport row.

    Opens its own session because the request's DB session is already
    closed by the time BackgroundTasks runs.
    """
    db: Session = SessionLocal()
    try:
        sighting = (
            db.query(SightingReport)
            .filter(SightingReport.id == sighting_id)
            .first()
        )
        if not sighting:
            log.error("sighting %d not found for matching", sighting_id)
            return

        active = (
            db.query(Case.id, Case.photo_path)
            .filter(Case.status == CaseStatus.active)
            .all()
        )
        if not active:
            log.info("no active cases to match against")
            return

        results = match_sighting_to_cases(
            sighting.sighting_photo_path,
            [(c.id, c.photo_path) for c in active],
        )
        if not results:
            log.info(
                "no matches above %.2f for sighting %d",
                MATCH_THRESHOLD,
                sighting_id,
            )
            return

        top = results[0]
        sighting.ai_match_score = top.similarity_score
        sighting.ai_match_case_id = top.case_id
        db.commit()
        log.info(
            "sighting %d matched case %d score=%.3f (%s)",
            sighting_id,
            top.case_id,
            top.similarity_score,
            confidence_level(top.similarity_score),
        )
    finally:
        db.close()
