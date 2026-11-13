"""Read-only API for the current process's startup inspection snapshot."""

import re
from collections import Counter
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.database.initialize import DEPLOYMENT_STATUSES, LEGACY_DEPLOYMENT_STATUSES
from backend.services.deployment_inspector import CLASSIFICATIONS


router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])

OUTCOMES = {"completed", "partial", "unavailable", "failed"}
DEPLOYMENT_ID_PATTERN = re.compile(r"^[a-f0-9]{24}$")
SAFE_STATUSES = frozenset((*DEPLOYMENT_STATUSES, *LEGACY_DEPLOYMENT_STATUSES))
HISTORICAL_WARNING = (
    "Historical startup observation only; it is not current deployment health."
)
SAFE_REASONS = {
    "running_verified": (
        "Docker identity and loopback port binding were verified at startup; "
        "application health was not checked."
    ),
    "exited_verified": "Docker reported the identity-verified container as exited at startup.",
    "created_verified": "Docker reported the identity-verified container as created at startup.",
    "container_not_found": "The expected container was not found during startup inspection.",
    "docker_unavailable": "Docker could not be inspected during startup.",
    "identity_mismatch": "Container identity or ownership could not be verified.",
    "ambiguous_lookup": "The expected container lookup was ambiguous.",
    "missing_identity": "Required deployment identity information was missing or invalid.",
    "unsupported_state": "Docker reported a state that startup inspection does not support.",
    "state_mismatch": "The recorded deployment state did not match the Docker observation.",
    "inspection_inconclusive": "Startup inspection could not establish a conclusive result.",
    "stale_database_snapshot": "The deployment record changed during startup inspection.",
}


class StartupInspectionFinding(BaseModel):
    deployment_id: str
    status: str | None
    classification: str
    reason: str


class StartupInspectionResponse(BaseModel):
    captured_at: str
    outcome: Literal["completed", "partial", "unavailable", "failed"]
    classification_counts: dict[str, int]
    results: list[StartupInspectionFinding]
    historical: bool
    warning: str


@router.get("/startup-inspection", response_model=StartupInspectionResponse)
def get_startup_inspection(request: Request) -> StartupInspectionResponse:
    """Return sanitized startup findings without querying Docker or MongoDB."""
    snapshot = getattr(request.app.state, "startup_inspection_snapshot", None)
    if not isinstance(snapshot, dict):
        raise HTTPException(
            status_code=503,
            detail={
                "code": "startup_snapshot_unavailable",
                "message": "No startup inspection snapshot is available.",
            },
        )

    captured_at = snapshot.get("captured_at")
    if not isinstance(captured_at, str):
        raise HTTPException(
            status_code=503,
            detail={
                "code": "startup_snapshot_unavailable",
                "message": "The startup inspection snapshot is invalid.",
            },
        )
    try:
        parsed_capture_time = datetime.fromisoformat(captured_at)
    except ValueError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "startup_snapshot_unavailable",
                "message": "The startup inspection snapshot is invalid.",
            },
        ) from error
    if parsed_capture_time.tzinfo is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "startup_snapshot_unavailable",
                "message": "The startup inspection snapshot is invalid.",
            },
        )

    outcome = snapshot.get("outcome")
    if not isinstance(outcome, str) or outcome not in OUTCOMES:
        outcome = "failed"

    findings = []
    results = snapshot.get("results")
    if not isinstance(results, list):
        results = []
    for result in results:
        if not isinstance(result, dict):
            continue
        classification = result.get("classification")
        if classification not in CLASSIFICATIONS:
            classification = "inspection_inconclusive"
        deployment_id = result.get("deployment_id")
        if (
            not isinstance(deployment_id, str)
            or not DEPLOYMENT_ID_PATTERN.fullmatch(deployment_id)
        ):
            deployment_id = "unavailable"
            classification = "inspection_inconclusive"
        status = result.get("status")
        findings.append(
            StartupInspectionFinding(
                deployment_id=deployment_id,
                status=(
                    status
                    if isinstance(status, str) and status in SAFE_STATUSES
                    else "unknown" if isinstance(status, str) else None
                ),
                classification=classification,
                reason=SAFE_REASONS[classification],
            )
        )

    counts = Counter(finding.classification for finding in findings)
    return StartupInspectionResponse(
        captured_at=captured_at,
        outcome=outcome,
        classification_counts={
            classification: counts[classification]
            for classification in CLASSIFICATIONS
            if counts[classification]
        },
        results=findings,
        historical=True,
        warning=HISTORICAL_WARNING,
    )
