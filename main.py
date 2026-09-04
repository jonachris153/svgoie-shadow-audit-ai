from __future__ import annotations

import json
from collections import Counter
from datetime import date
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


APP_TITLE = "OIE Shadow Audit AI Service"
APP_VERSION = "2.0.0"


# ============================================================
# CASE DATA MODEL
# ============================================================

class ExceptionCase(BaseModel):
    case_id: str = Field(min_length=1)
    case_date: Optional[date] = None
    exception_flag: Optional[bool] = None
    exception_type: str = Field(min_length=1)
    approval_status: Optional[str] = None
    branch: Optional[str] = None
    officer_name: Optional[str] = None
    officer_id: Optional[str] = None
    justification: Optional[str] = None


# ============================================================
# NEWGEN REQUEST MODEL
#
# Newgen sends the case information as a JSON STRING through
# AIRequestJson instead of trying to map a complex array.
# ============================================================

class ShadowAuditRequest(BaseModel):
    previous_period_exception_count: int = Field(ge=0)
    weeks_observed: int = Field(default=6, ge=1)
    threshold: float = Field(default=30.0, ge=0.0, le=100.0)
    AIRequestJson: str = Field(min_length=2)


# ============================================================
# AI RESPONSE MODEL
# ============================================================

class ShadowAuditResponse(BaseModel):
    driftScore: float
    driftStatus: str
    primarySignals: list[str]
    aiInsight: str
    recommendedAction: str
    affectedCaseIds: list[str]
    affectedOfficers: list[str]
    affectedBranches: list[str]
    exceptionBreakdown: dict[str, int]
    justificationPattern: Optional[str] = None
    modelVersion: str = APP_VERSION


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
)


# ============================================================
# SYNTHETIC NORMAL BASELINE
#
# This is a prototype baseline, not production-trained data.
# ============================================================

def build_baseline_model() -> Pipeline:
    rng = np.random.default_rng(42)

    n = 400

    prev = rng.poisson(8, n)

    growth = rng.normal(0.05, 0.08, n)

    current = np.maximum(
        0,
        np.rint(
            prev * (1 + growth)
            + rng.normal(0, 1.3, n)
        ),
    ).astype(int)

    approval = np.minimum(
        current,
        np.rint(
            current * rng.uniform(0.10, 0.35, n)
        ),
    ).astype(int)

    policy = np.minimum(
        current,
        np.rint(
            current * rng.uniform(0.08, 0.28, n)
        ),
    ).astype(int)

    documentation = np.minimum(
        current,
        np.rint(
            current * rng.uniform(0.04, 0.18, n)
        ),
    ).astype(int)

    recent_rate = np.clip(
        rng.normal(0.18, 0.07, n),
        0,
        1,
    )

    X = np.column_stack(
        [
            current,
            prev,
            approval,
            policy,
            documentation,
            recent_rate,
            (current - prev) / np.maximum(prev, 1),
        ]
    )

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "iforest",
                IsolationForest(
                    n_estimators=250,
                    contamination=0.08,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X)

    return model


MODEL = build_baseline_model()


# ============================================================
# UTILITY
# ============================================================

def clamp(
    value: float,
    low: float,
    high: float,
) -> float:
    return max(low, min(high, value))


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def case_features(
    current_count: int,
    previous_count: int,
    breakdown: Counter[str],
) -> np.ndarray:

    approval = breakdown.get(
        "Approval Override",
        0,
    )

    policy = breakdown.get(
        "Policy Exception",
        0,
    )

    documentation = breakdown.get(
        "Documentation Exception",
        0,
    )

    recent_rate = (
        current_count
        / max(current_count + previous_count, 1)
    )

    growth = (
        (current_count - previous_count)
        / max(previous_count, 1)
    )

    return np.array(
        [[
            current_count,
            previous_count,
            approval,
            policy,
            documentation,
            recent_rate,
            growth,
        ]],
        dtype=float,
    )


# ============================================================
# ISOLATION FOREST ANOMALY SCORE
# ============================================================

def anomaly_score(
    X: np.ndarray,
) -> float:

    decision = float(
        MODEL.decision_function(X)[0]
    )

    score = 50.0 - 180.0 * decision

    return round(
        clamp(score, 0.0, 100.0),
        2,
    )


# ============================================================
# JUSTIFICATION PATTERN DETECTION
#
# Transparent prototype method:
# repeated meaningful terms across justifications.
# ============================================================

def dominant_justification(
    cases: list[ExceptionCase],
) -> Optional[str]:

    texts = [
        c.justification.strip().lower()
        for c in cases
        if c.justification
        and c.justification.strip()
    ]

    if len(texts) < 2:
        return None

    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "for",
        "in",
        "on",
        "is",
        "was",
        "with",
        "this",
        "that",
        "case",
        "due",
    }

    token_sets = []

    for text in texts:

        tokens = {
            t.strip(
                ".,;:!?()[]{}\"'"
            )
            for t in text.split()
        }

        token_sets.append(
            {
                t
                for t in tokens
                if len(t) >= 5
                and t not in stop
            }
        )

    counts = Counter(
        token
        for token_set in token_sets
        for token in token_set
    )

    repeated = [
        token
        for token, count in counts.items()
        if count >= 2
    ]

    if not repeated:
        return None

    repeated.sort(
        key=lambda token: (
            -counts[token],
            token,
        )
    )

    return (
        "Repeated justification terms detected: "
        + " / ".join(repeated[:3])
    )


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health() -> dict:

    return {
        "status": "ok",
        "service": APP_TITLE,
        "modelVersion": APP_VERSION,
    }


# ============================================================
# SHADOW AUDIT ANALYSIS
# ============================================================

@app.post(
    "/shadow-audit/analyze",
    response_model=ShadowAuditResponse,
)
def analyze(
    payload: ShadowAuditRequest,
) -> ShadowAuditResponse:

    # --------------------------------------------------------
    # Parse JSON string received from Newgen
    # --------------------------------------------------------

    try:

        raw_data = json.loads(
            payload.AIRequestJson
        )

    except json.JSONDecodeError as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                "AIRequestJson contains invalid JSON: "
                f"{exc.msg}"
            ),
        ) from exc

    # --------------------------------------------------------
    # Accept either:
    #
    # 1. A direct array:
    #    [...]
    #
    # 2. An object containing:
    #    {"current_period_cases": [...]}
    #
    # This makes the API more tolerant of Newgen payloads.
    # --------------------------------------------------------

    if isinstance(raw_data, dict):

        raw_cases = raw_data.get(
            "current_period_cases",
            raw_data.get("cases"),
        )

    elif isinstance(raw_data, list):

        raw_cases = raw_data

    else:

        raw_cases = None

    if not isinstance(
        raw_cases,
        list,
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "AIRequestJson must contain a "
                "case array."
            ),
        )

    if len(raw_cases) == 0:

        raise HTTPException(
            status_code=400,
            detail=(
                "AIRequestJson must contain "
                "at least one case."
            ),
        )

    # --------------------------------------------------------
    # Validate each case
    # --------------------------------------------------------

    try:

        cases = [
            ExceptionCase.model_validate(case)
            for case in raw_cases
        ]

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid case data in AIRequestJson: "
                f"{exc}"
            ),
        ) from exc

    # --------------------------------------------------------
    # CLOSED CASES + EXCEPTION CASES
    #
    # Shadow Audit operates on already-closed cases.
    # --------------------------------------------------------

    closed_cases = [
        case
        for case in cases
        if (
            case.approval_status is None
            or case.approval_status.upper()
            == "CLOSED"
        )
    ]

    if not closed_cases:

        raise HTTPException(
            status_code=400,
            detail=(
                "No CLOSED cases were found "
                "for Shadow Audit analysis."
            ),
        )

    exception_cases = [
        case
        for case in closed_cases
        if (
            case.exception_flag is None
            or case.exception_flag is True
        )
    ]

    if not exception_cases:

        raise HTTPException(
            status_code=400,
            detail=(
                "No exception cases were found "
                "among the CLOSED cases."
            ),
        )

    # --------------------------------------------------------
    # CURRENT PERIOD
    # --------------------------------------------------------

    current_cases = exception_cases

    current_count = len(
        current_cases
    )

    previous_count = (
        payload.previous_period_exception_count
    )

    # --------------------------------------------------------
    # EXCEPTION TYPE BREAKDOWN
    # --------------------------------------------------------

    breakdown = Counter(
        c.exception_type.strip()
        for c in current_cases
    )

    # --------------------------------------------------------
    # OFFICER CONCENTRATION
    # --------------------------------------------------------

    officers = Counter(
        c.officer_id
        or c.officer_name
        or "Unknown Officer"
        for c in current_cases
    )

    # --------------------------------------------------------
    # BRANCH CONCENTRATION
    # --------------------------------------------------------

    branches = Counter(
        c.branch
        or "Unknown Branch"
        for c in current_cases
    )

    top_officer, top_officer_count = (
        officers.most_common(1)[0]
    )

    top_branch, top_branch_count = (
        branches.most_common(1)[0]
    )

    # --------------------------------------------------------
    # AI ANOMALY SCORE
    # --------------------------------------------------------

    score = anomaly_score(
        case_features(
            current_count,
            previous_count,
            breakdown,
        )
    )

    # --------------------------------------------------------
    # EXCEPTION VOLUME GROWTH
    # --------------------------------------------------------

    growth_pct = (
        (
            current_count
            - previous_count
        )
        / max(previous_count, 1)
    ) * 100

    signals: list[str] = []

    # --------------------------------------------------------
    # SIGNAL 1: VOLUME GROWTH
    # --------------------------------------------------------

    if (
        previous_count > 0
        and growth_pct >= 30
    ):

        signals.append(
            f"Exception volume increased "
            f"by {growth_pct:.0f}%."
        )

    # --------------------------------------------------------
    # SIGNAL 2: OFFICER CONCENTRATION
    # --------------------------------------------------------

    if (
        top_officer_count
        / max(current_count, 1)
        >= 0.40
    ):

        signals.append(
            f"{top_officer} accounts for "
            f"{top_officer_count / current_count * 100:.0f}% "
            f"of current exceptions."
        )

    # --------------------------------------------------------
    # SIGNAL 3: BRANCH CONCENTRATION
    # --------------------------------------------------------

    if (
        top_branch_count
        / max(current_count, 1)
        >= 0.50
    ):

        signals.append(
            f"{top_branch} accounts for "
            f"{top_branch_count / current_count * 100:.0f}% "
            f"of current exceptions."
        )

    # --------------------------------------------------------
    # SIGNAL 4: EXCEPTION TYPE CONCENTRATION
    # --------------------------------------------------------

    dominant_type, dominant_count = (
        breakdown.most_common(1)[0]
    )

    if (
        dominant_count
        / max(current_count, 1)
        >= 0.50
    ):

        signals.append(
            f"{dominant_type} represents "
            f"{dominant_count / current_count * 100:.0f}% "
            f"of exceptions."
        )

    # --------------------------------------------------------
    # SIGNAL 5: JUSTIFICATION PATTERN
    # --------------------------------------------------------

    justification_signal = dominant_justification(
        current_cases
    )

    if justification_signal:

        signals.append(
            justification_signal
        )

    # --------------------------------------------------------
    # DEFAULT SIGNAL
    # --------------------------------------------------------

    if not signals:

        signals.append(
            "Current exception patterns remain "
            "close to the prototype baseline."
        )

    # --------------------------------------------------------
    # GOVERNANCE STATUS
    #
    # The threshold is governance logic.
    # The AI produces the DriftScore.
    # --------------------------------------------------------

    if score >= payload.threshold:

        status = "DRIFT DETECTED"

        action = (
            "Escalate for targeted "
            "compliance review"
        )

    elif score >= max(
        payload.threshold - 20,
        20,
    ):

        status = "WATCH"

        action = (
            "Flag for monitoring "
            "and manual review"
        )

    else:

        status = "NORMAL"

        action = (
            "Continue routine monitoring"
        )

    # --------------------------------------------------------
    # AI INSIGHT
    # --------------------------------------------------------

    insight = (
        f"Shadow Audit identified a "
        f"{status.lower()} pattern with a "
        f"risk score of {score:.2f}. "
        + " ".join(signals)
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return ShadowAuditResponse(

        driftScore=score,

        driftStatus=status,

        primarySignals=signals[:5],

        aiInsight=insight,

        recommendedAction=action,

        affectedCaseIds=[
            c.case_id
            for c in current_cases
        ],

        affectedOfficers=[
            name
            for name, _ in officers.most_common(5)
        ],

        affectedBranches=[
            name
            for name, _ in branches.most_common(5)
        ],

        exceptionBreakdown=dict(
            breakdown
        ),

        justificationPattern=(
            justification_signal
        ),
    )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )