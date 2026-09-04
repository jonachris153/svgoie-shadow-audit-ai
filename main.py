from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

APP_TITLE = "OIE Shadow Audit AI Service"
APP_VERSION = "2.0.0"


class ExceptionCase(BaseModel):
    case_id: str = Field(min_length=1)
    case_date: Optional[date] = None
    exception_type: str = Field(min_length=1)
    approval_status: Optional[str] = None
    branch: Optional[str] = None
    officer_name: Optional[str] = None
    officer_id: Optional[str] = None
    justification: Optional[str] = None


class ShadowAuditRequest(BaseModel):
    current_period_cases: list[ExceptionCase] = Field(min_length=1)
    previous_period_exception_count: int = Field(ge=0)
    weeks_observed: int = Field(default=6, ge=1)
    threshold: float = Field(default=30.0, ge=0.0, le=100.0)


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


app = FastAPI(title=APP_TITLE, version=APP_VERSION)


def build_baseline_model() -> Pipeline:
    rng = np.random.default_rng(42)
    n = 400
    prev = rng.poisson(8, n)
    growth = rng.normal(0.05, 0.08, n)
    current = np.maximum(
        0, np.rint(prev * (1 + growth) + rng.normal(0, 1.3, n))
    ).astype(int)

    approval = np.minimum(
        current, np.rint(current * rng.uniform(0.10, 0.35, n))
    ).astype(int)
    policy = np.minimum(
        current, np.rint(current * rng.uniform(0.08, 0.28, n))
    ).astype(int)
    documentation = np.minimum(
        current, np.rint(current * rng.uniform(0.04, 0.18, n))
    ).astype(int)
    recent_rate = np.clip(rng.normal(0.18, 0.07, n), 0, 1)

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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def case_features(
    current_count: int,
    previous_count: int,
    breakdown: Counter[str],
) -> np.ndarray:
    approval = breakdown.get("Approval Override", 0)
    policy = breakdown.get("Policy Exception", 0)
    documentation = breakdown.get("Documentation Exception", 0)

    recent_rate = current_count / max(current_count + previous_count, 1)
    growth = (current_count - previous_count) / max(previous_count, 1)

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


def anomaly_score(X: np.ndarray) -> float:
    decision = float(MODEL.decision_function(X)[0])
    score = 50.0 - 180.0 * decision
    return round(clamp(score, 0.0, 100.0), 2)


def dominant_justification(cases: list[ExceptionCase]) -> Optional[str]:
    texts = [
        c.justification.strip().lower()
        for c in cases
        if c.justification and c.justification.strip()
    ]
    if len(texts) < 2:
        return None

    stop = {
        "the", "a", "an", "and", "or", "to", "of", "for", "in", "on",
        "is", "was", "with", "this", "that", "case", "due",
    }
    token_sets = []
    for text in texts:
        tokens = {t.strip(".,;:!?()[]{}\"'") for t in text.split()}
        token_sets.append({t for t in tokens if len(t) >= 5 and t not in stop})

    counts = Counter(token for s in token_sets for token in s)
    repeated = [token for token, count in counts.items() if count >= 2]
    if not repeated:
        return None

    repeated.sort(key=lambda token: (-counts[token], token))
    return "Repeated justification terms detected: " + " / ".join(repeated[:3])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": APP_TITLE, "modelVersion": APP_VERSION}


@app.post("/shadow-audit/analyze", response_model=ShadowAuditResponse)
def analyze(payload: ShadowAuditRequest) -> ShadowAuditResponse:
    cases = payload.current_period_cases
    current_count = len(cases)
    previous_count = payload.previous_period_exception_count
    breakdown = Counter(c.exception_type.strip() for c in cases)

    officers = Counter(c.officer_id or c.officer_name or "Unknown Officer" for c in cases)
    branches = Counter(c.branch or "Unknown Branch" for c in cases)

    top_officer, top_officer_count = officers.most_common(1)[0]
    top_branch, top_branch_count = branches.most_common(1)[0]

    score = anomaly_score(case_features(current_count, previous_count, breakdown))
    growth_pct = ((current_count - previous_count) / max(previous_count, 1)) * 100

    signals: list[str] = []

    if previous_count > 0 and growth_pct >= 30:
        signals.append(f"Exception volume increased by {growth_pct:.0f}%.")

    if top_officer_count / max(current_count, 1) >= 0.40:
        signals.append(
            f"{top_officer} accounts for "
            f"{top_officer_count / current_count * 100:.0f}% of current exceptions."
        )

    if top_branch_count / max(current_count, 1) >= 0.50:
        signals.append(
            f"{top_branch} accounts for "
            f"{top_branch_count / current_count * 100:.0f}% of current exceptions."
        )

    dominant_type, dominant_count = breakdown.most_common(1)[0]
    if dominant_count / max(current_count, 1) >= 0.50:
        signals.append(
            f"{dominant_type} represents "
            f"{dominant_count / current_count * 100:.0f}% of exceptions."
        )

    justification_signal = dominant_justification(cases)
    if justification_signal:
        signals.append(justification_signal)

    if not signals:
        signals.append("Current exception patterns remain close to the prototype baseline.")

    if score >= payload.threshold:
        status = "DRIFT DETECTED"
        action = "Escalate for targeted compliance review"
    elif score >= max(payload.threshold - 20, 20):
        status = "WATCH"
        action = "Flag for monitoring and manual review"
    else:
        status = "NORMAL"
        action = "Continue routine monitoring"

    insight = (
        f"Shadow Audit identified a {status.lower()} pattern with a "
        f"risk score of {score:.2f}. " + " ".join(signals)
    )

    return ShadowAuditResponse(
        driftScore=score,
        driftStatus=status,
        primarySignals=signals[:5],
        aiInsight=insight,
        recommendedAction=action,
        affectedCaseIds=[c.case_id for c in cases],
        affectedOfficers=[name for name, _ in officers.most_common(5)],
        affectedBranches=[name for name, _ in branches.most_common(5)],
        exceptionBreakdown=dict(breakdown),
        justificationPattern=justification_signal,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
