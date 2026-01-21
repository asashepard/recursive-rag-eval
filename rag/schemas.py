"""
schemas.py
----------
Pydantic models for structured RAG output with citations.
Forces the LLM to produce evaluation-ready JSON.
"""

from typing import Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A citation pointing back to a specific location in the corpus."""
    doc_id: str = Field(..., description="Document identifier from corpus")
    page: Optional[int] = Field(None, description="Page number (1-indexed) for PDFs")
    section: Optional[str] = Field(None, description="Section heading if available")
    quote: str = Field(..., description="Exact quote from the source (max 200 chars)")


class Obligation(BaseModel):
    """A single regulatory obligation for a given activity."""
    obligation: str = Field(..., description="What must be done")
    trigger: Optional[str] = Field(None, description="Event that triggers the obligation")
    deadline: Optional[str] = Field(None, description="Timeframe or deadline (e.g., '1 hour', '24 hours')")
    notify_who: Optional[str] = Field(None, description="Who must be notified")
    required_content: Optional[str] = Field(None, description="What info must be included")
    citations: list[Citation] = Field(
        default_factory=list,
        min_length=1,
        description="At least one citation required per obligation"
    )
    evidence_artifacts: list[str] = Field(
        default_factory=list,
        description="What to show an auditor (incident report, email log, form, etc.)"
    )
    process_gates: list[str] = Field(
        default_factory=list,
        description="Internal approval/review steps (legal review, change control, etc.)"
    )


class StateObligations(BaseModel):
    """Obligations for a single state/jurisdiction."""
    obligations: list[Obligation] = Field(default_factory=list)
    confidence: str = Field(
        default="unknown",
        description="'high' if found clear evidence, 'low' if not found, 'partial' if conflicting"
    )
    not_found_explanation: Optional[str] = Field(
        None,
        description="If no obligations found, explain what was searched"
    )
    not_found_citations: list[Citation] = Field(
        default_factory=list,
        description="Citations showing absence of regulation (if applicable)"
    )


class DiffItem(BaseModel):
    """A single difference between jurisdictions."""
    field: str = Field(..., description="Which field differs (deadline, notify_who, etc.)")
    values: dict[str, str] = Field(..., description="State -> value mapping for this field")


class ActivityResponse(BaseModel):
    """
    Complete structured response for a regulatory activity query.
    This is the output schema the LLM must produce.
    """
    activity: str = Field(..., description="The activity being queried")
    states: dict[str, StateObligations] = Field(
        ...,
        description="Mapping of state codes to their obligations"
    )
    diffs: list[DiffItem] = Field(
        default_factory=list,
        description="Key differences between jurisdictions"
    )
    federal_baseline: Optional[StateObligations] = Field(
        None,
        description="Federal/NERC requirements that apply to all"
    )


# JSON schema for prompt injection
ACTIVITY_RESPONSE_SCHEMA = ActivityResponse.model_json_schema()
