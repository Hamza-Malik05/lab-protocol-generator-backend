from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class Reference(BaseModel):
    title: str
    url: str
    source: str


class LiteratureQC(BaseModel):
    novelty_signal: Literal["not found", "similar work exists", "exact match found"] = Field(
        description="Must be 'not found', 'similar work exists', or 'exact match found'"
    )
    references: List[Reference]


class ProtocolStep(BaseModel):
    step_number: int
    title: str
    description: str
    duration_hours: float


class Material(BaseModel):
    item_name: str
    supplier: str
    catalog_number: str
    estimated_cost_usd: float


class ExperimentPlan(BaseModel):
    executive_summary: str
    protocol_steps: List[ProtocolStep]
    materials_list: List[Material]
    total_budget_usd: float
    timeline_weeks: float
    validation_approach: str = Field(
        description="CRITICAL: You MUST explicitly name the exact assays (e.g., ELISA), specific statistical tests (e.g., Two-way ANOVA, Student's t-test), and measurable success thresholds. DO NOT use generic phrases like 'use biological replicates'."
    )


class HypothesisRequest(BaseModel):
    hypothesis: str
    domain: str = "General Science"


class FeedbackSubmission(BaseModel):
    original_hypothesis: str
    domain: str
    corrected_plan: Dict[str, Any]
    scientist_notes: str
