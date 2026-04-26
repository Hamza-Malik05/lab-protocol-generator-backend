import json
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models import ExperimentPlan, FeedbackSubmission, HypothesisRequest, LiteratureQC
from services import (
    coerce_experiment_plan_payload,
    dummy_experiment_plan,
    dummy_literature_qc,
    get_semantic_expert_knowledge,
    invoke_bedrock_json,
    save_feedback_vector,
    tavily_search,
)

app = FastAPI(title="The AI Scientist API", version="1.0.0")

origins = [
    "https://lab-protocol-generator.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok", "service": "The AI Scientist API"}


@app.post("/api/literature-qc")
def literature_qc(payload: HypothesisRequest) -> Dict[str, Any]:
    tavily_results = tavily_search(
        query=f"Scientific protocols and prior work for: {payload.hypothesis}",
        include_domains=[
            "protocols.io",
            "nature.com/nprot",
            "jove.com",
            "arxiv.org",
            "bio-protocol.org",
            "ncbi.nlm.nih.gov",
            "nature.com",
            "cell.com",
            "sciencedirect.com",
            "biorxiv.org",
        ],
        max_results=5,
    )

    system_prompt = """
You are a scientific literature reviewer.
Return strictly valid JSON only (no markdown, no extra text).
The JSON must exactly match this schema:
{
  "novelty_signal": "not found | similar work exists | exact match found",
  "references": [{"title": "string", "url": "string", "source": "string"}]
}
Rules:
- novelty_signal must be exactly one of: "not found", "similar work exists", "exact match found"
- references must be grounded in the provided Tavily results
- CRITICAL LOGIC RULE: If you find and return 1 or more relevant references, the novelty_signal MUST be set to "similar work exists" or "exact match found". It CANNOT be "not found" if references are provided.
"""
    user_prompt = f"""
Hypothesis:
{payload.hypothesis}

Tavily Results:
{json.dumps(tavily_results, ensure_ascii=True)}
"""

    try:
        parsed = invoke_bedrock_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        validated = LiteratureQC.model_validate(parsed)
        return {"status": "success", "data": validated.model_dump()}
    except Exception as exc:
        print(f"Bedrock literature-qc failed: {exc}")
        fallback = dummy_literature_qc(payload.hypothesis)
        return {"status": "success", "data": fallback.model_dump()}


@app.post("/api/generate-plan")
def generate_plan(payload: HypothesisRequest) -> Dict[str, Any]:
    tavily_results = tavily_search(
        query=f"Reagents, catalog numbers, pricing, and validation protocols for: {payload.hypothesis}",
        include_domains=[
            # The Giants
            "thermofisher.com",
            "sigmaaldrich.com",
            "vwr.com",
            "fishersci.com",
            
            # The Specialists
            "atcc.org",
            "abcam.com",
            "neb.com",
            "qiagen.com",
            "biolegend.com",
            
            # QA, Protocols, and Validation
            "citeab.com",
            "biocompare.com",
            "protocols.io",
        ],
        max_results=8,
    )

    expert_context = get_semantic_expert_knowledge(payload.hypothesis, payload.domain)

    system_prompt = f"""
You are an expert Chief Scientific Officer designing an operationally realistic lab experiment plan.
{expert_context}
Return strictly valid JSON only (no markdown, no extra text).
The JSON must exactly match this schema:
{{
  "executive_summary": "string",
  "protocol_steps": [{{"step_number": 1, "title": "string", "description": "string", "duration_hours": 0.0}}],
  "materials_list": [{{"item_name": "string", "supplier": "string", "catalog_number": "string", "estimated_cost_usd": 0.0}}],
  "total_budget_usd": 0.0,
  "timeline_weeks": 0.0,
  "validation_approach": "string"
}}
Requirements:
- Include realistic suppliers, catalog numbers, and prices based on Tavily context.
- Ensure total_budget_usd is consistent with materials and operational plan.
- Keep steps practical and executable in a real lab setting.
- CRITICAL LOGIC RULE: For the validation_approach field, do not use generic phrases like "use biological replicates". You MUST explicitly name the exact assays, statistical tests (e.g., Two-way ANOVA), and measurable success thresholds that will be used to validate this specific hypothesis.
"""
    user_prompt = f"""
Hypothesis:
{payload.hypothesis}

Tavily Context:
{json.dumps(tavily_results, ensure_ascii=True)}
"""

    try:
        parsed = invoke_bedrock_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        parsed = coerce_experiment_plan_payload(parsed)
        validated = ExperimentPlan.model_validate(parsed)
        return {"status": "success", "data": validated.model_dump()}
    except Exception as exc:
        print(f"Bedrock generate-plan failed: {exc}")
        fallback = dummy_experiment_plan(payload.hypothesis)
        return {"status": "success", "data": fallback.model_dump()}


@app.post("/api/save-feedback")
def save_feedback(submission: FeedbackSubmission) -> Dict[str, str]:
    return save_feedback_vector(
        original_hypothesis=submission.original_hypothesis,
        domain=submission.domain,
        corrected_plan=submission.corrected_plan,
        scientist_notes=submission.scientist_notes,
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
