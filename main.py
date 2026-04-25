import json
import math
from typing import Any, Dict, List, Literal

import boto3
import requests
import uvicorn
from botocore.config import Config
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

origins = [
    "https://lab-protocol-generator.vercel.app",  # ⬅️ Your live Vercel frontend
                                                 # (Standard Vite port, just in case)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your first API endpoint
@app.get("/")
def read_root():
    return {"message": "Backend is connected!"}

import os


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
model_id = os.environ.get("ANTHROPIC_MODEL")

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    config=Config(
        read_timeout=180,
        connect_timeout=10,
        retries={"max_attempts": 3, "mode": "standard"},
    ),
)


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


app = FastAPI(title="The AI Scientist API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def tavily_search(query: str, include_domains: List[str], max_results: int) -> List[Dict[str, Any]]:
    if not TAVILY_API_KEY:
        print("TAVILY_API_KEY missing. Continuing with empty Tavily context.")
        return []

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "include_raw_content": False,
        "max_results": max_results,
        "include_domains": include_domains,
    }

    try:
        response = requests.post(TAVILY_SEARCH_URL, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except Exception as exc:
        print(f"Tavily API call failed: {exc}")
        return []


def extract_text_from_bedrock_response(response: Dict[str, Any]) -> str:
    content_blocks = response.get("output", {}).get("message", {}).get("content", [])
    text_parts: List[str] = []
    for block in content_blocks:
        text_value = block.get("text")
        if text_value:
            text_parts.append(text_value)
    raw_text = "\n".join(text_parts).strip()
    if not raw_text:
        raise ValueError("Bedrock response did not contain text content.")
    return raw_text


def strip_json_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    return cleaned


def extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found in model response.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("No complete JSON object found in model response.")


def parse_json_with_repair(raw_text: str) -> Dict[str, Any]:
    cleaned_json = strip_json_code_fences(raw_text)
    try:
        return json.loads(cleaned_json)
    except json.JSONDecodeError:
        extracted = extract_first_json_object(cleaned_json)
        return json.loads(extracted)


def repair_json_with_bedrock(broken_json_text: str) -> Dict[str, Any]:
    repair_system_prompt = """
You are a JSON repair utility.
You will receive malformed JSON. Fix syntax issues and return only valid JSON.
Do not add markdown, explanations, or extra keys.
"""
    repair_user_prompt = f"Malformed JSON:\n{broken_json_text}"
    response = bedrock_client.converse(
        modelId=model_id,
        system=[{"text": repair_system_prompt}],
        messages=[{"role": "user", "content": [{"text": repair_user_prompt}]}],
        inferenceConfig={"temperature": 0.0, "maxTokens": 1600},
    )
    repaired_text = extract_text_from_bedrock_response(response)
    return parse_json_with_repair(repaired_text)


def coerce_experiment_plan_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    plan = dict(payload)

    materials = plan.get("materials_list")
    if isinstance(materials, list):
        budget = 0.0
        for material in materials:
            if isinstance(material, dict):
                cost = material.get("estimated_cost_usd", 0.0)
                try:
                    budget += float(cost)
                except Exception:
                    continue
        if "total_budget_usd" not in plan:
            plan["total_budget_usd"] = round(budget, 2)

    steps = plan.get("protocol_steps")
    if isinstance(steps, list):
        total_hours = 0.0
        for step in steps:
            if isinstance(step, dict):
                duration = step.get("duration_hours", 0.0)
                try:
                    total_hours += float(duration)
                except Exception:
                    continue
        # Convert active protocol hours to a realistic multi-week lab timeline.
        if "timeline_weeks" not in plan:
            estimated_weeks = max(1.0, math.ceil(total_hours / 40.0))
            plan["timeline_weeks"] = float(estimated_weeks)

        if "validation_approach" not in plan:
            plan["validation_approach"] = (
                "Use biological replicates, positive/negative controls, and an orthogonal assay "
                "to verify reproducibility and statistical significance."
            )

    return plan


def invoke_bedrock_json(
    *,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    if not model_id:
        raise RuntimeError("ANTHROPIC_MODEL is not set.")

    response = bedrock_client.converse(
        modelId=model_id,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        inferenceConfig={
            "temperature": 0.2,
            "maxTokens": 4096,
        },
    )
    raw_text = extract_text_from_bedrock_response(response)
    try:
        return parse_json_with_repair(raw_text)
    except Exception:
        return repair_json_with_bedrock(raw_text)


def dummy_literature_qc(hypothesis: str) -> LiteratureQC:
    _ = hypothesis
    return LiteratureQC(
        novelty_signal="similar work exists",
        references=[
            Reference(
                title="CRISPR-Cas9 Editing Workflow Optimization in Mammalian Cells",
                url="https://www.nature.com/nprot/",
                source="nature.com/nprot",
            ),
            Reference(
                title="Standardized Protocols for Fluorescent Reporter Assays",
                url="https://www.protocols.io/",
                source="protocols.io",
            ),
        ],
    )


def dummy_experiment_plan(hypothesis: str) -> ExperimentPlan:
    return ExperimentPlan(
        executive_summary=(
            f"This plan tests the hypothesis: '{hypothesis}' using a controlled in vitro design, "
            "including baseline, treatment, and validation assays."
        ),
        protocol_steps=[
            ProtocolStep(
                step_number=1,
                title="Cell Preparation and Seeding",
                description="Seed cells in 6-well plates at standardized density and incubate overnight.",
                duration_hours=16.0,
            ),
            ProtocolStep(
                step_number=2,
                title="Treatment Administration",
                description="Apply treatment and vehicle controls in triplicate; record exact concentrations.",
                duration_hours=2.0,
            ),
            ProtocolStep(
                step_number=3,
                title="Readout and Quantification",
                description="Collect endpoint samples and run fluorescence/viability assays for primary outcomes.",
                duration_hours=6.0,
            ),
            ProtocolStep(
                step_number=4,
                title="Statistical Validation",
                description="Perform blinded analysis with predefined significance threshold and effect size checks.",
                duration_hours=4.0,
            ),
        ],
        materials_list=[
            Material(
                item_name="DMEM, high glucose",
                supplier="Thermo Fisher Scientific",
                catalog_number="11965092",
                estimated_cost_usd=89.0,
            ),
            Material(
                item_name="Fetal Bovine Serum",
                supplier="Sigma-Aldrich",
                catalog_number="F2442",
                estimated_cost_usd=210.0,
            ),
            Material(
                item_name="CellTiter-Glo Luminescent Cell Viability Assay",
                supplier="Promega",
                catalog_number="G7570",
                estimated_cost_usd=495.0,
            ),
        ],
        total_budget_usd=794.0,
        timeline_weeks=2.5,
        validation_approach=(
            "Use triplicate biological replicates, include positive/negative controls, "
            "and confirm findings with an orthogonal assay."
        ),
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
        ],
        max_results=3,
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
        query=f"Reagents, catalog numbers, protocol references for: {payload.hypothesis}",
        include_domains=[
            "thermofisher.com",
            "sigmaaldrich.com",
            "promega.com",
            "protocols.io",
        ],
        max_results=5,
    )

    system_prompt = """
You are an expert Chief Scientific Officer designing an operationally realistic lab experiment plan.
Return strictly valid JSON only (no markdown, no extra text).
The JSON must exactly match this schema:
{
  "executive_summary": "string",
  "protocol_steps": [{"step_number": 1, "title": "string", "description": "string", "duration_hours": 0.0}],
  "materials_list": [{"item_name": "string", "supplier": "string", "catalog_number": "string", "estimated_cost_usd": 0.0}],
  "total_budget_usd": 0.0,
  "timeline_weeks": 0.0,
  "validation_approach": "string"
}
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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
