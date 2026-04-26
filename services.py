import json
import math
import os
from typing import Any, Dict, List

import boto3
import requests
from botocore.config import Config
from dotenv import load_dotenv
from pinecone import Pinecone

from models import ExperimentPlan, LiteratureQC, Material, ProtocolStep, Reference

load_dotenv()

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

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("scientist-feedback")
else:
    pinecone_index = None


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
        inferenceConfig={"temperature": 0.0, "maxTokens": 8192},
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
            "maxTokens": 8192,
        },
    )
    raw_text = extract_text_from_bedrock_response(response)
    
    try:
        return parse_json_with_repair(raw_text)
    except Exception:
        return repair_json_with_bedrock(raw_text)


def get_semantic_expert_knowledge(hypothesis: str, domain: str) -> str:
    if not pinecone_index:
        return ""

    try:
        # Use search_records and move the filter INSIDE the query dict
        response = pinecone_index.search_records(
            namespace="__default__",
            query={
                "inputs": {"text": hypothesis},
                "top_k": 1,
                "filter": {"experiment_class": {"$eq": domain}}
            }
        )

        results = response if isinstance(response, dict) else response.to_dict()
        
        # The new records API nests hits under 'result' instead of 'results'
        hits = results.get("result", {}).get("hits", [])
        if not hits:
            return ""

        fields = hits[0].get("fields", {})

        knowledge_block = "\n\n### CRITICAL: SEMANTIC EXPERT MEMORY ###\n"
        knowledge_block += "You MUST incorporate the logic from this past expert review into your current plan if applicable:\n"
        knowledge_block += f"SIMILAR PAST HYPOTHESIS: {fields.get('hypothesis')}\n"
        knowledge_block += f"SCIENTIST REASONING FOR CORRECTION: {fields.get('notes')}\n"
        knowledge_block += f"CORRECTED PLAN STRUCTURE: {fields.get('corrected_plan')}\n---\n"
        return knowledge_block
    except Exception as exc:
        print(f"Pinecone retrieval failed: {exc}")
        return ""


def save_feedback_vector(
    *,
    original_hypothesis: str,
    domain: str,
    corrected_plan: Dict[str, Any],
    scientist_notes: str,
) -> Dict[str, str]:
    if not pinecone_index:
        return {"status": "error", "message": "Pinecone is not configured."}

    document_text = f"Hypothesis: {original_hypothesis} | Reasoning: {scientist_notes}"

    try:
        # Use __default__ namespace and _id
        pinecone_index.upsert_records(
            namespace="__default__",
            records=[
                {
                    "_id": f"id_{os.urandom(4).hex()}",
                    "chunk_text": document_text,
                    "experiment_class": domain,
                    "hypothesis": original_hypothesis,
                    "notes": scientist_notes,
                    "corrected_plan": json.dumps(corrected_plan),
                }
            ]
        )
        return {"status": "success", "message": "Feedback secured in cloud vector database."}
    except Exception as exc:
        print(f"Pinecone upsert failed: {exc}")
        return {"status": "error", "message": "Failed to save feedback."}


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
