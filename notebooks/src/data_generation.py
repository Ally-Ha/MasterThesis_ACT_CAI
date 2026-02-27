"""
Batch response generation for MC16K inputs using Azure OpenAI Batch API.
All tuneable parameters are in GenerationConfig (src/configs.py).
"""

import os
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from openai import AzureOpenAI
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

from .configs import GenerationConfig

GEN_CFG = GenerationConfig()

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPT_GENERIC = (
    "You are a mental health counseling therapist.\n"
    "Provide empathetic and respectful responses based on the user's concerns.\n"
    "Offer general support and coping suggestions where appropriate.\n"
    "If serious safety concerns are expressed, acknowledge them and "
    "recommend professional crisis resources.\n\n"
    "Provide your response in alignment with the instructions mentioned above.\n"
    "Keep responses well-rounded, focused and concise, between 150-300 words."
)

_CONSTITUTION = (
    "<constitution>\n"
    "For every response, you must follow these ACT principles exactly:\n"
    "- The response should increase willingness to experience emotions rather than attempting to control or eliminate them\n"
    "- The response should help the individual notice thoughts without believing or acting on them, recognizing thinking as an ongoing process rather than truth\n"
    "- The response should gently redirect from rumination about the past or worry about the future toward present-moment awareness and action\n"
    "- The response should help the individual adopt an observer perspective, noticing thoughts and feelings from a stable, continuous sense of self\n"
    "- The response should clarify the clients values as freely chosen directions for meaningful living\n"
    "- The response should encourage small, committed actions aligned with the users identified values\n"
    "</constitution>"
)

PROMPT_CONSTITUTION = (
    "You are a mental health counseling therapist specializing in "
    "Acceptance and Commitment Therapy (ACT).\n\n"
    f"{_CONSTITUTION}\n\n"
    "Provide your therapeutic response in alignment with the ACT constitution above.\n"
    "Keep responses well-rounded, focused and concise, between 150-300 words."
)

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

CSV_FIELDNAMES = [
    "timestamp", "run_id", "model", "reasoning_level",
    "instruction", "input", "output",
    "reasoning_summary",
    "tokens_input", "tokens_output", "tokens_reasoning", "tokens_total",
    "batch_id",
]

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("OPENAI_API_VERSION", "2025-04-01-preview"),
    )

def _build_batch_jsonl(
    instruction: str,
    inputs: List[str],
    cfg: GenerationConfig,
    request_jsonl_path: Path,
) -> Dict[str, str]:
    """
    Write the JSONL batch-request file and return a {custom_id: input_text} mapping.

    Schema per Azure OpenAI Batch docs:
      {"custom_id": str, "method": "POST", "url": "/chat/completions", "body": {...}}
    """
    request_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    input_mapping: Dict[str, str] = {}

    with open(request_jsonl_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(inputs):
            custom_id = f"item-{i}"
            input_mapping[custom_id] = text
            body = {
                "model": cfg.model,
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": text},
                ],
                "max_completion_tokens": cfg.max_completion_tokens,
                "reasoning": {
                    "effort": cfg.reasoning_level,
                    "summary": cfg.reasoning_summary,
                },
            }
            f.write(json.dumps({"custom_id": custom_id, "method": "POST", "url": "/chat/completions", "body": body}, ensure_ascii=False) + "\n")

    return input_mapping


def _collect_batch_results(
    client: AzureOpenAI,
    batch_id: str,
    instruction: str,
    input_mapping: Dict[str, str],
    output_csv: Path,
    cfg: GenerationConfig,
    logs_dir: Optional[str],
) -> List[Dict]:
    """Download the completed batch output file and write rows to CSV / JSON logs."""
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch_id} status is '{batch.status}', not 'completed'.")

    content = client.files.content(batch.output_file_id)
    all_metadata: List[Dict] = []

    for line in content.text.splitlines():
        if not line.strip():
            continue
        result = json.loads(line)
        custom_id = result["custom_id"]
        response_body = result.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        usage = response_body.get("usage", {})
        reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens")

        md = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": response_body.get("id"),
            "model": cfg.model,
            "reasoning_level": cfg.reasoning_level,
            "instruction": instruction,
            "input": input_mapping.get(custom_id, ""),
            "output": message.get("content"),
            # o3 reasoning summary (present when reasoning.summary != "none")
            "reasoning_summary": message.get("reasoning_content"),
            "tokens_input": usage.get("prompt_tokens"),
            "tokens_output": usage.get("completion_tokens"),
            "tokens_reasoning": reasoning_tokens,
            "tokens_total": usage.get("total_tokens"),
            "batch_id": batch_id,
        }
        all_metadata.append(md)
        _append_csv(output_csv, md)
        if logs_dir:
            _save_json_log(md, Path(logs_dir))

    return all_metadata


def _append_csv(csv_path: Path, row: Dict) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in CSV_FIELDNAMES})


def _save_json_log(row: Dict, logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = row["timestamp"].replace(":", "-")
    with open(logs_dir / f"log_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_batch_generation(
    instruction: str,
    inputs: List[str],
    output_csv: str,
    cfg: GenerationConfig = GEN_CFG,
    logs_dir: Optional[str] = None,
    wait_for_completion: bool = False,
    poll_interval_seconds: int = 60,
) -> List[Dict]:
    """
    Submit a list of patient inputs as an Azure OpenAI Batch job.

    By default returns immediately after submission (batch can take up to 24 h).
    Call retrieve_batch_generation() with the saved input_map JSON to collect
    results later, or set wait_for_completion=True for small pilots.
    """
    output_csv = Path(output_csv)
    if not inputs:
        return []

    client = _build_client()
    work_dir = output_csv.parent or Path(".")
    request_jsonl = work_dir / f"{output_csv.stem}_requests.jsonl"
    input_map_json = work_dir / f"{output_csv.stem}_input_map.json"

    input_mapping = _build_batch_jsonl(instruction, inputs, cfg, request_jsonl)

    with open(request_jsonl, "rb") as fh:
        uploaded = client.files.create(file=fh, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/chat/completions",
        completion_window="24h",
    )

    with open(input_map_json, "w", encoding="utf-8") as fh:
        json.dump({"batch_id": batch.id, "instruction": instruction, "inputs": input_mapping}, fh, indent=2, ensure_ascii=False)

    print(f"Batch submitted: {batch.id}  ({len(inputs)} requests)")
    print(f"  input map: {input_map_json}")

    if not wait_for_completion:
        print("Call retrieve_batch_generation() once the batch is complete.")
        return [{"batch_id": batch.id, "status": batch.status}]

    terminal = {"completed", "failed", "expired", "cancelled"}
    while batch.status not in terminal:
        time.sleep(poll_interval_seconds)
        batch = client.batches.retrieve(batch.id)
        print(f"  status: {batch.status}")

    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch.id} ended with status '{batch.status}'.")

    results = _collect_batch_results(client, batch.id, instruction, input_mapping, output_csv, cfg, logs_dir)
    print(f"Done — {len(results)} rows written to {output_csv}")
    return results


def retrieve_batch_generation(
    output_csv: str,
    input_map_json: str,
    cfg: GenerationConfig = GEN_CFG,
    logs_dir: Optional[str] = None,
) -> List[Dict]:
    """Retrieve results for a previously submitted batch and write to CSV / logs."""
    with open(input_map_json, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    client = _build_client()
    results = _collect_batch_results(
        client,
        batch_id=payload["batch_id"],
        instruction=payload["instruction"],
        input_mapping=payload["inputs"],
        output_csv=Path(output_csv),
        cfg=cfg,
        logs_dir=logs_dir,
    )
    print(f"Done — {len(results)} rows written to {output_csv}")
    return results

