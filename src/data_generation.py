"""
Response generation for MC16K inputs using OpenAI API.
All tuneable parameters are in GenerationConfig (src/configs.py).
"""

import os
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

from ..configs.configs import GenerationConfig

# default
GEN_CFG = GenerationConfig()

# Load the constitution from configs directory
_const_path = Path(__file__).parent.parent / "configs" / "constitution.json"
with open(_const_path, 'r') as f:
    _data = json.load(f)
principles = [c.get("principle", "").strip() for c in _data.get("constitutions", [])]
constitution = "\n\n".join(f"{i+1}. {p}" for i, p in enumerate(principles))

# Prompt templates (instruction text)
PROMPT_GENERIC = (
    "You are a mental health counseling therapist.\n"
    "Provide empathetic and respectful responses based on the user's concerns.\n"
    "Offer general support and coping suggestions where appropriate.\n"
    "If serious safety concerns are expressed, acknowledge them and "
    "recommend professional crisis resources.\n\n"
    "Provide your response in alignment with the instructions mentioned above.\n"
    "Keep responses well-rounded, focused and concise, between 150-300 words."
)

PROMPT_CONSTITUTION = (
    "You are a mental health counseling therapist specializing in "
    "Acceptance and Commitment Therapy (ACT).\n\n"
    "Please comply with the following principles:\n\n"
    f"<Constitution>\n{constitution}\n</Constitution>\n\n"
    "Provide your therapeutic response in alignment with the ACT constitution above.\n"
    "Keep responses well-rounded, focused and concise, between 150-300 words."
)


def generate_response(
    instruction: str,
    input_text: str,
    cfg: GenerationConfig = GEN_CFG,
) -> object:
    """Call OpenAI chat completions API and return the response object."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text}
    ]

    return client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        max_tokens=cfg.max_output_tokens,
        top_p=1,
        temperature=0.7,
    )


# Metadata to flat dict 
def extract_metadata(
    response: object,
    instruction: str,
    input_text: str,
    cfg: GenerationConfig = GEN_CFG,
) -> Dict:
    """Return a dict with columns matching the target dataset schema."""
    now = datetime.utcnow().isoformat()

    # Extract output from chat completions response
    output_text = None
    if hasattr(response, 'choices') and len(response.choices) > 0:
        output_text = response.choices[0].message.content

    md = {
        "timestamp": now,
        "run_id": getattr(response, "id", None),
        "model": cfg.model,
        "reasoning_level": cfg.reasoning_level,
        "verbosity_level": cfg.verbosity_level,
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
    }

    try:
        usage = response.usage
        md["tokens_input"] = usage.prompt_tokens
        md["tokens_output"] = usage.completion_tokens
        md["tokens_total"] = usage.total_tokens
    except Exception:
        md.update({"tokens_input": None, "tokens_output": None, "tokens_total": None})

    return md


# CSV helpers
CSV_FIELDNAMES = [
    "timestamp", "run_id", "model",
    "reasoning_level", "verbosity_level",
    "instruction", "input", "output",
]


def append_csv_response(csv_path: Path, metadata_dict: Dict) -> None:
    """Append one row to the CSV (creates file + header if needed)."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: metadata_dict.get(k) for k in CSV_FIELDNAMES})


def save_json_log(metadata_dict: Dict, logs_dir: Path) -> Path:
    """Save a single response as a JSON log file."""
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = metadata_dict["timestamp"].replace(":", "-")
    filepath = logs_dir / f"log_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(metadata_dict, f, indent=4)
    return filepath


# Batch Generation
def run_batch_generation(
    instruction: str,
    inputs: List[str],
    output_csv: str,
    cfg: GenerationConfig = GEN_CFG,
    logs_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Generate responses for every input with one instruction prompt.

    Args:
        instruction: System prompt text (becomes the `instruction` column).
        inputs: Patient input texts.
        output_csv: Where to append CSV rows.
        cfg: GenerationConfig (model, reasoning_level, verbosity_level …).
        logs_dir: Optional dir for per-response JSON logs.

    Returns:
        List of metadata dicts.
    """
    output_csv = Path(output_csv)
    all_metadata: List[Dict] = []
    total = len(inputs)

    for i, input_text in enumerate(inputs, 1):
        print(f"[{i}/{total}] Generating response …")

        response = generate_response(instruction=instruction, input_text=input_text, cfg=cfg)
        metadata = extract_metadata(response=response, instruction=instruction, input_text=input_text, cfg=cfg)

        append_csv_response(output_csv, metadata)
        if logs_dir is not None:
            save_json_log(metadata, Path(logs_dir))

        all_metadata.append(metadata)
        time.sleep(cfg.delay)

    print(f" Completed {total} generations → {output_csv}")
    return all_metadata