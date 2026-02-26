"""
Constitutional AI Critique-Revision Pipeline.
Loads ds_constitution, applies N rounds of critique→revise using random principles.
"""
import os
import json
import random
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
from datasets import load_from_disk
from dotenv import load_dotenv
from openai import AzureOpenAI

from src.configs import OPENAI_API_VERSION

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent


@dataclass
class Config:
    constitution_path: str = str(BASE_DIR / "configs" / "constitution.json")
    dataset_path: str = str(BASE_DIR / "notebooks" / "data" / "responses" / "final" / "ds_constitution")
    output_dir: str = "data/critique_revision/"
    n_rounds: int = 3
    max_completion_tokens: int = 16000
    reasoning_effort: str = "medium"
    max_samples: int = -1
    seed: int = 42


class ConstitutionalAI:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.constitution = json.load(open(cfg.constitution_path))["constitutions"]
        random.seed(cfg.seed)
        self.client = AzureOpenAI(
            api_version=OPENAI_API_VERSION,
            azure_endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
        self.model = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", "o4-mini")

    def _call(self, messages: List[Dict]) -> Dict[str, Any]:
        """Call reasoning model, return content + reasoning + usage."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.cfg.max_completion_tokens,
            reasoning_effort=self.cfg.reasoning_effort,
        )
        return {
            "content": resp.choices[0].message.content,
            "reasoning": getattr(resp.choices[0].message, "reasoning", None),
            "usage": {"prompt": resp.usage.prompt_tokens, "completion": resp.usage.completion_tokens},
        }

    def critique_revise(self, user_input: str, response: str, principle: Dict, history: List[Dict]) -> Dict:
        """Single critique→revise step for one principle."""
        # Critique
        crit_prompt = f"""Critique this response for ACT alignment.

User: {user_input}
Response: {response}
Principle: {principle['principle']}

{principle['critic']}"""
        
        history.append({"role": "user", "content": crit_prompt})
        crit = self._call(history)
        history.append({"role": "assistant", "content": crit["content"]})

        # Revise
        rev_prompt = f"""Revise based on the critique.

{principle['revision']}

Output ONLY the revised response."""
        
        history.append({"role": "user", "content": rev_prompt})
        rev = self._call(history)
        history.append({"role": "assistant", "content": rev["content"]})

        return {
            "principle": principle["principle"],
            "critique": crit["content"],
            "critique_reasoning": crit["reasoning"],
            "revision": rev["content"],
            "revision_reasoning": rev["reasoning"],
        }

    def process(self, user_input: str, init_response: str, row_idx: int) -> Dict:
        """N rounds of critique-revision with random principle sampling."""
        history = [
            {"role": "system", "content": "You are a Constitutional AI assistant for ACT therapy."},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": init_response},
        ]
        
        current = init_response
        rounds = []
        
        for r in range(self.cfg.n_rounds):
            principle = random.choice(self.constitution)
            step = self.critique_revise(user_input, current, principle, history)
            current = step["revision"]
            rounds.append(step)
            logger.info(f"  Round {r+1}: {principle['principle'][:50]}...")

        return {
            "row_idx": row_idx,
            "input": user_input,
            "init_response": init_response,
            "revision_response": current,
            "rounds": rounds,
            "conversation": history,
        }


def run_pipeline(cfg: Config = None, output_file: str = "cai_results") -> List[Dict]:
    """Run CAI pipeline on ds_constitution dataset."""
    cfg = cfg or Config()
    ds = load_from_disk(cfg.dataset_path)
    if cfg.max_samples > 0:
        ds = ds.select(range(min(cfg.max_samples, len(ds))))
    
    cai = ConstitutionalAI(cfg)
    results = []
    
    for i, sample in enumerate(ds):
        logger.info(f"Processing {i+1}/{len(ds)}")
        results.append(cai.process(sample["input"], sample["output"], sample["row_idx"]))
    
    # Save outputs
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / f"{output_file}_traces.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # SFT + preference data
    sft = [{"prompt": r["input"], "response": r["revision_response"]} for r in results]
    pref = [{"prompt": r["input"], "chosen": r["revision_response"], "rejected": r["init_response"]} for r in results]
    pd.DataFrame(sft).to_json(out_dir / f"{output_file}_sft.jsonl", orient="records", lines=True)
    pd.DataFrame(pref).to_json(out_dir / f"{output_file}_pref.jsonl", orient="records", lines=True)
    
    logger.info(f"Saved {len(results)} results to {out_dir}")
    return results


if __name__ == "__main__":
    results = run_pipeline(Config(n_rounds=3, max_samples=5))
