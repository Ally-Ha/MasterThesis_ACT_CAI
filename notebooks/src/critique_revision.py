"""
Critique-and-Revision data generation for Constitutional AI.
Adapted from: https://github.com/huggingface/llm-swarm

Columns produced per row:
  - input              : original user query
  - init_response      : original model output (from ds_constitution)
  - critic_principle   : the principle text used
  - critic_prompt      : the critique prompt sent
  - critic_response    : the LLM critique
  - revision_prompt    : the revision prompt sent
  - revision_response  : the revised output
"""
import asyncio
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset
from huggingface_hub import AsyncInferenceClient
from llm_swarm import LLMSwarm, LLMSwarmConfig
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer

#Config 
@dataclass
class CritiqueRevisionConfig:
    """All tuneable knobs for a critique-revision run."""
    constitution_path: str = field(
        default_factory=lambda: str(Path(__file__).parent.parent / "configs" / "constitution.json")
    )
    tokenizer_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    max_new_tokens: int = 1500
    temperature: float = 1.0
    max_samples: int = -1          # -1 = use all rows
    stop_sequences: List[str] = field(
        default_factory=lambda: ["User:", "###", "<|endoftext|>"]
    )
    output_dir: str = "data"
    push_to_hub: bool = False
    repo_id: Optional[str] = None
    seed: int = 42


async def _process_one(
    idx: int,
    user_input: str,
    init_response: str,
    constitutions: List[dict],
    system_chat: List[dict],
    client: AsyncInferenceClient,
    tokenizer,
    semaphore: asyncio.Semaphore,
    cfg: CritiqueRevisionConfig,
) -> Dict:
    """Run critique → revision for a single (input, init_response) pair."""
    chat = system_chat.copy()

    # Seed the conversation with the original exchange
    chat.append({"role": "user", "content": user_input})
    chat.append({"role": "assistant", "content": init_response})

    constitution = random.choice(constitutions)
    row = {
        "input": user_input,
        "init_response": init_response,
        "critic_principle": constitution.get("principle", ""),
    }
    token_length = 0

    for prompt_text, prompt_key, response_key in [
        (constitution["critic"], "critic_prompt", "critic_response"),
        (constitution["revision"], "revision_prompt", "revision_response"),
    ]:
        async with semaphore:
            chat.append({"role": "user", "content": prompt_text})
            completion = await client.text_generation(
                prompt=tokenizer.apply_chat_template(chat, tokenize=False),
                max_new_tokens=cfg.max_new_tokens,
                stop_sequences=cfg.stop_sequences,
                temperature=cfg.temperature,
            )
            for stop_seq in cfg.stop_sequences:
                if completion.endswith(stop_seq):
                    completion = completion[: -len(stop_seq)].rstrip()
            chat.append({"role": "assistant", "content": completion})
            token_length += len(tokenizer.encode(completion))

        row[prompt_key] = prompt_text
        row[response_key] = completion

    row["_token_length"] = token_length
    row["_idx"] = idx
    return row


async def _run_pipeline(
    inputs: List[str],
    init_responses: List[str],
    constitutions: List[dict],
    system_chat: List[dict],
    swarm_cfg: LLMSwarmConfig,
    cfg: CritiqueRevisionConfig,
) -> List[Dict]:
    """Async entry point that spins up LLMSwarm and processes all rows."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    tokenizer.add_special_tokens(
        {"sep_token": "", "cls_token": "", "mask_token": "", "pad_token": "[PAD]"}
    )

    with LLMSwarm(swarm_cfg) as llm_swarm:
        semaphore = asyncio.Semaphore(llm_swarm.suggested_max_parallel_requests)
        client = AsyncInferenceClient(model=llm_swarm.endpoint)

        tasks = [
            _process_one(
                idx=i,
                user_input=inp,
                init_response=resp,
                constitutions=constitutions,
                system_chat=system_chat,
                client=client,
                tokenizer=tokenizer,
                semaphore=semaphore,
                cfg=cfg,
            )
            for i, (inp, resp) in enumerate(zip(inputs, init_responses))
        ]
        print(
            f"Launching critique-revision for {len(tasks)} rows …\n"
            "NOTE: first batch may take a while as the multi-turn context is built."
        )
        start = time.time()
        results = await tqdm_asyncio.gather(*tasks)
        elapsed = time.time() - start
        total_tokens = sum(r["_token_length"] for r in results)
        print(f"Done in {elapsed:.1f}s  |  {total_tokens / max(elapsed, 1):.0f} tok/s")

    return sorted(results, key=lambda r: r["_idx"])

#build final dataset
def _post_process(rows: List[Dict]) -> Dataset:
    """Convert raw result dicts into a clean HF Dataset."""
    for r in rows:
        r.pop("_token_length", None)
        r.pop("_idx", None)

    ds = Dataset.from_list(rows)

    def add_preference_cols(example):
        return {
            "prompt": example["input"].strip(),
            "messages": [
                {"role": "user", "content": example["input"].strip()},
                {"role": "assistant", "content": example["revision_response"].strip()},
            ],
            "chosen": [
                {"role": "user", "content": example["input"].strip()},
                {"role": "assistant", "content": example["revision_response"].strip()},
            ],
            "rejected": [
                {"role": "user", "content": example["input"].strip()},
                {"role": "assistant", "content": example["init_response"].strip()},
            ],
        }

    return ds.map(add_preference_cols)


def run_critique_revision(
    inputs: List[str],
    init_responses: List[str],
    cfg: CritiqueRevisionConfig = CritiqueRevisionConfig(),
    swarm_cfg: LLMSwarmConfig = LLMSwarmConfig(
        debug_endpoint="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
    ),
) -> Dataset:
    """
    End-to-end critique-revision pipeline.

    Args:
        inputs: list of user queries.
        init_responses: list of original model responses to critique.
        cfg: CritiqueRevisionConfig with constitution path, generation params, etc.
        swarm_cfg: LLMSwarmConfig – set ``debug_endpoint`` for quick runs
                   or configure Slurm for large-scale generation.

    Returns:
        HuggingFace Dataset with all intermediate columns plus
        ``chosen`` / ``rejected`` preference pairs.
    """
    random.seed(cfg.seed)

    with open(cfg.constitution_path) as f:
        data = json.load(f)
    constitutions = data["constitutions"]
    system_chat = [
        msg for sublist in data.get("system_chat", []) for msg in sublist
    ]

    n = len(inputs)
    if cfg.max_samples > 0:
        n = min(n, cfg.max_samples)
    inputs = inputs[:n]
    init_responses = init_responses[:n]

    raw_results = asyncio.run(
        _run_pipeline(inputs, init_responses, constitutions, system_chat, swarm_cfg, cfg)
    )
    return _post_process(raw_results)