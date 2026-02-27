"""
AI-as-Judge evaluation pipeline + analysis utilities.

Uses Claude-Opus via Azure AI Foundry for evaluation.
Questionnaire prompts are in questionnaires.py (unchanged).
Analysis utilities for scoring, statistics, and NLP metrics.
"""

import os
import re
import csv
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats
from openai import OpenAI
from dotenv import load_dotenv

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

from .configs import EvalConfig
from .questionnaires import (
    AI_JUDGE_PROMPT,
    ACT_SQ,
    MentalHealth16K_Metrics,
    FORMATTING_REMINDER,
    CLEANUP_PROMPT,
)

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK setup
# ---------------------------------------------------------------------------

def _ensure_nltk_data():
    """Download required NLTK data if not present."""
    required = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger',
                'averaged_perceptron_tagger_eng']
    for resource in required:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource
                          else f'taggers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

_ensure_nltk_data()


# ===================================================================
# AI-as-Judge evaluation (Claude-Opus via Azure AI Foundry)
# ===================================================================

def _build_eval_client() -> OpenAI:
    """
    Build client for Claude-Opus evaluation via Azure AI Foundry.

    Uses the GitHub Models / Azure AI Foundry pattern:
        base_url = "https://<resource>.services.ai.azure.com/openai/deployments/<deployment>"
    with a standard API key.
    """
    endpoint = os.getenv("AZURE_AI_EVAL_ENDPOINT")
    api_key = os.getenv("AZURE_AI_EVAL_API_KEY")
    if not endpoint or not api_key:
        raise EnvironmentError(
            "Set AZURE_AI_EVAL_ENDPOINT and AZURE_AI_EVAL_API_KEY in your .env file.\n"
            "These should point to your Claude-Opus deployment on Azure AI Foundry."
        )
    return OpenAI(base_url=endpoint, api_key=api_key)


def build_judge_prompt(input_text: str, response_text: str) -> str:
    """Build the full evaluation prompt with both questionnaires."""
    return (
        f"{AI_JUDGE_PROMPT}\n\n"
        f"{ACT_SQ}\n\n"
        f"{MentalHealth16K_Metrics}\n\n"
        f"{FORMATTING_REMINDER}\n\n"
        f"---\n"
        f"Patient Input:\n{input_text}\n\n"
        f"Therapist Response:\n{response_text}\n"
        f"---\n\n"
        f"Please evaluate the above therapist response using both assessment instruments."
    )


def parse_ratings(raw_output: str) -> Dict[str, Optional[float]]:
    """
    Parse numerical ratings from AI judge output.

    Returns dict with keys like 'item_MH16K_AL', 'item_ACT_SQ_Acceptance', etc.
    plus computed 'MH16K_score' (mean) and 'act_sq_total' (sum).
    """
    ratings: Dict[str, Optional[float]] = {}

    # MH16K items (1–10 scale)
    mh16k_items = ['MH16K_AL', 'MH16K_EV', 'MH16K_ST', 'MH16K_OMNJ',
                   'MH16K_CE', 'MH16K_BE', 'MH16K_HA']
    for item in mh16k_items:
        match = re.search(rf'{item}.*?Rating:\s*(\d+)', raw_output, re.IGNORECASE)
        ratings[f'item_{item}'] = float(match.group(1)) if match else None

    # ACT-SQ items (0–4 scale)
    actsq_items = ['ACT_SQ_Acceptance', 'ACT_SQ_Defusion', 'ACT_SQ_PresentMoment',
                   'ACT_SQ_Self-as-context', 'ACT_SQ_Values', 'ACT_SQ_ComAction']
    for item in actsq_items:
        match = re.search(rf'{item}.*?Rating:\s*(\d+)', raw_output, re.IGNORECASE)
        ratings[f'item_{item}'] = float(match.group(1)) if match else None

    # Computed totals
    mh16k_vals = [v for k, v in ratings.items() if k.startswith('item_MH16K') and v is not None]
    actsq_vals = [v for k, v in ratings.items() if k.startswith('item_ACT_SQ') and v is not None]
    ratings['MH16K_score'] = sum(mh16k_vals) / len(mh16k_vals) if mh16k_vals else None
    ratings['act_sq_total'] = sum(actsq_vals) if actsq_vals else None

    return ratings


def evaluate_single(
    client: OpenAI,
    model: str,
    input_text: str,
    response_text: str,
    cfg: EvalConfig,
) -> Dict:
    """Evaluate a single response using AI judge (Claude-Opus)."""
    prompt = build_judge_prompt(input_text, response_text)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )

    raw_output = resp.choices[0].message.content
    ratings = parse_ratings(raw_output)

    return {
        "raw_output": raw_output,
        "ratings": ratings,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        },
    }


def cleanup_ratings(client: OpenAI, model: str, raw_output: str, cfg: EvalConfig) -> Dict:
    """Use CLEANUP_PROMPT to extract ratings from messy output."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"{CLEANUP_PROMPT}\n\n{raw_output}"}],
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    return parse_ratings(resp.choices[0].message.content)


def run_ai_judge(
    inputs: List[str],
    responses: List[str],
    model_label: str = "modelpilot",
    cfg: EvalConfig = None,
    output_dir: str = None,
    logs_dir: str = None,
) -> pd.DataFrame:
    """
    Run AI judge evaluation on a batch of input-response pairs.

    Args:
        inputs: List of patient input texts
        responses: List of model response texts
        model_label: Label for this model (e.g., "modelpilot", "model0")
        cfg: Evaluation configuration (defaults to EvalConfig())
        output_dir: Directory to save results CSV
        logs_dir: Directory to save individual evaluation logs

    Returns:
        DataFrame with evaluation results
    """
    cfg = cfg or EvalConfig()
    output_dir = Path(output_dir or cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = _build_eval_client()
    results = []

    for i, (inp, resp_text) in enumerate(zip(inputs, responses)):
        logger.info(f"Evaluating {i+1}/{len(inputs)}: {inp[:60]}...")

        try:
            eval_result = evaluate_single(client, cfg.model, inp, resp_text, cfg)

            # If parsing failed on many items, try cleanup pass
            ratings = eval_result["ratings"]
            missing_count = sum(1 for v in ratings.values() if v is None)
            if missing_count > 2:
                logger.warning(f"  {missing_count} ratings missing, attempting cleanup...")
                ratings = cleanup_ratings(client, cfg.model, eval_result["raw_output"], cfg)

            row = {
                "index": i,
                "model_label": model_label,
                "input": inp,
                "response": resp_text,
                "raw_evaluation": eval_result["raw_output"],
                **ratings,
                "tokens_prompt": eval_result["usage"]["prompt_tokens"],
                "tokens_completion": eval_result["usage"]["completion_tokens"],
                "timestamp": datetime.utcnow().isoformat(),
            }
            results.append(row)

            # Save individual log
            if logs_dir:
                log_path = Path(logs_dir)
                log_path.mkdir(parents=True, exist_ok=True)
                with open(log_path / f"eval_{model_label}_{i:04d}.json", "w") as f:
                    json.dump(row, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"  Error evaluating item {i}: {e}")
            results.append({
                "index": i, "model_label": model_label,
                "input": inp, "response": resp_text, "error": str(e),
            })

    df = pd.DataFrame(results)
    csv_path = output_dir / f"eval_{model_label}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df)} evaluations to {csv_path}")

    return df


# ===================================================================
# Analysis utilities (used in 2_evaluation notebook)
# ===================================================================

def get_pillar_columns() -> Dict[str, str]:
    """Return mapping of ACT-SQ pillar names to column names."""
    return {
        'Acceptance': 'item_ACT_SQ_Acceptance',
        'Defusion': 'item_ACT_SQ_Defusion',
        'Present Moment': 'item_ACT_SQ_PresentMoment',
        'Self-as-context': 'item_ACT_SQ_Self-as-context',
        'Values': 'item_ACT_SQ_Values',
        'Committed Action': 'item_ACT_SQ_ComAction',
    }


def calculate_human_actsq_scores(row: pd.Series) -> Dict:
    """
    Calculate ACT-SQ scores from human Qualtrics survey data.
    Converts from 1-based to 0-based scale.
    """
    def get_score(q_num):
        val = pd.to_numeric(row.get(q_num, 0), errors='coerce') or 0
        return val - 1 if val > 0 else 0

    scores = {
        'item_ACT_SQ_Acceptance': get_score('Q1'),
        'item_ACT_SQ_Defusion': get_score('Q2'),
        'item_ACT_SQ_PresentMoment': get_score('Q3'),
        'item_ACT_SQ_Self-as-context': get_score('Q4'),
        'item_ACT_SQ_Values': get_score('Q5'),
        'item_ACT_SQ_ComAction': get_score('Q6'),
    }
    scores['act_sq_total'] = sum(scores.values())
    return scores


def compute_agreement_metrics(ai_scores: pd.Series, human_scores: pd.Series) -> Dict:
    """
    Compute agreement metrics between AI and human ratings.
    Returns correlation, mean difference, and t-test results.
    """
    metrics = {}
    metrics['pearson_r'], metrics['pearson_p'] = stats.pearsonr(ai_scores, human_scores)
    metrics['spearman_rho'], metrics['spearman_p'] = stats.spearmanr(ai_scores, human_scores)
    diff = ai_scores - human_scores
    metrics['mean_diff'] = diff.mean()
    metrics['std_diff'] = diff.std()
    metrics['t_stat'], metrics['t_p'] = stats.ttest_rel(ai_scores, human_scores)
    return metrics


def load_and_merge_data(evaluations_path: str, responses_path: str) -> pd.DataFrame:
    """Load evaluations and merge with response metadata."""
    evaluations = pd.read_csv(evaluations_path)
    metadata_cols = ['prompt_type', 'reasoning_level', 'input_text', 'input_group_id']
    responses = pd.read_csv(responses_path, usecols=metadata_cols)
    responses['index'] = responses.index
    return evaluations.merge(responses, on='index', how='left')


def compute_prompt_summary(df: pd.DataFrame, pillar_cols: List[str]) -> pd.DataFrame:
    """Compute summary statistics by prompt type and reasoning level."""
    score_cols = ['MH16K_score', 'act_sq_total']
    all_cols = score_cols + pillar_cols
    summary = df.groupby(['prompt_type', 'reasoning_level']).agg(
        {**{col: ['mean', 'std', 'count'] for col in all_cols}}
    ).round(2)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    return summary.reset_index()


# ===================================================================
# NLP text analysis
# ===================================================================

def compute_text_statistics(text: str) -> Dict:
    """Compute basic text statistics for a response."""
    if pd.isna(text) or not isinstance(text, str):
        return {
            'char_count': 0, 'word_count': 0, 'sentence_count': 0,
            'avg_word_length': 0, 'avg_sentence_length': 0,
            'unique_words': 0, 'lexical_diversity': 0,
        }
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    word_count = len(words)
    unique_words = len(set(words))
    return {
        'char_count': len(text),
        'word_count': word_count,
        'sentence_count': len(sentences),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'avg_sentence_length': word_count / len(sentences) if sentences else 0,
        'unique_words': unique_words,
        'lexical_diversity': unique_words / word_count if word_count > 0 else 0,
    }


def get_sentiment_scores(text: str, analyzer) -> Dict:
    """Get VADER sentiment scores for text."""
    if pd.isna(text) or not isinstance(text, str):
        return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
    return analyzer.polarity_scores(text)


def get_pos_distribution(text: str) -> Dict:
    """Get POS tag distribution for text."""
    if pd.isna(text) or not isinstance(text, str):
        return {'noun_ratio': 0, 'verb_ratio': 0, 'adj_ratio': 0, 'adv_ratio': 0}
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    total = len(pos_tags)
    if total == 0:
        return {'noun_ratio': 0, 'verb_ratio': 0, 'adj_ratio': 0, 'adv_ratio': 0}
    nouns = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
    verbs = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
    adjs = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
    advs = sum(1 for _, tag in pos_tags if tag.startswith('RB'))
    return {
        'noun_ratio': nouns / total,
        'verb_ratio': verbs / total,
        'adj_ratio': adjs / total,
        'adv_ratio': advs / total,
    }
