"""
Analysis utilities for ACT evaluation notebooks.
Shared functions for data processing, scoring, and statistical analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats


def get_pillar_columns() -> Dict[str, str]:
    """Return mapping of ACT-SQ pillar names to column names."""
    return {
        'Acceptance': 'item_ACT_SQ_Acceptance',
        'Defusion': 'item_ACT_SQ_Defusion',
        'Present Moment': 'item_ACT_SQ_PresentMoment',
        'Self-as-context': 'item_ACT_SQ_Self-as-context',
        'Values': 'item_ACT_SQ_Values',
        'Committed Action': 'item_ACT_SQ_ComAction'
    }


def calculate_human_actsq_scores(row: pd.Series) -> Dict:
    """
    Calculate ACT-SQ scores from human Qualtrics survey data.
    Converts from 1-based to 0-based scale.
    
    Args:
        row: DataFrame row with Q1-Q6 columns
    
    Returns:
        Dictionary with pillar and total scores
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
    Compute comprehensive agreement metrics between AI and human ratings.
    
    Args:
        ai_scores: AI evaluation scores
        human_scores: Human evaluation scores
    
    Returns:
        Dictionary with correlation, mean difference, and t-test results
    """
    metrics = {}
    
    # Correlation
    metrics['pearson_r'], metrics['pearson_p'] = stats.pearsonr(ai_scores, human_scores)
    metrics['spearman_rho'], metrics['spearman_p'] = stats.spearmanr(ai_scores, human_scores)
    
    # Mean difference
    diff = ai_scores - human_scores
    metrics['mean_diff'] = diff.mean()
    metrics['std_diff'] = diff.std()
    
    # Paired t-test
    metrics['t_stat'], metrics['t_p'] = stats.ttest_rel(ai_scores, human_scores)
    
    return metrics


def load_and_merge_data(evaluations_path, responses_path) -> pd.DataFrame:
    """
    Load evaluations and merge with response metadata.
    
    Args:
        evaluations_path: Path to evaluations CSV
        responses_path: Path to responses CSV
    
    Returns:
        Merged DataFrame
    """
    evaluations = pd.read_csv(evaluations_path)
    
    metadata_cols = ['prompt_type', 'reasoning_level', 'verbosity_level', 'input_text', 'input_group_id']
    responses = pd.read_csv(responses_path, usecols=metadata_cols)
    responses['index'] = responses.index
    
    df = evaluations.merge(responses, on='index', how='left')
    
    return df


def compute_prompt_summary(df: pd.DataFrame, pillar_cols: List[str]) -> pd.DataFrame:
    """
    Compute summary statistics by prompt type and reasoning level.
    
    Args:
        df: DataFrame with evaluations
        pillar_cols: List of pillar column names
    
    Returns:
        Summary DataFrame
    """
    score_cols = ['MH16K_score', 'act_sq_total']
    all_cols = score_cols + pillar_cols
    
    summary = df.groupby(['prompt_type', 'reasoning_level']).agg({
        **{col: ['mean', 'std', 'count'] for col in all_cols}
    }).round(2)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    return summary


# NLP
def compute_text_statistics(text):
    """Compute basic text statistics for a response."""
    if pd.isna(text) or not isinstance(text, str):
        return {
            'char_count': 0, 'word_count': 0, 'sentence_count': 0,
            'avg_word_length': 0, 'avg_sentence_length': 0,
            'unique_words': 0, 'lexical_diversity': 0
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
        'lexical_diversity': unique_words / word_count if word_count > 0 else 0
    }

# Sentiment
def get_sentiment_scores(text, analyzer):
    """Get VADER sentiment scores for text."""
    if pd.isna(text) or not isinstance(text, str):
        return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
    return analyzer.polarity_scores(text)

# POS Analysis
def get_pos_distribution(text):
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
        'adv_ratio': advs / total
    }