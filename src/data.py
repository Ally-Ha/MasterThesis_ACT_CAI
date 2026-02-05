"""
Data loading and preprocessing following alignment-handbook pattern.
"""
import logging
from datasets import load_dataset, DatasetDict
from typing import Optional

logger = logging.getLogger(__name__)


def load_and_split_dataset(
    dataset_id: str,
    dataset_config: Optional[str] = None,
    dataset_split: str = "train",
    test_split_size: int = 0,
    seed: int = 42,
    **kwargs
) -> DatasetDict:
    """
    Load and split dataset following alignment-handbook pattern.
    
    Args:
        dataset_id: HuggingFace dataset ID
        dataset_config: Dataset configuration name
        dataset_split: Split to load
        test_split_size: Number of examples for test split (0 = no split)
        seed: Random seed for reproducibility
    
    Returns:
        DatasetDict with train (and optionally test) splits
    """
    logger.info(f"Loading dataset: {dataset_id}")
    
    dataset = load_dataset(
        dataset_id,
        dataset_config,
        split=dataset_split
    )
    
    if test_split_size > 0:
        logger.info(f"Splitting dataset with test_size={test_split_size}")
        split = dataset.train_test_split(test_size=test_split_size, seed=seed)
        return DatasetDict({
            'train': split['train'],
            'test': split['test']
        })
    
    return DatasetDict({'train': dataset})


def format_to_messages(example: dict) -> dict:
    """
    Convert dataset format to chat messages.
    Expected input columns: instruction, input, output
    """
    messages = [
        {"role": "system", "content": example.get('instruction', '')},
        {"role": "user", "content": example.get('input', '')},
        {"role": "assistant", "content": example.get('output', '')}
    ]
    return {"messages": messages}


def apply_chat_template(example: dict, tokenizer) -> dict:
    """
    Apply tokenizer's chat template to messages.
    
    Args:
        example: Dict with 'messages' key
        tokenizer: HuggingFace tokenizer with chat_template
    
    Returns:
        Dict with 'text' key containing formatted conversation
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}


def prepare_dataset(dataset: DatasetDict, tokenizer, num_proc: int = 4) -> DatasetDict:
    """
    Prepare dataset for SFT training.
    
    Args:
        dataset: Raw dataset with instruction/input/output columns
        tokenizer: Tokenizer for chat template
        num_proc: Number of processes for mapping
    
    Returns:
        Processed dataset with 'text' column ready for SFTTrainer
    """
    logger.info("Formatting dataset to messages...")
    dataset = dataset.map(format_to_messages, num_proc=num_proc)
    
    logger.info("Applying chat template...")
    dataset = dataset.map(
        lambda x: apply_chat_template(x, tokenizer),
        num_proc=num_proc
    )
    
    return dataset