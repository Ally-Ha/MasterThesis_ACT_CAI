"""
Model loading with Unsloth for efficient QLoRA training.
Following alignment-handbook pattern but using Unsloth backend.
"""
import logging
import torch
from typing import Tuple, Optional
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)


def get_model_and_tokenizer(
    model_name: str,
    max_seq_length: int = 2048,
    dtype: Optional[torch.dtype] = None,
    load_in_4bit: bool = True,
) -> Tuple[FastLanguageModel, any]:
    """
    Load model and tokenizer using Unsloth for efficiency.
    
    Args:
        model_name: HuggingFace model ID or Unsloth optimized model
        max_seq_length: Maximum sequence length
        dtype: Model dtype (None = auto-detect)
        load_in_4bit: Whether to use 4-bit quantization
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    logger.info(f"Model loaded successfully")
    return model, tokenizer


def apply_peft(
    model: FastLanguageModel,
    r: int = 64,  # MentalChat16K paper
    lora_alpha: int = 16,  # MentalChat16K paper
    lora_dropout: float = 0.1,  # MentalChat16K paper
    target_modules: list = None,
    bias: str = "none",
    use_gradient_checkpointing: str = "unsloth",
    random_state: int = 42,
) -> FastLanguageModel:
    """
    Apply PEFT/LoRA configuration using Unsloth's optimized implementation.
    
    Args:
        model: Base model from get_model_and_tokenizer
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout probability
        target_modules: List of modules to apply LoRA
        bias: Bias setting ("none", "all", "lora_only")
        use_gradient_checkpointing: Gradient checkpointing mode
        random_state: Random seed
    
    Returns:
        Model with PEFT applied
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    logger.info(f"Applying LoRA with r={r}, alpha={lora_alpha}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
    )
    
    return model


def prepare_for_inference(model: FastLanguageModel) -> FastLanguageModel:
    """
    Prepare model for inference (2x faster generation).
    
    Args:
        model: Trained model
    
    Returns:
        Model optimized for inference
    """
    FastLanguageModel.for_inference(model)
    return model

