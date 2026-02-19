"""
SFT Training Package - Following HuggingFace alignment-handbook pattern
with Unsloth efficiency and Optuna HPO.
"""
from ..configs.configs import (
    ModelConfig,
    LoraConfig, 
    DataConfig,
    GenerationConfig,
    TrainingConfig,
    SFTScriptConfig,
)
from .data_handling import (
    load_and_split_dataset,
    format_to_messages,
    apply_chat_template,
    prepare_dataset,
)
from .model import (
    get_model_and_tokenizer,
    apply_peft,
    prepare_for_inference,
)
from .sft import (
    create_training_args,
    create_trainer,
    train,
)

from .questionnaires import (
    AI_JUDGE_PROMPT, 
    ACT_SQ, 
    MentalHealth16K_Metrics, 
    FORMATTING_REMINDER, 
    CLEANUP_PROMPT, 
)

__all__ = [
    # Configs
    "ModelConfig",
    "LoraConfig",
    "DataConfig",
    "GenerationConfig",
    "TrainingConfig",
    "SFTScriptConfig",
    # Data
    "load_and_split_dataset",
    "format_to_messages",
    "apply_chat_template",
    "prepare_dataset",
    # Model
    "get_model_and_tokenizer",
    "apply_peft",
    "prepare_for_inference",
    # Training
    "create_training_args",
    "create_trainer",
    "train",
    #evaluation 
    "AI_JUDGE_PROMPT", 
    "ACT_SQ", 
    "MentalHealth16K_Metrics",
    "FORMATTING_REMINDER", 
    "CLEANUP_PROMPT", 
]

__version__ = "0.1.0"
