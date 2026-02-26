"""
Configuration dataclasses following alignment-handbook pattern.
All hyperparameters follow MentalChat16K paper (Xu et al., 2025).
"""
from dataclasses import dataclass, field
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv(".env")

# Model Presets - Datasets and Output Directory
MODEL_PRESETS = {
    "modelpilot":{
        "dataset": "ShenLab/MentalChat16K",
        "output_dir": "data/model_pilot-sft-llama-3.1-8b-instruct",
    },
    "model0": {
        "dataset": "ShenLab/MentalChat16K",
        "output_dir": "data/model_0-sft-llama-3.1-8b-instruct",
    },
    "model1": {
        "dataset": "data/ds_generic",
        "output_dir": "data/model_1-sft-llama-3.1-8b-instruct",
    },
    "model2": {
        "dataset": "data/ds_constitution",
        "output_dir": "data/model_2-sft-llama-3.1-8b-instruct",
    },
    "model3": {
        "dataset": "data/ds_constitution_revised",
        "output_dir": "data/model_3-sft-llama-3.1-8b-instruct",
    },
}


# Configuration Dataclasses
@dataclass
class ModelConfig:
    """Model configuration."""
    model_name_or_path: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    model_revision: str = "main"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    trust_remote_code: bool = False
    max_seq_length: int = 2048
    load_in_4bit: bool = True


@dataclass 
class LoraConfig:
    """LoRA/QLoRA configuration following MentalChat16K paper."""
    r: int = 64  
    lora_alpha: int = 16  
    lora_dropout: float = 0.1 
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 42


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_id: str = "ShenLab/MentalChat16K"
    dataset_config: Optional[str] = None
    dataset_split: str = "train"
    test_split_size: float = 0.2
    seed: int = 42
    num_proc: int = 4


@dataclass
class GenerationConfig:
    """Data generation configuration using AZURE OpenAI API."""
    model: str = "gpt-5.1"
    endpoint: str = "https://alina.openai.azure.com/"
    reasoning_level: str = "medium"
    verbosity_level: str = "high"
    max_completion_tokens: int = 500
    delay: float = 0.3
    # HuggingFace Hub
    hf_username: str = "AIforAlly"
    repo_generic: str = "mentalchat16k-generic-responses"
    repo_constitution: str = "mentalchat16k-constitution-responses"
    # Output paths
    output_dir: str = "data/responses/working_files/"
    generic_csv: str = "response_generic.csv"
    constitution_csv: str = "response_constitution.csv"


@dataclass
class TrainingConfig:
    """Training configuration following MentalChat16K paper."""
    output_dir: str = "data/sft-output"
    
    # Optimizer settings (MentalChat16K paper)
    learning_rate: float = 2.0e-4
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8  # Effective batch size: 64
    num_train_epochs: int = 5
    max_steps: int = -1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_32bit"
    
    # Evaluation and saving
    eval_strategy: str = "no"  # No eval during training for reproduction
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 2
    logging_steps: int = 10
    
    # Precision and efficiency
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    
    # Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    
    # Reporting
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    seed: int = 42


# Combined Configuration

@dataclass
class SFTScriptConfig:
    """Combined configuration for SFT script."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def for_model(cls, model_id: str) -> "SFTScriptConfig":
        """
        Get config for specific model experiment.
        
        Args:
            model_id: One of 'model0', 'model1', 'model2', 'model3'
        
        Returns:
            SFTScriptConfig with model-specific settings applied
        
        Example:
            config = SFTScriptConfig.for_model("model0")
        """
        if model_id not in MODEL_PRESETS:
            raise ValueError(f"Unknown model_id: {model_id}. Choose from {list(MODEL_PRESETS.keys())}")
        
        preset = MODEL_PRESETS[model_id]
        config = cls()
        config.data.dataset_id = preset["dataset"]
        config.training.output_dir = preset["output_dir"]
        return config
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)