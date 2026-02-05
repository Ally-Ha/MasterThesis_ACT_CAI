"""
Configuration dataclasses following alignment-handbook pattern.
"""
from dataclasses import dataclass, field
from typing import Any, Optional, List
import yaml


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name_or_path: str = "unsloth/llama-3.1-8b-instruct-bnb-4bit"
    model_revision: str = "main"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    trust_remote_code: bool = False
    max_seq_length: int = 2048
    load_in_4bit: bool = True


@dataclass 
class LoraConfig:
    """LoRA/QLoRA configuration."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
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
    dataset_id: str = "ShenLab/MentalHealth16K"
    dataset_config: Optional[str] = "default"
    dataset_split: str = "train"
    test_split_size: int = 800
    seed: int = 42
    num_proc: int = 12


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "data/llama-3.1-8b-instruct-sft-pilot"
    learning_rate: float = 2.0e-5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    max_steps: int = -1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    
    # Evaluation and saving
    eval_strategy: str = "epoch"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 2
    logging_steps: int = 5
    
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


@dataclass
class SFTScriptConfig:
    """Combined configuration for SFT script."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SFTScriptConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            lora=LoraConfig(**config_dict.get('lora', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)
