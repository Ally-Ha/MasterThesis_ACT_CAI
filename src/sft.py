"""
SFT training with Optuna hyperparameter optimization.
Following alignment-handbook pattern with Unsloth efficiency.
"""
import logging
import os
from typing import Optional, Dict, Any

from trl import SFTTrainer, SFTConfig
from transformers import set_seed

from .model import get_model_and_tokenizer, apply_peft
from .data_handling import load_and_split_dataset, prepare_dataset
from .configs import SFTScriptConfig

logger = logging.getLogger(__name__)


def create_training_args(
    training_cfg,  # TrainingConfig
    max_seq_length: int,
) -> SFTConfig:
    """
    Create SFT training arguments from TrainingConfig.
    
    Args:
        training_cfg: TrainingConfig dataclass instance
        max_seq_length: From ModelConfig (needed for SFTConfig)
    
    Returns:
        SFTConfig for TRL SFTTrainer
    """
    return SFTConfig(
        output_dir=training_cfg.output_dir,
        learning_rate=training_cfg.learning_rate,
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=training_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        num_train_epochs=training_cfg.num_train_epochs,
        max_seq_length=max_seq_length,
        eval_strategy=training_cfg.eval_strategy,
        save_strategy=training_cfg.save_strategy,
        save_steps=training_cfg.save_steps,
        logging_steps=training_cfg.logging_steps,
        warmup_ratio=training_cfg.warmup_ratio,
        weight_decay=training_cfg.weight_decay,
        max_grad_norm=training_cfg.max_grad_norm,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        optim=training_cfg.optim,
        bf16=training_cfg.bf16,
        fp16=training_cfg.fp16,
        gradient_checkpointing=training_cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_total_limit=training_cfg.save_total_limit,
        seed=training_cfg.seed,
        dataset_text_field="text",
        packing=False,
        report_to=training_cfg.report_to,
    )


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_args: SFTConfig,
) -> SFTTrainer:
    """
    Create SFT trainer following alignment-handbook pattern.
    """
    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )


def train(
    config: SFTScriptConfig
) -> Dict[str, Any]:
    """
    Main training function.
    
    Args:
        config: Full SFT configuration
        trial: Optional Optuna trial for hyperparameter search
    
    Returns:
        Dictionary with training results
    """
    set_seed(config.training.seed)
    
    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(
        model_name=config.model.model_name_or_path,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
    )
    
    # Apply PEFT
    model = apply_peft(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        use_gradient_checkpointing=config.lora.use_gradient_checkpointing,
        random_state=config.lora.random_state,
    )
    
    # Load and prepare dataset
    dataset = load_and_split_dataset(
        dataset_id=config.data.dataset_id,
        dataset_config=config.data.dataset_config,
        dataset_split=config.data.dataset_split,
        test_split_size=config.data.test_split_size,
        seed=config.data.seed,
    )
    dataset = prepare_dataset(dataset, tokenizer, num_proc=config.data.num_proc)
    
    # Create training arguments
    training_args = create_training_args(
        training_cfg=config.training,
        max_seq_length=config.model.max_seq_length,
    )
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        training_args=training_args,
    )
    
    # Train
    logger.info("*** Starting training ***")
    train_result = trainer.train()
    
    # Evaluate
    metrics = {}
    if dataset.get("test") is not None:
        eval_metrics = trainer.evaluate()
        metrics["eval_loss"] = eval_metrics["eval_loss"]
    
    metrics["train_loss"] = train_result.training_loss
    
    return {
        "trainer": trainer,
        "model": model,
        "tokenizer": tokenizer,
        "metrics": metrics,
    }