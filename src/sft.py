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
from ..configs.configs import SFTScriptConfig

logger = logging.getLogger(__name__)


def create_training_args(
    output_dir: str,
    learning_rate: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    max_seq_length: int,
    eval_strategy: str = "no",
    eval_steps: int = 100,
    save_steps: int = 100,
    logging_steps: int = 10,
    warmup_ratio: float = 0.03,  # MentalChat16K paper
    weight_decay: float = 0.01,
    max_grad_norm: float = 0.3,  # MentalChat16K paper
    lr_scheduler_type: str = "cosine",
    optim: str = "paged_adamw_32bit",  # MentalChat16K paper
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    save_total_limit: int = 2,
    seed: int = 42,
    report_to: list = None,
    **kwargs
) -> SFTConfig:
    """
    Create SFT training arguments.
    
    Returns:
        SFTConfig for TRL SFTTrainer
    """
    return SFTConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        max_seq_length=max_seq_length,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,  # MentalChat16K paper
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        bf16=bf16,
        fp16=False,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_total_limit=save_total_limit,
        seed=seed,
        dataset_text_field="text",
        packing=False,
        report_to=report_to or ["wandb"],
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
        output_dir=config.training.output_dir,
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=config.training.num_train_epochs,
        max_seq_length=config.model.max_seq_length,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        logging_steps=config.training.logging_steps,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        lr_scheduler_type=config.training.lr_scheduler_type,
        optim=config.training.optim,
        bf16=config.training.bf16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        save_total_limit=config.training.save_total_limit,
        seed=config.training.seed,
        report_to=config.training.report_to,
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