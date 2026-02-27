"""
SFT Training Package - Following HuggingFace alignment-handbook pattern
with Unsloth efficiency and Optuna HPO.

GPU-dependent modules (model, sft, data_handling) are imported lazily so that
lightweight modules (configs, data_generation, questionnaires) work everywhere,
including macOS laptops without NVIDIA GPUs.
"""
import importlib as _importlib

# ── Always-safe imports (no GPU / torch / unsloth needed) ──────────────
from .configs import (
    MODEL_PRESETS,
    OPENAI_API_VERSION,
    ModelConfig,
    LoraConfig,
    DataConfig,
    GenerationConfig,
    TrainingConfig,
    EvalConfig,
    SFTScriptConfig,
)

from .questionnaires import (
    AI_JUDGE_PROMPT,
    ACT_SQ,
    MentalHealth16K_Metrics,
    FORMATTING_REMINDER,
    CLEANUP_PROMPT,
)

# ── Lazy accessors for GPU-dependent symbols ───────────────────────────
def _lazy(module_attr: str):
    """Return a module-level lazy accessor for *module_attr* (e.g. '.model.get_model_and_tokenizer')."""
    parts = module_attr.rsplit(".", 1)
    mod_path, attr_name = parts[0], parts[1]

    def _get():
        mod = _importlib.import_module(mod_path, package=__name__)
        return getattr(mod, attr_name)
    return _get

_LAZY_MAP = {
    # .model
    "get_model_and_tokenizer": _lazy(".model.get_model_and_tokenizer"),
    "apply_peft":              _lazy(".model.apply_peft"),
    "prepare_for_inference":   _lazy(".model.prepare_for_inference"),
    # .data_handling
    "load_and_split_dataset":  _lazy(".data_handling.load_and_split_dataset"),
    "format_to_messages":      _lazy(".data_handling.format_to_messages"),
    "apply_chat_template":     _lazy(".data_handling.apply_chat_template"),
    "prepare_dataset":         _lazy(".data_handling.prepare_dataset"),
    # .sft
    "create_training_args":    _lazy(".sft.create_training_args"),
    "create_trainer":          _lazy(".sft.create_trainer"),
    "train":                   _lazy(".sft.train"),
    # .eval
    "run_ai_judge":            _lazy(".eval.run_ai_judge"),
    "evaluate_single":         _lazy(".eval.evaluate_single"),
    "parse_ratings":           _lazy(".eval.parse_ratings"),
    "build_judge_prompt":      _lazy(".eval.build_judge_prompt"),
    "get_pillar_columns":      _lazy(".eval.get_pillar_columns"),
    "compute_agreement_metrics": _lazy(".eval.compute_agreement_metrics"),
    "compute_text_statistics": _lazy(".eval.compute_text_statistics"),
}


def __getattr__(name: str):
    if name in _LAZY_MAP:
        return _LAZY_MAP[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Configs
    "MODEL_PRESETS",
    "OPENAI_API_VERSION",
    "ModelConfig",
    "LoraConfig",
    "DataConfig",
    "GenerationConfig",
    "TrainingConfig",
    "EvalConfig",
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
    # Evaluation
    "run_ai_judge",
    "evaluate_single",
    "parse_ratings",
    "build_judge_prompt",
    "get_pillar_columns",
    "compute_agreement_metrics",
    "compute_text_statistics",
    "AI_JUDGE_PROMPT",
    "ACT_SQ",
    "MentalHealth16K_Metrics",
    "FORMATTING_REMINDER",
    "CLEANUP_PROMPT",
]

__version__ = "0.1.0"
