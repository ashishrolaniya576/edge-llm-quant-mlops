"""
QLoRA Instruction Fine-Tuning and Model Merge Pipeline.

Implements parameter-efficient fine-tuning (PEFT) with 4-bit NormalFloat (nf4) quantization,
gradient checkpointing, memory profiling callbacks, and automated FP16 adapter merging.
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s",
)
logger = logging.getLogger("edge_train")


class MemoryUsageCallback(TrainerCallback):
    """Callback to monitor and log CPU RSS RAM and CUDA VRAM usage across training steps."""

    def __init__(self) -> None:
        super().__init__()
        self.process = psutil.Process()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if state.global_step % max(1, args.logging_steps) == 0:
            rss_mb = self.process.memory_info().rss / (1024 * 1024)
            if torch.cuda.is_available():
                cuda_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
                cuda_max = torch.cuda.max_memory_allocated() / (1024 * 1024)
                logger.info(
                    "Step %d | Host RSS: %.2f MB | GPU Alloc: %.2f MB | GPU Peak: %.2f MB",
                    state.global_step,
                    rss_mb,
                    cuda_alloc,
                    cuda_max,
                )
            else:
                logger.info("Step %d | Host RSS: %.2f MB", state.global_step, rss_mb)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_quantization_config(
    use_cuda: bool, quant_cfg: Dict[str, Any]
) -> Optional[Any]:
    """Configure bitsandbytes 4-bit NormalFloat quantization when GPU is available."""
    if not use_cuda:
        logger.warning(
            "CUDA device not detected. BitsAndBytes 4-bit quantization requires a CUDA GPU. "
            "Falling back to standard float32 precision for development/testing execution."
        )
        return None

    try:
        from transformers import BitsAndBytesConfig

        compute_dtype_str = quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
        compute_dtype = (
            torch.bfloat16
            if compute_dtype_str == "bfloat16" and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        return BitsAndBytesConfig(
            load_in_4bit=quant_cfg.get("load_in_4bit", True),
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype,
        )
    except ImportError:
        logger.warning(
            "bitsandbytes package not available. Skipping 4-bit quantization config."
        )
        return None


def run_qlora_training(
    config: Dict[str, Any], data_path_override: Optional[str] = None
) -> str:
    """
    Execute QLoRA fine-tuning using Hugging Face Transformers, PEFT, and TRL.

    Args:
        config: Dictionary parsed from training_config.yaml.
        data_path_override: Optional path to JSONL data to override config.

    Returns:
        Path to the saved LoRA adapter directory.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    from src.data_loader import prepare_hf_dataset

    model_cfg = config.get("model", {})
    quant_cfg = config.get("quantization", {})
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    model_name = model_cfg.get("name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
    output_dir = train_cfg.get("output_dir", "./checkpoints/qlora_adapter")
    data_path = data_path_override or data_cfg.get(
        "train_data_path", "data/sample_logs.jsonl"
    )

    use_cuda = torch.cuda.is_available()
    device_map = "auto" if use_cuda else None
    bnb_config = build_quantization_config(use_cuda, quant_cfg)

    logger.info("Initializing Tokenizer from %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading Base Model: %s (CUDA Available: %s)", model_name, use_cuda)
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": model_cfg.get("trust_remote_code", True),
    }

    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = device_map
    else:
        # CPU or standard GPU fallback
        model_kwargs["torch_dtype"] = torch.float32 if not use_cuda else torch.float16
        if use_cuda:
            model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if use_cuda and bnb_config is not None:
        logger.info("Preparing model for k-bit training with gradient checkpointing...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        )

    logger.info(
        "Configuring LoRA Adapter: r=%d, alpha=%d",
        lora_cfg.get("r", 16),
        lora_cfg.get("lora_alpha", 32),
    )
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=lora_cfg.get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    logger.info("Preparing datasets from %s", data_path)
    train_dataset, eval_dataset = prepare_hf_dataset(
        data_path=data_path,
        tokenizer=None,  # Pass raw text formatting for SFTTrainer
        max_length=train_cfg.get("max_seq_length", 512),
        train_ratio=data_cfg.get("train_split_ratio", 0.85),
    )

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.3)),
        logging_steps=train_cfg.get("logging_steps", 5),
        eval_strategy=(
            "steps" if hasattr(TrainingArguments, "eval_strategy") else "steps"
        ),
        eval_steps=train_cfg.get("eval_steps", 25),
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 25),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        bf16=train_cfg.get("bf16", False)
        and use_cuda
        and torch.cuda.is_bf16_supported(),
        fp16=train_cfg.get("fp16", False) and use_cuda,
        optim=train_cfg.get("optim", "paged_adamw_8bit" if use_cuda else "adamw_torch"),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        report_to=train_cfg.get("report_to", ["tensorboard"]),
        seed=train_cfg.get("seed", 42),
        remove_unused_columns=False,
    )

    try:
        from trl import SFTTrainer

        trainer = SFTTrainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field=data_cfg.get("dataset_text_field", "text"),
            max_seq_length=train_cfg.get("max_seq_length", 512),
            tokenizer=tokenizer,
            callbacks=[MemoryUsageCallback()],
        )
    except ImportError:
        logger.warning(
            "TRL SFTTrainer not found. Falling back to standard HuggingFace Trainer."
        )
        from transformers import DataCollatorForLanguageModeling, Trainer

        from src.data_loader import prepare_hf_dataset as prep_tok

        train_tok, eval_tok = prep_tok(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=train_cfg.get("max_seq_length", 512),
            train_ratio=data_cfg.get("train_split_ratio", 0.85),
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=train_tok,
            eval_dataset=eval_tok,
            data_collator=data_collator,
            callbacks=[MemoryUsageCallback()],
        )

    logger.info("Beginning model training pipeline...")
    trainer.train()

    logger.info("Saving fine-tuned LoRA adapter weights to %s", output_dir)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir


def merge_lora_to_base_model(
    adapter_dir: str,
    base_model_name: str,
    output_dir: str,
    device: str = "cpu",
) -> str:
    """
    Merge fine-tuned LoRA adapter back into base FP16 model checkpoint for GGUF conversion.

    Args:
        adapter_dir: Directory containing adapter_model.safetensors and adapter_config.json.
        base_model_name: HuggingFace model repo name or local path.
        output_dir: Target directory for the merged standalone checkpoint.
        device: Target device for merging operations ('cpu' or 'cuda').

    Returns:
        Path to the merged model directory.
    """
    from peft import PeftModel

    logger.info("Commencing LoRA weight merge...")
    logger.info(
        "Base model: %s | Adapter: %s | Target: %s",
        base_model_name,
        adapter_dir,
        output_dir,
    )

    target_path = Path(output_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    torch_dtype = torch.float16
    logger.info("Loading base model in FP16 precision on %s...", device)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    logger.info("Attaching PEFT adapter weights from %s...", adapter_dir)
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    logger.info("Executing merge_and_unload() operation...")
    merged_model = model.merge_and_unload()

    logger.info(
        "Exporting merged FP16 model and tokenizer artifacts to %s...", output_dir
    )
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # Sanitize tokenizer_config.json to ensure compatibility with transformers and llama.cpp
    tok_cfg_path = target_path / "tokenizer_config.json"
    if tok_cfg_path.is_file():
        try:
            import json

            with open(tok_cfg_path, "r", encoding="utf-8") as f:
                tok_cfg = json.load(f)
            if isinstance(tok_cfg.get("extra_special_tokens"), list):
                tok_cfg["extra_special_tokens"] = {}
                with open(tok_cfg_path, "w", encoding="utf-8") as f:
                    json.dump(tok_cfg, f, indent=2)
                logger.info(
                    "Normalized extra_special_tokens in %s to dictionary format.",
                    tok_cfg_path,
                )
        except Exception as exc:
            logger.warning("Could not sanitize tokenizer_config.json: %s", exc)

    # Clean memory
    del base_model
    del model
    del merged_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        "Merged model checkpoint successfully exported to: %s", target_path.resolve()
    )
    return str(target_path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QLoRA Instruction Tuning and FP16 Model Merging Pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration YAML file.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional data path override.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip training and perform LoRA merge into base model only.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Perform training but skip post-training FP16 merge routine.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    base_model_name = model_cfg.get("name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter_dir = train_cfg.get("output_dir", "./checkpoints/qlora_adapter")
    merged_output_dir = train_cfg.get("merged_output_dir", "./checkpoints/merged_model")

    if args.merge_only:
        logger.info("Running in --merge-only mode.")
        merge_lora_to_base_model(
            adapter_dir=adapter_dir,
            base_model_name=base_model_name,
            output_dir=merged_output_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        return

    # Run fine-tuning
    adapter_output = run_qlora_training(config, data_path_override=args.data_path)

    # Run merge unless skipped
    if not args.skip_merge:
        logger.info("Proceeding with post-training model merge...")
        merge_lora_to_base_model(
            adapter_dir=adapter_output,
            base_model_name=base_model_name,
            output_dir=merged_output_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )


if __name__ == "__main__":
    main()
