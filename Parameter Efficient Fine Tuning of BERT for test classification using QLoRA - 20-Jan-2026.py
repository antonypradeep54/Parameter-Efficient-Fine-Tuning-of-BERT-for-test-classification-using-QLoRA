#!/usr/bin/env python3
"""
train.py

Parameter-efficient fine-tuning of BERT for binary classification using
4-bit quantization + LoRA (QLoRA-style).

Run from terminal:
  python train.py \
    --model_name bert-base-uncased \
    --dataset_name dipanjanS/imdb_sentiment_finetune_dataset20k \
    --output_dir ./qlora_bert_checkpoint \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 3 \
    --logging_steps 50 \
    --report_to wandb

Adjust batch sizes depending on GPU VRAM.

"""
import argparse
import os
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    #BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# Optional: bitsandbytes optimizer
try:
    import bitsandbytes as bnb
    from bitsandbytes.optim import AdamW8bit
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--dataset_name", type=str, default="dipanjanS/imdb_sentiment_finetune_dataset20k")
    parser.add_argument("--output_dir", type=str, default="./qlora_bert_out")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--report_to", type=str, default="none")  # set to "wandb" to log
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def compute_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model_name = "bert-base-uncased"


    print("Loading dataset:", args.dataset_name)
    ds = load_dataset(args.dataset_name)

    print(ds)

    # ---------------------------------------------------------
    # ✅ Robust column detection
    # ---------------------------------------------------------
    candidate_text_cols = ["text", "review", "sentence", "content"]
    candidate_label_cols = ["label", "labels", "sentiment", "target"]

    train_columns = ds["train"].column_names

    text_col = next((c for c in candidate_text_cols if c in train_columns), None)
    label_col = next((c for c in candidate_label_cols if c in train_columns), None)

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not infer text/label columns. Found columns: {train_columns}"
        )

    print(f"Using text column: '{text_col}', label column: '{label_col}'")

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------
    print("Loading tokenizer:", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def preprocess_fn(examples):
        return tokenizer(
            examples[text_col],
            truncation=True,
            padding=False,
            max_length=256,
        )

    tokenized = ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=[c for c in train_columns if c not in (text_col, label_col)],
    )

    tokenized = tokenized.rename_column(label_col, "labels")
    tokenized.set_format(type="torch")

    train_dataset = tokenized["train"]
    eval_dataset = tokenized["test"]

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ---------------------------------------------------------
    # ⚠️ QLoRA warning for BERT (important)
    # ---------------------------------------------------------
    print(
        "⚠️ NOTE: QLoRA is experimental for encoder-only models like BERT.\n"
        "If you encounter instability, consider fp16 LoRA instead."
    )
    
    # -------------------------------
    # Load BERT model in 4-bit + prepare for k-bit training
    # -------------------------------
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
      model_name,
      num_labels=num_labels,
    )

    model.to("cuda")


    # Setup LoRA (PEFT) config
    lora_r = 8
    lora_alpha = 32
    target_modules = ["query", "value"]  # common for transformer attention layers
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    print("Applying LoRA / PEFT...")
    model = get_peft_model(model, lora_config)

    # Show parameters counts
    total_params, trainable_params = compute_params(model)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.4f} % )")

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=1,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        warmup_ratio=0.03,
        weight_decay=0.01,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=args.report_to,
    )

    # Opt: use bitsandbytes 8-bit optimizer if available (lower CPU memory)
    optimizer = None
    if BNB_AVAILABLE:
        try:
            optimizer = AdamW8bit(model.parameters(), lr=args.learning_rate)
            print("Using bitsandbytes AdamW8bit optimizer")
        except Exception as e:
            print("Could not create AdamW8bit optimizer, falling back. Error:", e)
            optimizer = None

    # Metrics
    metric_acc = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return metric_acc.compute(predictions=preds, references=labels)

    print("Creating Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None) if optimizer is not None else (None, None),
    )

    # Print sample GPU memory (nvidia-smi) and PyTorch summary if GPU available
    if torch.cuda.is_available():
        try:
            print("nvidia-smi output (summary):")
            os.system("nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv -l 1 -n 1")
        except Exception:
            pass

    # Train
    print("Starting training...")
    trainer.train()

    print("Evaluating...")
    eval_res = trainer.evaluate()
    print("Eval results:", eval_res)

    # Save PEFT adapter weights only (small)
    print("Saving PEFT adapter to:", args.output_dir)
    model.save_pretrained(args.output_dir)

    print("Done.")

if __name__ == "__main__":
    main()
