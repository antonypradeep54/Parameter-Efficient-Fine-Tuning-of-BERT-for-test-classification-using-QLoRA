#!/usr/bin/env python3
"""Fine-tune BERT for binary sentiment classification with 4-bit quantization + LoRA.

This is an experimental QLoRA-style setup for an encoder-only model. It requires
a CUDA-capable NVIDIA GPU and compatible PyTorch/bitsandbytes installations.
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from evaluate import load as load_metric
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a sequence classifier with 4-bit quantization and LoRA."
    )
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="dipanjanS/imdb_sentiment_finetune_dataset20k",
    )
    parser.add_argument("--output_dir", type=str, default="./qlora_bert_out")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help='Set to "wandb" to enable Weights & Biases logging.',
    )
    return parser.parse_args()


def compute_params(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This QLoRA script requires a CUDA-capable NVIDIA GPU. "
            "Run it in a compatible GPU environment (for example, a CUDA-enabled "
            "Colab runtime) or implement a separate non-quantized LoRA workflow."
        )

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "bitsandbytes is required for 4-bit quantization. Install dependencies "
            "from requirements.txt in a compatible CUDA environment."
        ) from exc

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    use_bf16 = compute_dtype == torch.bfloat16

    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)
    if "train" not in dataset:
        raise ValueError(
            f"Dataset must contain a 'train' split. Available splits: {list(dataset.keys())}"
        )

    # Use validation for model selection; keep the test split for final reporting.
    eval_split = "validation" if "validation" in dataset else (
        "test" if "test" in dataset else None
    )
    if eval_split is None:
        raise ValueError(
            "Dataset must contain a 'test' or 'validation' split for evaluation."
        )

    candidate_text_columns = ["text", "review", "sentence", "content"]
    candidate_label_columns = ["label", "labels", "sentiment", "target"]
    train_columns = dataset["train"].column_names

    text_column = next(
        (column for column in candidate_text_columns if column in train_columns), None
    )
    label_column = next(
        (column for column in candidate_label_columns if column in train_columns), None
    )
    if text_column is None or label_column is None:
        raise ValueError(
            "Could not infer text/label columns. "
            f"Found columns: {train_columns}"
        )

    print(f"Using text column: '{text_column}', label column: '{label_column}'")
    print(f"Using evaluation split: '{eval_split}'")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def preprocess(examples):
        encoded = tokenizer(
            examples[text_column],
            truncation=True,
            padding=False,
            max_length=args.max_length,
        )
        encoded["labels"] = examples[label_column]
        return encoded

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=train_columns,
        desc="Tokenizing dataset",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model in 4-bit mode: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        quantization_config=quantization_config,
        device_map="auto",
    )

    # Prepare the quantized base model before adding trainable LoRA adapters.
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"],
        modules_to_save=["classifier"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    print("Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)

    total_params, trainable_params = compute_params(model)
    trainable_pct = 100 * trainable_params / total_params if total_params else 0
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_pct:.4f}%)")
    print(f"4-bit quantization configured: {getattr(model, 'is_loaded_in_4bit', False)}")
    print(f"Compute dtype: {compute_dtype}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=not use_bf16,
        bf16=use_bf16,
        warmup_ratio=0.03,
        weight_decay=0.01,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=args.report_to,
        seed=args.seed,
    )

    accuracy_metric = load_metric("accuracy")

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = logits.argmax(axis=-1)
        return accuracy_metric.compute(
            predictions=predictions,
            references=labels,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized[eval_split],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating on the selected evaluation split...")
    evaluation_results = trainer.evaluate()
    print("Evaluation results:", evaluation_results)

    # Evaluate the selected model on the held-out test split only after training
    # and model selection are complete. Never use test metrics to select a model.
    if "test" in tokenized and eval_split != "test":
        print("Evaluating final model on the held-out test split...")
        test_results = trainer.evaluate(
            eval_dataset=tokenized["test"],
            metric_key_prefix="test",
        )
        print("Final test results:", test_results)
    elif eval_split == "test":
        print(
            "No separate validation split was available; the test split was used "
            "for training-time evaluation, so it is not an independent final test."
        )
    else:
        print("No separate test split is available; skipping final test evaluation.")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving PEFT adapter and tokenizer to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Training run complete.")


if __name__ == "__main__":
    main()
