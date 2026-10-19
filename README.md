# BERT Binary Classification with PEFT/LoRA

This repository contains a terminal-based experiment for fine-tuning BERT for binary sentiment classification using the Hugging Face Transformers and PEFT libraries.

## Files

- `train.py` — training script
- `requirements.txt` — Python dependencies

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run training

```bash
python train.py \
  --model_name bert-base-uncased \
  --dataset_name dipanjanS/imdb_sentiment_finetune_dataset20k \
  --output_dir ./qlora_bert_checkpoint \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --num_train_epochs 3 \
  --report_to none
```

Adjust batch sizes to suit the available hardware. The script currently moves the model to CUDA, so a compatible CUDA GPU is required for the present implementation.

## Important implementation note

Although this experiment was originally labelled QLoRA-style and includes a 4-bit quantization configuration, the current model-loading path does not pass that configuration to `from_pretrained`. Therefore, the script should **not be treated as verified 4-bit QLoRA training in its current state**. It applies PEFT/LoRA adapters, but the 4-bit quantization path needs to be implemented and tested separately before making QLoRA-specific claims.

The script reports evaluation metrics when training completes. No accuracy or memory-efficiency result is claimed here; results depend on the environment, dataset, and run configuration.
