# BERT Sentiment Classification with 4-bit Quantization and LoRA

This repository explores parameter-efficient fine-tuning for binary sentiment classification using Hugging Face Transformers and PEFT. The training script is designed to load a BERT sequence-classification model with bitsandbytes 4-bit quantization and train LoRA adapters.

> **Experiment status:** a GPU smoke test of the training pipeline completed successfully in Google Colab. This verifies that the script can load the dataset and quantized model, train briefly, evaluate, and save the adapter. It is not a full training run and does not establish model quality, memory savings, or speed improvements.

## Files

- `train.py` — training script
- `requirements.txt` — Python dependencies

## Approach

1. Load a dataset with a `train` split.
2. If the dataset has no `validation` split, create a reproducible 10% validation split from `train` using the configured seed. Any existing `test` split remains untouched.
3. Infer the text and label columns from common names.
4. Tokenize text using the selected Hugging Face tokenizer.
5. Load the sequence-classification model with bitsandbytes 4-bit NF4 quantization and double quantization.
6. Prepare the quantized model for k-bit training.
7. Add LoRA adapters to the BERT attention query and value projections. The classifier head is included in `modules_to_save` for task-specific training.
8. Use only the validation split for training-time evaluation and best-checkpoint selection.
9. After training and checkpoint selection are complete, evaluate the selected model once on the separate held-out test split, if one exists, then save the adapter and tokenizer.

QLoRA combines a quantized base model with trainable low-rank adapters. LoRA without quantization is a related but different setup; a configuration object alone does not make a run QLoRA.

## Requirements

- Python 3.10 or 3.11 recommended
- CUDA-capable NVIDIA GPU and a compatible CUDA/PyTorch/bitsandbytes installation
- Internet access for downloading the model and dataset on first use

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Important:** install the PyTorch build appropriate for your CUDA environment if the default package resolution does not provide a compatible build. See the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) and [bitsandbytes documentation](https://huggingface.co/docs/bitsandbytes/en/installation).

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

Adjust batch sizes to fit the available GPU memory. The script stops with a clear error if CUDA is unavailable. It chooses BF16 compute when the GPU reports support for it; otherwise it uses FP16.

To enable Weights & Biases logging, install the optional `wandb` dependency, authenticate with Weights & Biases, and pass `--report_to wandb`.

## GPU smoke test

A short end-to-end smoke test was run in Google Colab on an NVIDIA Tesla T4 using PyTorch 2.11.0+cu130, Transformers 4.57.6, Accelerate 1.15.0, Datasets 4.8.5, PEFT 0.21.2, and bitsandbytes 0.50.2.

- Dataset: `dipanjanS/imdb_sentiment_finetune_dataset20k`
- Smoke-test subset: 8 training examples and 4 validation examples
- Training: 2 steps completed
- Validation loss: approximately 0.5757
- Validation accuracy: 0.75 on just 4 examples
- Output: PEFT adapter and tokenizer saved successfully

The four-example validation result is too small to estimate generalization performance. Treat this as pipeline verification only; run a full training and a separate final test-set evaluation before reporting model-quality results. The smoke test used temporary local test settings and did not change the repository's default training duration.

## Evaluation and outputs

The script reports parameter counts and validation metrics returned by the Trainer. If the dataset does not provide a `validation` split, the script creates one by splitting 10% from `train` with the configured seed. This is used for training-time evaluation and best-checkpoint selection; an existing `test` split is not used for either purpose. After training, the selected model is evaluated on `test` once for final reporting, if that split exists. If the dataset has no separate `test` split, the script reports that final test evaluation was skipped. The script saves the PEFT adapter and tokenizer under the selected output directory.

Record actual run details before publishing results, including the GPU model, CUDA/PyTorch versions, model and dataset revision, batch size, training duration, evaluation metrics, and any observed memory usage. Do not treat a configured run as evidence of measured performance.

## Limitations

- This is an experimental QLoRA-style setup using encoder-only BERT, not a general-purpose large language model.
- The script assumes a binary classification task with two labels.
- Dataset columns must use a recognized text name (`text`, `review`, `sentence`, or `content`) and label name (`label`, `labels`, `sentiment`, or `target`).
- Quantization and training compatibility depend on the installed hardware and library versions.
- Results may vary by dataset, random seed, model, and training configuration.

## References

- [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)
- [bitsandbytes documentation](https://huggingface.co/docs/bitsandbytes/en/integrations)
