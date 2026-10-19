# BERT Sentiment Classification with 4-bit Quantization and LoRA

This repository explores parameter-efficient fine-tuning for binary sentiment classification using Hugging Face Transformers and PEFT. The training script is designed to load a BERT sequence-classification model with bitsandbytes 4-bit quantization and train LoRA adapters.

> **Experiment status:** the code now includes the intended 4-bit quantization path and k-bit preparation. A successful training run has not been verified as part of this repository update, so no accuracy, memory reduction, or speed improvement is claimed.

## Files

- `train.py` — training script
- `requirements.txt` — Python dependencies

## Approach

1. Load a dataset with a `train` split and a `test` or `validation` split.
2. Infer the text and label columns from common names.
3. Tokenize text using the selected Hugging Face tokenizer.
4. Load the sequence-classification model with bitsandbytes 4-bit NF4 quantization and double quantization.
5. Prepare the quantized model for k-bit training.
6. Add LoRA adapters to the BERT attention query and value projections. The classifier head is included in `modules_to_save` for task-specific training.
7. Train and evaluate with the Hugging Face `Trainer`, then save the adapter and tokenizer.

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

## Evaluation and outputs

The script reports parameter counts, evaluation accuracy, and the evaluation metrics returned by the Trainer. It saves the PEFT adapter and tokenizer under the selected output directory.

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
