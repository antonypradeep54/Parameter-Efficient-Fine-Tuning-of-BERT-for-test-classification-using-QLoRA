# QLoRA + BERT binary classification (terminal training)

Files:
- train.py : training script (uses bitsandbytes 4-bit + PEFT/LoRA)
- requirements.txt : Python dependencies

Basic steps:

1. Create and activate venv:
   python3 -m venv ~/venv_qLoRA
   source ~/venv_qLoRA/bin/activate

2. Install requirements:
   pip install -r requirements.txt

3. (Optional) Login to Hugging Face and Wandb:
   huggingface-cli login
   wandb login

4. Run training:
   python train.py \
     --model_name bert-base-uncased \
     --dataset_name dipanjanS/imdb_sentiment_finetune_dataset20k \
     --output_dir ./qlora_bert_checkpoint \
     --per_device_train_batch_size 16 \
     --per_device_eval_batch_size 32 \
     --num_train_epochs 3 \
     --report_to wandb

Adjust batch sizes for your GPU. For <24GB VRAM use per_device_train_batch_size=4 and increase gradient_accumulation_steps.

Notes:
- The script automatically loads model in 4-bit and adds LoRA adapters.
- After training, the small adapter is saved to `output_dir`.
- For memory comparison, try loading the plain float32 model (remove quantization / PEFT) and compare `nvidia-smi` while loading.