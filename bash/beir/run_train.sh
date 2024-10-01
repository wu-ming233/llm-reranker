#!/bin/bash

# Define model, dataset paths, and output directory
BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
TRAIN_DATA_PATH="rryisthebest/rank_zephyr_training_data_alpha"  # Train Dataset --> Hugging Face dataset or Local dataset
EVAL_DATA_PATH="rryisthebest/evaluation_data_alpha"  # Eval Dataset --> Hugging Face dataset or Local dataset
OUTPUT_DIR="${REPO_DIR}/models/ranking/FIRST_mistral-7B-0.3"  # Directory to save the trained model
BEIR_DATA_DIR="${REPO_DIR}/datasets/beir/"

# RESUME_CHECKPOINT="${OUTPUT_DIR}/epoch_1" # currently init from epoch 1

# Launch training with DeepSpeed configuration
accelerate launch --config_file "${REPO_DIR}/train_configs/accel_config_deepspeed.yaml" "${REPO_DIR}/scripts/train_ranking.py" \
    --model_name_or_path "${BASE_MODEL}" \
    --train_dataset_path "${TRAIN_DATA_PATH}" \
    --eval_dataset_path "${EVAL_DATA_PATH}" \
    --beir_data_path "${BEIR_DATA_DIR}" \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 4 \
    --seed 42 \
    --per_device_train_batch_size 2 \
    --eval_steps 400 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 20 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 50 \
    --output_dir "${OUTPUT_DIR}" \
    --noisy_embedding_alpha 5 \
    --objective combined \
    --with_tracking \
    --report_to wandb 
    # --resume_from_checkpoint "${RESUME_CHECKPOINT}"
