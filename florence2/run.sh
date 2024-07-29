#!/bin/bash

# Default values
MODEL_NAME="microsoft/Florence-2-base-ft"
DATASET="AdithyaSK/img2Latex-v2-mini"
LEARNING_RATE=5e-6
LR_SCHEDULER_TYPE="cosine"
GRADIENT_ACCUMULATION_STEPS=1
PREPROCESSING_NUM_WORKERS=0
OUTPUT_DIR="./output"
BATCH_SIZE=1
EPOCHS=1
WANDB_PROJECT="florence-finetuning"
EVALS_PER_EPOCH=10
PUSH_TO_HUB="Latexify"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lr_scheduler_type)
            LR_SCHEDULER_TYPE="$2"
            shift 2
            ;;
        --gradient_accumulation_steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --preprocessing_num_workers)
            PREPROCESSING_NUM_WORKERS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --evals_per_epoch)
            EVALS_PER_EPOCH="$2"
            shift 2
            ;;
        --push_to_hub)
            PUSH_TO_HUB="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Prepare the push_to_hub argument
PUSH_TO_HUB_ARG=""
if [ ! -z "$PUSH_TO_HUB" ]; then
    PUSH_TO_HUB_ARG="--push_to_hub $PUSH_TO_HUB"
fi

# Run the Python script with the specified arguments
accelerate launch train.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --preprocessing_num_workers "$PREPROCESSING_NUM_WORKERS" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --wandb_project "$WANDB_PROJECT" \
    --evals_per_epoch "$EVALS_PER_EPOCH" \
    $PUSH_TO_HUB_ARG