# Florence Model Fine-Tuning

This repository contains two scripts for fine-tuning a Florence model on custom datasets using PyTorch, Hugging Face Transformers, and Weights & Biases (wandb) for experiment tracking.

1. `train.py`: Standard fine-tuning without Low-Rank Adaptation (LoRA)
2. `lora_train.py`: Fine-tuning with Low-Rank Adaptation (LoRA) support

## Standard Fine-Tuning (train.py)

### Usage

The `train.py` script supports various command-line arguments for customizing the training process. Below is a description of the available arguments:

- `--model_name`: Name or path of the model to use (default: `microsoft/Florence-2-base-ft`).
- `--dataset`: Name or path of the dataset to use (default: `AdithyaSK/img2Latex-v2`).
- `--learning_rate`: Learning rate for training (default: `5e-6`).
- `--lr_scheduler_type`: Type of learning rate scheduler to use (default: `cosine`).
- `--gradient_accumulation_steps`: Number of update steps to accumulate before performing a backward/update pass (default: `2`).
- `--preprocessing_num_workers`: Number of workers to use for preprocessing (default: `0`).
- `--output_dir`: Directory to save the model checkpoints (default: `./output`).
- `--batch_size`: Batch size for training and evaluation (default: `4`).
- `--epochs`: Number of epochs to train (default: `1`).
- `--wandb_project`: Weights & Biases project name (default: `florence-finetuning`).
- `--evals_per_epoch`: Number of evaluations to perform per epoch (default: `4`).
- `--push_to_hub`: Repository name to push the model to Hugging Face Hub (optional).

To run the standard training script:
```bash
python train.py
```

## LoRA Fine-Tuning (lora_train.py)

### Usage

The `lora_train.py` script supports LoRA-based fine-tuning and includes additional features. Here are the command-line arguments:

- `--dataset`: Dataset to train on (required, choices: "docvqa", "cauldron", "vqainstruct").
- `--batch-size`: Batch size for training (default: `6`).
- `--use-lora`: Flag to enable Low-Rank Adaptation (LoRA) for training.
- `--epochs`: Number of epochs to train for (default: `10`).
- `--lr`: Learning rate (default: `1e-6`).
- `--eval-steps`: Number of steps between evaluations (default: `1000`).
- `--run-name`: Run name for wandb (optional).
- `--max-val-item-count`: Maximum number of items to evaluate on during validation (default: `1000`).

To run the LoRA training script:
```bash
python lora_train.py --dataset docvqa --use-lora
```

### LoRA Configuration

When LoRA is enabled, the script applies the following configuration:

- Rank (r): 8
- Alpha: 8
- Target modules: "q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"
- Task type: CAUSAL_LM
- LoRA dropout: 0.05
- Bias: none
- Use RS-LoRA: True
- Weight initialization: gaussian

## Training Process (Both Scripts)

Both scripts perform the following general steps:

1. Parse command-line arguments.
2. Initialize Weights & Biases for experiment tracking.
3. Load and prepare the dataset.
4. Initialize the model and processor.
5. Set up data loaders and optimizer.
6. Train the model with periodic evaluation.
7. Log metrics and save checkpoints.

## Evaluation Metrics

Both scripts calculate the following evaluation metrics:

- BLEU score
- METEOR score
- Levenshtein distance (Edit distance)

These metrics are logged to Weights & Biases during evaluation steps.

## Model Checkpointing

Model checkpoints are saved periodically. In the LoRA script, only LoRA weights are saved when LoRA is enabled.

## Requirements

Ensure you have the required libraries installed:

```bash
pip install transformers torch wandb datasets nltk Levenshtein peft
```
