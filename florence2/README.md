# Florence Model Fine-Tuning

This repository contains a script for fine-tuning a Florence model on a custom dataset using PyTorch, Hugging Face Transformers, and Weights & Biases (wandb) for experiment tracking.

## Usage

The script supports various command-line arguments for customizing the training process. Below is a description of the available arguments:

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

To run the training script with default settings:
```bash
python train.py
```

To customize the training process:
```bash
python train.py --model_name "your-model-name" --dataset "your-dataset" --learning_rate 1e-5 --epochs 3
```

## Training Process

The training script performs the following steps:

1. Parses command-line arguments.
2. Initializes Weights & Biases for experiment tracking.
3. Downloads and prepares the dataset.
4. Initializes the model and processor from the Hugging Face library.
5. Defines a custom dataset and data loader.
6. Defines a training loop with evaluation and logging.
7. Calculates evaluation metrics (BLEU, METEOR, Levenshtein distance).
8. Saves model checkpoints and optionally pushes them to the Hugging Face Hub.

## Evaluation Metrics

The script calculates the following evaluation metrics:

- BLEU score
- METEOR score
- Levenshtein distance

These metrics are logged to Weights & Biases for each evaluation step.

## Model Checkpointing

Model checkpoints are saved to the specified output directory after each epoch. Optionally, the model can be pushed to the Hugging Face Hub.


