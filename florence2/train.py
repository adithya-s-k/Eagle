# Import necessary libraries
import torch
import wandb
import numpy as np

from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from datasets import load_dataset
from transformers import AdamW, AutoModelForCausalLM, AutoProcessor, get_scheduler
from tqdm import tqdm
import os
import json
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Generator
from accelerate import Accelerator
import argparse
import base64
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
from Levenshtein import distance as levenshtein_distance

nltk.download("wordnet")


# Define function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a model on a dataset.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Florence-2-base-ft",
        help="Name or path of the model to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="AdithyaSK/img2Latex-v2",
        help="Name or path of the dataset to use",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-6, help="Learning rate for training"
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        help="Type of learning rate scheduler to use",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=0,
        help="Number of workers to use for preprocessing",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save the model checkpoints",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="florence-finetuning",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--evals_per_epoch",
        type=int,
        default=4,
        help="Number of evaluations to perform per epoch",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="Repository name to push the model to Hugging Face Hub. If not provided, model won't be pushed.",
    )
    return parser.parse_args()


# Main function to run the training process
def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Initialize Weights & Biases for experiment tracking
    wandb.init(project=args.wandb_project)

    # Set up training parameters
    BATCH_SIZE = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    NUM_WORKERS = args.preprocessing_num_workers
    epochs = args.epochs
    learning_rate = args.learning_rate
    CHECKPOINT = args.model_name
    output_dir = args.output_dir

    print("Downloading Dataset")
    # Load the dataset
    ds = load_dataset(args.dataset)

    # Initialize the Accelerator for distributed training
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

    DEVICE = accelerator.device
    # Load the model and processor
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True).to(
        DEVICE
    )
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)

    # Define a custom Dataset class for Florence model
    class FlorenceDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            example = self.data[idx]
            question = "<OCR>"
            first_answer = example["text"]
            image = example["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            return question, first_answer, image

    # Define a collate function for DataLoader
    def collate_fn(batch):
        questions, answers, images = zip(*batch)
        inputs = processor(
            text=list(questions), images=list(images), return_tensors="pt", padding=True
        ).to(DEVICE)
        return inputs, answers

    # Create datasets and dataloaders
    train_dataset = FlorenceDataset(ds["train"])
    val_dataset = FlorenceDataset(ds["test"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )
    test_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        shuffle=False,
    )

    # Function to run inference on a single example
    def run_example(task_prompt, text_input, image, model, processor, device):
        prompt = task_prompt + text_input
        # Ensure the image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Run model inference
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1020,
        )
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )
        try:
            parsed_answer = parsed_answer["<OCR>"]
        except Exception as e:
            print("An Exception has occurred while parsing output: " + str(e))

        return parsed_answer

    # Function to calculate evaluation metrics
    def calculate_metrics(reference, hypothesis):
        bleu_score = sentence_bleu([reference.split()], hypothesis.split())
        meteor_score_value = meteor_score([reference.split()], hypothesis.split())
        edit_distance = levenshtein_distance(reference, hypothesis)
        return bleu_score, meteor_score_value, edit_distance

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

    # Main training function
    def train_model(
        train_loader,
        val_loader,
        model,
        processor,
        epochs,
        lr,
        lr_scheduler,
        evals_per_epoch,
        push_to_hub,
        smoothing_window=10,
    ):
        # global eval_interation_count
        # Initialize optimizer and learning rate scheduler
        optimizer = AdamW(model.parameters(), lr=lr)
        num_training_steps = epochs * len(train_loader)
        lr_scheduler = get_scheduler(
            name=lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=50,
            num_training_steps=num_training_steps,
        )
        eval_interation_count = 0

        results_table = wandb.Table(
            columns=[
                "Eval Iteration",
                "Image",
                "Actual Prediction",
                "Generated Prediction",
                "BLEU",
                "METEOR",
                "Edit Distance",
            ]
        )
        # Prepare model, optimizer, dataloader, and scheduler for distributed training
        model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_loader, lr_scheduler
        )
        all_train_losses = []
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            total_steps = len(train_loader)
            for step, (inputs, answers) in enumerate(
                tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")
            ):
                with accelerator.accumulate(model):
                    input_ids = inputs["input_ids"]
                    pixel_values = inputs["pixel_values"]
                    labels = processor.tokenizer(
                        text=answers,
                        return_tensors="pt",
                        padding=True,
                        return_token_type_ids=False,
                    ).input_ids.to(DEVICE)

                    outputs = model(
                        input_ids=input_ids, pixel_values=pixel_values, labels=labels
                    )
                    loss = outputs.loss

                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    train_loss += loss.item()

                    all_train_losses.append(loss.item())
                    # Calculate global step
                    global_step = epoch * len(train_loader) + step

                    # Log training metrics
                    wandb.log(
                        {
                            "train_loss": loss.item(),
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "step": global_step,
                        }
                    )
                    # Log smoothened training loss
                    if len(all_train_losses) >= smoothing_window:
                        smoothened_loss = moving_average(
                            all_train_losses, smoothing_window
                        )[-1]
                        wandb.log(
                            {"smoothened_train_loss": smoothened_loss, "epoch": epoch}
                        )

                    if (step + 1) % (total_steps // evals_per_epoch) == 0:
                        eval_interation_count += 1
                        print(
                            f"\nPerforming evaluation at step {step + 1}/{total_steps}"
                        )
                        # Validation loop
                        model.eval()
                        val_loss = 0
                        with torch.no_grad():
                            for inputs, answers in tqdm(
                                val_loader,
                                desc=f"Validation Epoch {epoch + 1}/{epochs}",
                            ):
                                input_ids = inputs["input_ids"]
                                pixel_values = inputs["pixel_values"]
                                labels = processor.tokenizer(
                                    text=answers,
                                    return_tensors="pt",
                                    padding=True,
                                    return_token_type_ids=False,
                                ).input_ids.to(DEVICE)

                                outputs = model(
                                    input_ids=input_ids,
                                    pixel_values=pixel_values,
                                    labels=labels,
                                )
                                loss = outputs.loss

                                val_loss += loss.item()

                            avg_val_loss = val_loss / len(val_loader)
                            print(f"Average Validation Loss: {avg_val_loss}")
                            wandb.log(
                                {
                                    "avg_eval_loss": avg_val_loss,
                                }
                            )

                        # Evaluate on a subset of validation samples
                        eval_samples = list(val_dataset)[:100]
                        bleu_scores = []
                        meteor_scores = []
                        edit_distances = []

                        with torch.no_grad():
                            for sample in tqdm(
                                eval_samples, desc="Evaluating on random samples"
                            ):
                                question, reference_answer, image = sample
                                generated_answer = run_example(
                                    "<OCR>", "", image, model, processor, DEVICE
                                )

                                bleu, meteor, edit_dist = calculate_metrics(
                                    reference_answer, generated_answer
                                )
                                bleu_scores.append(bleu)
                                meteor_scores.append(meteor)
                                edit_distances.append(edit_dist)

                                # Log individual sample results
                                # wandb.log(
                                #     {
                                #         "sample_bleu": bleu,
                                #         "sample_meteor": meteor,
                                #         "sample_edit_distance": edit_dist,
                                #     }
                                # )
                                results_table.add_data(
                                    eval_interation_count,
                                    wandb.Image(image),
                                    reference_answer,
                                    generated_answer,
                                    bleu,
                                    meteor,
                                    edit_dist,
                                )

                        # Calculate and log average scores
                        avg_bleu = sum(bleu_scores) / len(bleu_scores)
                        avg_meteor = sum(meteor_scores) / len(meteor_scores)
                        avg_edit_distance = sum(edit_distances) / len(edit_distances)

                        wandb.log(
                            {
                                "evaluation_results": results_table,
                                "avg_bleu": avg_bleu,
                                "avg_meteor": avg_meteor,
                                "avg_edit_distance": avg_edit_distance,
                                "eval_iteration": eval_interation_count,
                                "epoch": epoch,
                                "step": global_step,
                            }
                        )

                        model.train()

            avg_train_loss = train_loss / len(train_loader)
            print(f"Average Training Loss: {avg_train_loss}")
            wandb.log({"avg_train_loss": avg_train_loss, "epoch": epoch})

            # Save model checkpoint
            weights_output_dir = output_dir + f"/epoch_{epoch+1}"
            os.makedirs(weights_output_dir, exist_ok=True)
            accelerator.save_model(model, weights_output_dir)

            if push_to_hub:
                print("Pushing Model to Hub")
                model.push_to_hub(push_to_hub)

            # Log model checkpoint to wandb
            wandb.save(os.path.join(weights_output_dir, "*"))

    # Freeze the vision tower parameters
    for param in model.vision_tower.parameters():
        param.is_trainable = False

    # Start training
    train_model(
        train_loader,
        test_loader,
        model,
        processor,
        epochs=epochs,
        lr=learning_rate,
        lr_scheduler=args.lr_scheduler_type,
        evals_per_epoch=args.evals_per_epoch,
        push_to_hub=args.push_to_hub,
    )

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
