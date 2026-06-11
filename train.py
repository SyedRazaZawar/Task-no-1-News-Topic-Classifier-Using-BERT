import argparse
import os
import sys
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score

# Define labels mapping
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
NUM_LABELS = len(LABEL_NAMES)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT on AG News dataset.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Pretrained model identifier from huggingface.co/models (e.g. bert-base-uncased, distilbert-base-uncased, prajjwal1/bert-tiny)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation."
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=0,
        help="Number of training samples to use (0 for full dataset). Useful for fast testing on CPU."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save training outputs and logs."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/best_model",
        help="Directory to save the final fine-tuned model."
    )
    return parser.parse_args()

def compute_metrics(eval_pred):
    """Computes accuracy and weighted F1-score."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    
    return {
        "accuracy": acc,
        "f1_score": f1
    }

def main():
    args = parse_args()
    
    print("=" * 60)
    print("      News Topic Classifier - Training Pipeline")
    print("=" * 60)
    print(f"Model:           {args.model_name}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Learning Rate:   {args.lr}")
    print(f"Subset Size:     {args.subset_size if args.subset_size > 0 else 'Full Dataset'}")
    print(f"Device:          {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("-" * 60)
    
    # 1. Load the AG News dataset
    print("Loading AG News dataset from Hugging Face...")
    dataset = load_dataset("ag_news")
    
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    # Create subset if specified (highly recommended for CPU execution)
    if args.subset_size > 0:
        print(f"Selecting subset of {args.subset_size} samples for training...")
        train_dataset = train_dataset.shuffle(seed=42).select(range(min(args.subset_size, len(train_dataset))))
        # Select a corresponding smaller subset for testing
        test_subset_size = max(int(args.subset_size * 0.1), 100)
        test_dataset = test_dataset.shuffle(seed=42).select(range(min(test_subset_size, len(test_dataset))))
        
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    
    # 2. Tokenization and Preprocessing
    print(f"Loading tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    def tokenize_function(examples):
        # Clean text column in AG News dataset is 'text'
        return tokenizer(examples["text"], truncation=True, max_length=128)
    
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    # 3. Model setup
    print(f"Loading pretrained model: {args.model_name}...")
    # Map index to label name and vice versa for metadata serialization
    id2label = {i: label for i, label in enumerate(LABEL_NAMES)}
    label2id = {label: i for i, label in enumerate(LABEL_NAMES)}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id
    )
    
    # Collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 4. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=100,
        report_to="none", # Disable logging to external trackers
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU is available
        push_to_hub=False,
    )
    
    # 5. Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 6. Fine-tuning
    print("Starting fine-tuning...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current checkpoint...")
    
    # 7. Evaluate on Test set
    print("Evaluating model performance on test set...")
    metrics = trainer.evaluate()
    print("\n" + "=" * 60)
    print("                     EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {metrics.get('eval_accuracy', 0.0):.4f}")
    print(f"Test F1-Score: {metrics.get('eval_f1_score', 0.0):.4f}")
    print(f"Eval Loss:     {metrics.get('eval_loss', 0.0):.4f}")
    print("=" * 60)
    
    # 8. Save the best model
    print(f"Saving best model and tokenizer to {args.save_dir}...")
    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
