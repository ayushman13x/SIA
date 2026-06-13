import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def create_model_input(row):
    return (
        "Assigned Priority: " + str(row["Priority_Level"])
        + " | Issue Category: " + str(row["Issue_Category"])
        + " | Ticket Channel: " + str(row["Ticket_Channel"])
        + " | Resolution Time Hours: " + str(row["Resolution_Time_Hours"])
        + " | Subject: " + str(row["Ticket_Subject"])
        + " | Description: " + str(row["Ticket_Description"])
    )


class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=96):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_function = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_function(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    recalls = recall_score(labels, predictions, average=None, labels=[0, 1])

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "recall_consistent": recalls[0],
        "recall_mismatch": recalls[1],
    }


def main():
    parser = argparse.ArgumentParser(description="Train SIA DistilBERT classifier")

    parser.add_argument(
        "--input",
        default="data/processed/sia_dossiers.csv",
        help="Path to pseudo-labeled CSV file",
    )

    parser.add_argument(
        "--model_output",
        default="models/sia_distilbert_final",
        help="Folder where trained model will be saved",
    )

    parser.add_argument(
        "--metrics_output",
        default="outputs/distilbert_test_metrics.json",
        help="Path to save test metrics JSON",
    )

    parser.add_argument(
        "--config_output",
        default="outputs/experiment_config.json",
        help="Path to save experiment configuration JSON",
    )

    parser.add_argument(
        "--split_output_dir",
        default="data/processed",
        help="Folder to save train/validation/test split files",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )

    args = parser.parse_args()

    os.makedirs(args.model_output, exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.config_output), exist_ok=True)
    os.makedirs(args.split_output_dir, exist_ok=True)

    df = pd.read_csv(args.input)

    required_columns = [
        "Priority_Level",
        "Issue_Category",
        "Ticket_Channel",
        "Resolution_Time_Hours",
        "Ticket_Subject",
        "Ticket_Description",
        "mismatch_label",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df_model = df.copy()
    df_model["model_input"] = df_model.apply(create_model_input, axis=1)
    df_model["label"] = df_model["mismatch_label"].astype(int)

    train_df, temp_df = train_test_split(
        df_model,
        test_size=0.30,
        random_state=42,
        stratify=df_model["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"],
    )

    train_df.to_csv(os.path.join(args.split_output_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(args.split_output_dir, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(args.split_output_dir, "test_split.csv"), index=False)

    model_name = "distilbert-base-uncased"
    max_length = 96

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_dataset = TicketDataset(
        train_df["model_input"],
        train_df["label"],
        tokenizer,
        max_length=max_length,
    )

    val_dataset = TicketDataset(
        val_df["model_input"],
        val_df["label"],
        tokenizer,
        max_length=max_length,
    )

    test_dataset = TicketDataset(
        test_df["model_input"],
        test_df["label"],
        tokenizer,
        max_length=max_length,
    )

    label_counts = train_df["label"].value_counts().sort_index()
    total_samples = len(train_df)
    class_weights = total_samples / (2 * label_counts)
    class_weights = torch.tensor(class_weights.values, dtype=torch.float)

    training_args = TrainingArguments(
        output_dir="outputs/sia_distilbert_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=42,
        fp16=torch.cuda.is_available(),
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    test_predictions = trainer.predict(test_dataset)
    logits = test_predictions.predictions
    y_true = test_predictions.label_ids
    y_pred = np.argmax(logits, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    recalls = recall_score(y_true, y_pred, average=None, labels=[0, 1])
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Consistent", "Mismatch"],
        output_dict=True,
    )
    matrix = confusion_matrix(y_true, y_pred).tolist()

    final_metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "recall_consistent": recalls[0],
        "recall_mismatch": recalls[1],
        "confusion_matrix": matrix,
        "classification_report": report,
    }

    with open(args.metrics_output, "w", encoding="utf-8") as file:
        json.dump(final_metrics, file, indent=4)

    experiment_config = {
        "model_name": model_name,
        "task": "binary_priority_mismatch_classification",
        "target_column": "mismatch_label",
        "label_mapping": {"0": "Consistent", "1": "Mismatch"},
        "max_length": max_length,
        "split_strategy": "70/15/15 stratified split",
        "train_size": len(train_df),
        "validation_size": len(val_df),
        "test_size": len(test_df),
        "num_train_epochs": args.epochs,
        "train_batch_size": 16,
        "eval_batch_size": 32,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "class_weights": class_weights.tolist(),
        "imbalance_strategy": "weighted_cross_entropy_loss",
        "best_model_metric": "macro_f1",
    }

    with open(args.config_output, "w", encoding="utf-8") as file:
        json.dump(experiment_config, file, indent=4)

    trainer.save_model(args.model_output)
    tokenizer.save_pretrained(args.model_output)

    print("Training complete.")
    print(f"Model saved to: {args.model_output}")
    print(f"Metrics saved to: {args.metrics_output}")
    print(f"Config saved to: {args.config_output}")
    print("\nFinal test metrics:")
    print(json.dumps(final_metrics, indent=4))


if __name__ == "__main__":
    main()