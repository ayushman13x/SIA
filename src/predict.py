import argparse
import ast
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dossier_utils import create_realtime_dossier

LABEL_MAP = {
    0: "Consistent",
    1: "Mismatch",
}


def create_model_input(row):
    return (
        "Assigned Priority: " + str(row["Priority_Level"])
        + " | Issue Category: " + str(row["Issue_Category"])
        + " | Ticket Channel: " + str(row["Ticket_Channel"])
        + " | Resolution Time Hours: " + str(row["Resolution_Time_Hours"])
        + " | Subject: " + str(row["Ticket_Subject"])
        + " | Description: " + str(row["Ticket_Description"])
    )


def safe_parse_dossier(value):
    if isinstance(value, dict):
        return value

    if pd.isna(value):
        return None

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None

    return None


def load_dossier_lookup(dossier_path):
    if dossier_path is None:
        return {}

    if not os.path.exists(dossier_path):
        return {}

    with open(dossier_path, "r", encoding="utf-8") as file:
        dossiers = json.load(file)

    dossier_lookup = {}

    if isinstance(dossiers, list):
        for dossier in dossiers:
            if isinstance(dossier, dict) and "ticket_id" in dossier:
                dossier_lookup[str(dossier["ticket_id"])] = dossier

    return dossier_lookup


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_batch(df, tokenizer, model, device, batch_size=64, max_length=96):
    model_inputs = df.apply(create_model_input, axis=1).tolist()

    all_predictions = []
    all_probs = []

    for start in range(0, len(model_inputs), batch_size):
        end = start + batch_size
        batch_texts = model_inputs[start:end]

        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = np.argmax(probs, axis=1)

        all_predictions.extend(predictions.tolist())
        all_probs.extend(probs.tolist())

        print(f"Predicted {min(end, len(model_inputs))}/{len(model_inputs)} rows")

    pred_df = pd.DataFrame(
        {
            "model_prediction": all_predictions,
            "model_prediction_label": [LABEL_MAP[pred] for pred in all_predictions],
            "model_confidence": [float(max(prob)) for prob in all_probs],
            "prob_consistent": [float(prob[0]) for prob in all_probs],
            "prob_mismatch": [float(prob[1]) for prob in all_probs],
        }
    )

    return pred_df


def run_batch_prediction(input_csv, output_csv, model_path, dossier_path=None):
    df = pd.read_csv(input_csv)

    required_columns = [
        "Priority_Level",
        "Issue_Category",
        "Ticket_Channel",
        "Resolution_Time_Hours",
        "Ticket_Subject",
        "Ticket_Description",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    tokenizer, model, device = load_model(model_path)
    dossier_lookup = load_dossier_lookup(dossier_path)

    print("Running prediction...")
    pred_df = predict_batch(
        df=df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=64,
        max_length=96,
    )

    output_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)

    final_dossiers = []

    for _, row in output_df.iterrows():
        if row["model_prediction"] != 1:
            final_dossiers.append(None)
            continue

        ticket_id = str(row.get("Ticket_ID", ""))

        if ticket_id in dossier_lookup:
            dossier = dossier_lookup[ticket_id]
        elif "evidence_dossier" in output_df.columns:
            dossier = safe_parse_dossier(row["evidence_dossier"])
        else:
            dossier = None

        if dossier is None:
             dossier = create_realtime_dossier(row)
        final_dossiers.append(json.dumps(dossier, ensure_ascii=False))

    output_df["final_dossier"] = final_dossiers

    original_cols = [
        "Ticket_ID",
        "Customer_Name",
        "Customer_Email",
        "Ticket_Subject",
        "Ticket_Description",
        "Issue_Category",
        "Priority_Level",
        "Ticket_Channel",
        "Submission_Date",
        "Resolution_Time_Hours",
        "Assigned_Agent",
        "Satisfaction_Score",
    ]

    prediction_cols = [
        "model_prediction",
        "model_prediction_label",
        "model_confidence",
        "prob_consistent",
        "prob_mismatch",
        "final_dossier",
    ]

    if "label" in output_df.columns:
        output_df["actual_label"] = output_df["label"]
        prediction_cols.insert(0, "actual_label")
    elif "mismatch_label" in output_df.columns:
        output_df["actual_label"] = output_df["mismatch_label"]
        prediction_cols.insert(0, "actual_label")

    final_cols = original_cols + prediction_cols
    final_cols = [col for col in final_cols if col in output_df.columns]

    output_df = output_df[final_cols]
    output_df.to_csv(output_csv, index=False)

    print("Prediction complete.")
    print(f"Input file: {input_csv}")
    print(f"Output file: {output_csv}")
    print("\nPrediction distribution:")
    print(output_df["model_prediction_label"].value_counts())

    return output_df


def main():
    parser = argparse.ArgumentParser(
        description="SIA Priority Mismatch Prediction Script"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to save output CSV file",
    )

    parser.add_argument(
        "--model_path",
        default="models/sia_distilbert_final",
        help="Path to trained DistilBERT model folder",
    )

    parser.add_argument(
        "--dossier_path",
        default=None,
        help="Optional path to evidence dossier JSON file",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    run_batch_prediction(
        input_csv=args.input,
        output_csv=args.output,
        model_path=args.model_path,
        dossier_path=args.dossier_path,
    )


if __name__ == "__main__":
    main()