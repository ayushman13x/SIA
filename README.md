# Support Integrity Auditor

Support Integrity Auditor is a ticket-auditing system that checks whether a customer support ticket appears to be assigned the wrong priority.

The project focuses on two cases:

* **Hidden Crisis:** the ticket looks serious, but it was assigned a low priority.
* **False Alarm:** the ticket looks minor, but it was assigned a high priority.

The dataset does not contain ground-truth mismatch labels, so the first part of the project creates pseudo-labels using ticket text and metadata. A fine-tuned classifier is then trained on those labels and used for prediction.

---

## Dataset

I used the Customer Support Tickets CRM dataset.

Main columns used:

| Column                  | Use                           |
| ----------------------- | ----------------------------- |
| `Ticket_Subject`        | Short summary of the issue    |
| `Ticket_Description`    | Main ticket text              |
| `Priority_Level`        | Human-assigned priority       |
| `Issue_Category`        | Ticket category               |
| `Ticket_Channel`        | Intake channel                |
| `Resolution_Time_Hours` | Structured operational signal |
| `Customer_Email`        | Customer identifier           |
| `Assigned_Agent`        | Operational metadata          |
| `Satisfaction_Score`    | Customer feedback metadata    |

Total records used: **20,000 tickets**.

---

## Approach

The project is divided into three main stages.

### 1. Pseudo-label generation

Since no mismatch labels were available, I first estimated an independent severity level for each ticket. This severity estimate was created using three signals:

#### Rule-based severity signal

I built a rule-based signal from phrases found during EDA. Examples include:

* API/server failures
* suspicious account activity
* payment and billing problems
* login failures
* dashboard loading or data sync issues
* general inquiry phrases such as pricing, demo, and patch questions

Each matched rule produces a severity score and traceable evidence.

#### Semantic clustering signal

Ticket text was embedded using `sentence-transformers/all-MiniLM-L6-v2`. I then clustered the embeddings using KMeans and manually inspected the main terms in each cluster.

Each semantic cluster was mapped to a severity level based on the type of issue it represented.

#### Resolution-time signal

Resolution time was used as a weak structured signal. Since resolution time depends on workflow and agent handling, it was not treated as a direct severity label. It was only used as a supporting signal.

The final severity estimate used this fusion:

```text
0.45 * rule severity
+ 0.45 * semantic severity
+ 0.10 * resolution-time signal
```

After calculating inferred severity, I compared it with the assigned priority:

```text
severity_delta = inferred_severity_score - assigned_priority_score
```

A ticket was marked as mismatch if:

```text
abs(severity_delta) >= 2
```

Mismatch type:

```text
severity_delta >= 2  -> Hidden Crisis
severity_delta <= -2 -> False Alarm
```

---

## Signal agreement

I checked pairwise agreement between the signals before using them for pseudo-label generation.

| Signal pair               | Exact agreement | Relaxed agreement |
| ------------------------- | --------------: | ----------------: |
| Rule vs Semantic          |          0.4878 |            0.6574 |
| Rule vs Resolution        |          0.2490 |            0.5592 |
| Rule vs Resolution Signal |          0.2369 |            0.5335 |

The resolution signal was weaker than the text-based signals, so I kept its fusion weight low.

---

## Fusion ablation

| Strategy       | Rule weight | Semantic weight | Resolution weight | Observation                          |
| -------------- | ----------: | --------------: | ----------------: | ------------------------------------ |
| Rule heavy     |        0.55 |            0.35 |              0.10 | More conservative and interpretable  |
| Balanced text  |        0.45 |            0.45 |              0.10 | Final choice                         |
| Semantic heavy |        0.35 |            0.55 |              0.10 | Relied more on cluster-level meaning |
| Text only      |        0.50 |            0.50 |              0.00 | Removed structured resolution signal |

I selected the balanced text strategy because it gave similar importance to explicit rule evidence and semantic grouping, while still keeping a small structured metadata signal.

---

## Model training

I trained a binary classifier to predict:

```text
0 = Consistent
1 = Mismatch
```

Model used:

```text
distilbert-base-uncased
```

The model input combines text and structured metadata:

```text
Assigned Priority + Issue Category + Ticket Channel + Resolution Time + Subject + Description
```

Class imbalance was handled using weighted cross-entropy loss.

Data split:

| Split      |   Rows |
| ---------- | -----: |
| Train      | 14,000 |
| Validation |  3,000 |
| Test       |  3,000 |

---

## Test results

The final model was evaluated on the held-out pseudo-labeled test split.

| Metric             |  Score |
| ------------------ | -----: |
| Accuracy           | 0.9983 |
| Macro F1           | 0.9979 |
| Recall: Consistent | 0.9991 |
| Recall: Mismatch   | 0.9964 |

Confusion matrix:

```text
[[2163,    2],
 [   3,  832]]
```

These metrics are measured against pseudo-labels, not human ground-truth labels. So the result should be interpreted as how well the classifier learned the pseudo-labeling policy.

---

## Evidence dossier

For every predicted mismatch, the system creates an Evidence Dossier.

Schema:

```json
{
  "ticket_id": "...",
  "assigned_priority": "...",
  "inferred_severity": "...",
  "mismatch_type": "Hidden Crisis | False Alarm",
  "severity_delta": 0,
  "feature_evidence": [
    {
      "signal": "keyword",
      "source_field": "Ticket_Subject",
      "value": "...",
      "weight": 4
    },
    {
      "signal": "resolution_time",
      "source_field": "Resolution_Time_Hours",
      "value": "...",
      "interpretation": "..."
    }
  ],
  "constraint_analysis": "...",
  "confidence": "High"
}
```

Grounding rule used in the project:

* Every evidence item must come from an actual ticket field.
* No dossier uses unsupported or invented evidence.
* For processed dataset tickets, saved Stage-1 dossiers are used.
* For new unseen tickets, a real-time dossier is generated from the ticket fields.

---

## Streamlit app

The Streamlit app supports:

* Single ticket audit
* Batch CSV upload
* Prediction output download
* Evidence Dossier display
* Priority mismatch dashboard
* Hidden Crisis vs False Alarm distribution
* Top contributing signals
* Severity delta heatmap by issue category and channel

Run locally:

```bash
streamlit run app.py
```

---

## Inference

Run prediction on a CSV file:

```bash
python src/predict.py --input data/processed/test_split.csv --output data/processed/test_prediction_output.csv --model_path models/sia_distilbert_final --dossier_path data/processed/evidence_dossiers_v1.json
```

The output file keeps the original ticket columns and adds:

* `model_prediction`
* `model_prediction_label`
* `model_confidence`
* `prob_consistent`
* `prob_mismatch`
* `final_dossier`

---

## Training

The standalone training script is:

```text
src/train_pipeline.py
```

Run training from scratch:

```bash
python src/train_pipeline.py --input data/processed/sia_dossiers.csv
```

This recreates the train/validation/test split, fine-tunes the classifier, saves metrics, and exports the trained model.

---

## Project structure

```text
SIA/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── customer_support_tickets.csv
│   └── processed/
│       ├── sia_dossiers.csv
│       ├── evidence_dossiers_v1.json
│       ├── train_split.csv
│       ├── val_split.csv
│       ├── test_split.csv
│       └── test_prediction_output.csv
├── models/
│   └── sia_distilbert_final/
├── notebooks/
│   └── notebook.ipynb
├── outputs/
│   ├── experiment_config.json
│   └── distilbert_test_metrics.json
└── src/
    ├── dossier_utils.py
    ├── predict.py
    └── train_pipeline.py
```

---

## Links

GitHub Repository: `ADD_LINK_HERE`

Hosted Streamlit App: `ADD_LINK_HERE`

Demo Video: `ADD_LINK_HERE`
