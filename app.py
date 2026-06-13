import json
import os
import sys
import plotly.express as px
import pandas as pd
import streamlit as st

sys.path.append("src")

from predict import load_model, predict_batch, safe_parse_dossier, load_dossier_lookup
from dossier_utils import create_realtime_dossier

MODEL_PATH = "models/sia_distilbert_final"
DOSSIER_PATH = "data/processed/evidence_dossiers_v1.json"


st.set_page_config(
    page_title="Support Integrity Auditor",
    layout="wide",
)


@st.cache_resource
def get_model():
    tokenizer, model, device = load_model(MODEL_PATH)
    return tokenizer, model, device


@st.cache_data
def get_dossier_lookup():
    return load_dossier_lookup(DOSSIER_PATH)


def attach_dossiers(df):
    dossier_lookup = get_dossier_lookup()
    final_dossiers = []

    for _, row in df.iterrows():
        if row["model_prediction"] != 1:
            final_dossiers.append(None)
            continue

        ticket_id = str(row.get("Ticket_ID", ""))

        if ticket_id in dossier_lookup:
            dossier = dossier_lookup[ticket_id]
        elif "evidence_dossier" in row:
            dossier = safe_parse_dossier(row["evidence_dossier"])
        else:
            dossier = None

        if dossier is None:
            dossier = create_realtime_dossier(row)

        final_dossiers.append(json.dumps(dossier, ensure_ascii=False, indent=2))

    df["final_dossier"] = final_dossiers
    return df


def show_dossier(dossier_text):
    if dossier_text is None or pd.isna(dossier_text):
        st.info("No dossier available for this ticket.")
        return

    try:
        dossier = json.loads(dossier_text)
        st.json(dossier)
    except Exception:
        st.write(dossier_text)

def parse_dashboard_dossier(value):
    if isinstance(value, dict):
        return value

    if value is None or pd.isna(value):
        return None

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        parsed = safe_parse_dossier(value)
        if isinstance(parsed, dict):
            return parsed

    return None


def get_mismatch_type_from_dossier(value):
    dossier = parse_dashboard_dossier(value)

    if dossier is None:
        return "Unavailable"

    return dossier.get("mismatch_type", "Unavailable")


def get_severity_delta_from_dossier(value):
    dossier = parse_dashboard_dossier(value)

    if dossier is None:
        return None

    try:
        return int(dossier.get("severity_delta"))
    except Exception:
        return None


def get_signal_names_from_dossier(value):
    dossier = parse_dashboard_dossier(value)

    if dossier is None:
        return []

    feature_evidence = dossier.get("feature_evidence", [])

    if not isinstance(feature_evidence, list):
        return []

    signal_names = []

    for item in feature_evidence:
        if not isinstance(item, dict):
            continue

        rule_name = item.get("rule_name")
        signal = item.get("signal")

        if rule_name:
            signal_names.append(rule_name)
        elif signal:
            signal_names.append(signal)

    return signal_names

st.title("Support Integrity Auditor")
st.caption(
    "A semantics-driven auditor for detecting customer support priority mismatches."
)

tab1, tab2, tab3 = st.tabs(
    ["Single Ticket Audit", "Batch CSV Audit", "Mismatch Dashboard"]
)


with tab1:
    st.subheader("Single Ticket Audit")

    col1, col2 = st.columns(2)

    with col1:
        ticket_id = st.text_input("Ticket ID", value="CUSTOM-001")
        priority = st.selectbox(
            "Assigned Priority",
            ["Low", "Medium", "High", "Critical"],
        )
        issue_category = st.text_input("Issue Category", value="Technical")
        ticket_channel = st.selectbox(
            "Ticket Channel",
            ["Email", "Chat", "Phone", "Web", "Social Media"],
        )

    with col2:
        resolution_time = st.number_input(
            "Resolution Time Hours",
            min_value=0.0,
            value=8.0,
        )
        subject = st.text_input("Ticket Subject", value="API error 500")
        description = st.text_area(
            "Ticket Description",
            value=(
                "Users are unable to access the dashboard because the API "
                "returns internal server error."
            ),
            height=140,
        )

    if st.button("Audit Ticket"):
        input_df = pd.DataFrame(
            [
                {
                    "Ticket_ID": ticket_id,
                    "Customer_Name": "Manual Input",
                    "Customer_Email": "",
                    "Ticket_Subject": subject,
                    "Ticket_Description": description,
                    "Issue_Category": issue_category,
                    "Priority_Level": priority,
                    "Ticket_Channel": ticket_channel,
                    "Submission_Date": "",
                    "Resolution_Time_Hours": resolution_time,
                    "Assigned_Agent": "",
                    "Satisfaction_Score": "",
                }
            ]
        )

        tokenizer, model, device = get_model()

        pred_df = predict_batch(
            df=input_df,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=1,
            max_length=96,
        )

        result_df = pd.concat([input_df, pred_df], axis=1)
        result_df = attach_dossiers(result_df)

        result = result_df.iloc[0]

        st.markdown("### Result")

        if result["model_prediction"] == 1:
            st.error("Mismatch Detected")
        else:
            st.success("Priority Looks Consistent")

        st.write("Prediction:", result["model_prediction_label"])
        st.write("Confidence:", round(result["model_confidence"], 4))
        st.write("Probability Consistent:", round(result["prob_consistent"], 4))
        st.write("Probability Mismatch:", round(result["prob_mismatch"], 4))

        st.markdown("### Evidence Dossier")
        show_dossier(result["final_dossier"])


with tab2:
    st.subheader("Batch CSV Audit")

    uploaded_file = st.file_uploader(
        "Upload a CSV file with support tickets",
        type=["csv"],
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Uploaded rows:", len(batch_df))
        st.dataframe(batch_df.head())

        if st.button("Run Batch Audit"):
            tokenizer, model, device = get_model()

            pred_df = predict_batch(
                df=batch_df,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=64,
                max_length=96,
            )

            output_df = pd.concat([batch_df.reset_index(drop=True), pred_df], axis=1)
            output_df = attach_dossiers(output_df)

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

            final_cols = original_cols + prediction_cols
            final_cols = [col for col in final_cols if col in output_df.columns]
            output_df = output_df[final_cols]

            st.success("Batch audit complete.")
            st.dataframe(output_df.head(20))

            csv_data = output_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Output CSV",
                data=csv_data,
                file_name="sia_prediction_output.csv",
                mime="text/csv",
            )


with tab3:
    st.subheader("Priority Mismatch Dashboard")

    default_path = "data/processed/test_prediction_output.csv"

    if os.path.exists(default_path):
        dashboard_df = pd.read_csv(default_path)

        total_tickets = len(dashboard_df)
        mismatch_count = int((dashboard_df["model_prediction"] == 1).sum())
        consistent_count = total_tickets - mismatch_count

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Tickets", total_tickets)
        col2.metric("Predicted Mismatches", mismatch_count)
        col3.metric("Predicted Consistent", consistent_count)

        st.markdown("### Prediction Distribution")
        prediction_counts = dashboard_df["model_prediction_label"].value_counts()
        st.bar_chart(prediction_counts)

        mismatch_df = dashboard_df[dashboard_df["model_prediction"] == 1].copy()

        if len(mismatch_df) == 0:
            st.info("No mismatches found in the current prediction file.")
        else:
            mismatch_df["parsed_mismatch_type"] = mismatch_df["final_dossier"].apply(
                get_mismatch_type_from_dossier
            )

            mismatch_df["parsed_severity_delta"] = mismatch_df["final_dossier"].apply(
                get_severity_delta_from_dossier
            )

            st.markdown("### Mismatch Type Distribution")
            mismatch_type_counts = mismatch_df["parsed_mismatch_type"].value_counts()
            st.bar_chart(mismatch_type_counts)

            st.markdown("### Mismatch Count by Assigned Priority")

            if "Priority_Level" in mismatch_df.columns:
                priority_chart = mismatch_df["Priority_Level"].value_counts()
                st.bar_chart(priority_chart)
            else:
                st.warning("Priority_Level column not found.")

            st.markdown("### Mismatch Count by Issue Category")

            if "Issue_Category" in mismatch_df.columns:
                category_chart = mismatch_df["Issue_Category"].value_counts()
                st.bar_chart(category_chart)
            else:
                st.warning("Issue_Category column not found.")

            st.markdown("### Top Contributing Signals")

            all_signals = []

            for dossier_text in mismatch_df["final_dossier"]:
                all_signals.extend(get_signal_names_from_dossier(dossier_text))

            if len(all_signals) > 0:
                signal_counts = pd.Series(all_signals).value_counts().head(10)
                st.bar_chart(signal_counts)
            else:
                st.info("No contributing signals found in available dossiers.")

            st.markdown("### Severity Delta Heatmap")

            heatmap_required_cols = [
                "Issue_Category",
                "Ticket_Channel",
                "parsed_severity_delta",
            ]

            has_heatmap_cols = all(
                col in mismatch_df.columns for col in heatmap_required_cols
            )

            if has_heatmap_cols:
                heatmap_data = mismatch_df.pivot_table(
                    index="Issue_Category",
                    columns="Ticket_Channel",
                    values="parsed_severity_delta",
                    aggfunc="mean",
                )

                if heatmap_data.empty:
                    st.info("Not enough data to create severity delta heatmap.")
                else:
                    fig = px.imshow(
                        heatmap_data,
                        text_auto=True,
                        aspect="auto",
                        title="Average Severity Delta by Issue Category and Channel",
                    )

                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    "Issue_Category, Ticket_Channel, or severity delta data is missing."
                )

            st.markdown("### Sample Hidden Crisis Tickets")
            hidden_crisis_df = mismatch_df[
                mismatch_df["parsed_mismatch_type"] == "Hidden Crisis"
            ]

            if len(hidden_crisis_df) > 0:
                st.dataframe(hidden_crisis_df.head(10))
            else:
                st.info("No Hidden Crisis tickets found in this prediction file.")

            st.markdown("### Sample False Alarm Tickets")
            false_alarm_df = mismatch_df[
                mismatch_df["parsed_mismatch_type"] == "False Alarm"
            ]

            if len(false_alarm_df) > 0:
                st.dataframe(false_alarm_df.head(10))
            else:
                st.info("No False Alarm tickets found in this prediction file.")

    else:
        st.warning(
            "Run predict.py first to create "
            "data/processed/test_prediction_output.csv"
        )