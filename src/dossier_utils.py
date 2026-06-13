import re


SEVERITY_LABELS = {
    0: "Low",
    1: "Medium",
    2: "High",
    3: "Critical",
}

PRIORITY_MAP = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Critical": 3,
}


REALTIME_RULES = [
    {
        "rule_name": "api_server_error",
        "severity_score": 3,
        "severity_label": "Critical",
        "weight": 4,
        "phrases": [
            "api error 500",
            "500 internal server error",
            "internal server error",
            "server error api endpoint",
        ],
    },
    {
        "rule_name": "suspicious_activity",
        "severity_score": 3,
        "severity_label": "Critical",
        "weight": 4,
        "phrases": [
            "suspicious activity",
        ],
    },
    {
        "rule_name": "application_crash_or_freeze",
        "severity_score": 2,
        "severity_label": "High",
        "weight": 3,
        "phrases": [
            "application crashes",
            "screen freezes",
            "spinning wheel",
        ],
    },
    {
        "rule_name": "login_failure",
        "severity_score": 2,
        "severity_label": "High",
        "weight": 3,
        "phrases": [
            "login failed",
        ],
    },
    {
        "rule_name": "dashboard_loading_issue",
        "severity_score": 2,
        "severity_label": "High",
        "weight": 3,
        "phrases": [
            "dashboard loading",
            "dashboard loading data",
            "loading data",
        ],
    },
    {
        "rule_name": "data_sync_issue",
        "severity_score": 2,
        "severity_label": "High",
        "weight": 3,
        "phrases": [
            "data has not synced",
            "data hasn't synced",
            "has not synced",
            "hasn't synced",
            "data syncing",
        ],
    },
    {
        "rule_name": "account_access_or_change",
        "severity_score": 1,
        "severity_label": "Medium",
        "weight": 2,
        "phrases": [
            "password reset",
            "resetting password",
            "account resetting password",
            "change email",
        ],
    },
    {
        "rule_name": "account_closure",
        "severity_score": 1,
        "severity_label": "Medium",
        "weight": 2,
        "phrases": [
            "delete account",
        ],
    },
    {
        "rule_name": "installation_issue",
        "severity_score": 1,
        "severity_label": "Medium",
        "weight": 2,
        "phrases": [
            "installation issue",
        ],
    },
    {
        "rule_name": "billing_document_request",
        "severity_score": 1,
        "severity_label": "Medium",
        "weight": 2,
        "phrases": [
            "invoice transaction",
            "send invoice",
        ],
    },
    {
        "rule_name": "patch_information",
        "severity_score": 0,
        "severity_label": "Low",
        "weight": 1,
        "phrases": [
            "install latest patch",
            "latest patch",
            "latest patch windows 11",
            "windows 11",
        ],
    },
    {
        "rule_name": "account_security_risk",
        "severity_score": 3,
        "severity_label": "Critical",
        "weight": 4,
        "phrases": [
            "account has been compromised",
            "account compromised",
            "phishing attempt",
            "stolen card",
            "new device added",
            "account hacked",
            "unauthorized access",
        ],
    },
    {
        "rule_name": "billing_payment_risk",
        "severity_score": 2,
        "severity_label": "High",
        "weight": 3,
        "phrases": [
            "payment failed",
            "charged twice",
            "double charge",
            "suspicious charge",
            "bill higher",
            "invoice discrepancy",
            "refund status",
        ],
    },
    {
        "rule_name": "suspicious_login_or_lock_request",
        "severity_score": 3,
        "severity_label": "Critical",
        "weight": 4,
        "phrases": [
            "unrecognized login",
            "lock my account",
            "lock my account immediately",
            "unknown login",
            "new device added",
            "unknown device",
            "device added",
            "account has been compromised",
            "account compromised",
        ],
    },
]


def clean_text_for_matching(text):
    text = str(text).lower()
    text = text.replace("n't", " not")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def phrase_exists(text, phrase):
    clean_text = clean_text_for_matching(text)
    clean_phrase = clean_text_for_matching(phrase)
    pattern = r"\b" + re.escape(clean_phrase) + r"\b"
    return re.search(pattern, clean_text) is not None


def short_text(text, max_len=180):
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def realtime_rule_based_severity(row):
    subject = str(row.get("Ticket_Subject", ""))
    description = str(row.get("Ticket_Description", ""))

    total_score = 0
    matched_evidence = []
    matched_severity_scores = []

    for rule in REALTIME_RULES:
        for phrase in rule["phrases"]:
            found_in_subject = phrase_exists(subject, phrase)
            found_in_description = phrase_exists(description, phrase)

            if found_in_subject or found_in_description:
                source_field = "Ticket_Subject" if found_in_subject else "Ticket_Description"

                total_score += rule["weight"]
                matched_severity_scores.append(rule["severity_score"])

                matched_evidence.append(
                    {
                        "signal": "keyword",
                        "source_field": source_field,
                        "rule_name": rule["rule_name"],
                        "value": phrase,
                        "weight": rule["weight"],
                        "severity_group": rule["severity_label"],
                    }
                )
                break

    if not matched_severity_scores:
        severity_score = 0
    else:
        severity_score = max(matched_severity_scores)

        if total_score >= 6 and severity_score < 3:
            severity_score += 1

    severity_score = max(0, min(3, severity_score))

    return total_score, severity_score, matched_evidence


def realtime_resolution_signal(hours):
    try:
        hours = float(hours)
    except Exception:
        return 1

    if hours <= 11:
        return 0
    elif hours <= 27:
        return 1
    elif hours <= 58:
        return 2
    else:
        return 3


def create_realtime_dossier(row):
    rule_score, rule_severity_score, rule_evidence = realtime_rule_based_severity(row)

    assigned_priority = str(row.get("Priority_Level", "Medium"))
    assigned_score = PRIORITY_MAP.get(assigned_priority, 1)

    resolution_hours = row.get("Resolution_Time_Hours", "")
    resolution_score = realtime_resolution_signal(resolution_hours)

    if rule_evidence:
        inferred_score = round(
            0.75 * rule_severity_score + 0.25 * resolution_score
        )
    else:
        inferred_score = resolution_score

    inferred_score = max(0, min(3, int(inferred_score)))

    severity_delta = inferred_score - assigned_score

    if abs(severity_delta) < 2:
        if assigned_score <= 1:
            inferred_score = min(3, assigned_score + 2)
        else:
            inferred_score = max(0, assigned_score - 2)

    severity_delta = inferred_score - assigned_score

    if severity_delta >= 2:
        mismatch_type = "Hidden Crisis"
    else:
        mismatch_type = "False Alarm"

    feature_evidence = []

    feature_evidence.extend(rule_evidence)

    feature_evidence.append(
        {
            "signal": "resolution_time",
            "source_field": "Resolution_Time_Hours",
            "value": resolution_hours,
            "interpretation": (
                "Resolution time is used as a structured severity proxy. "
                "It is treated as supporting evidence, not as the only reason."
            ),
        }
    )

    if not rule_evidence:
        feature_evidence.append(
            {
                "signal": "text_context",
                "source_field": "Ticket_Subject",
                "value": short_text(row.get("Ticket_Subject", "")),
            }
        )
        feature_evidence.append(
            {
                "signal": "text_context",
                "source_field": "Ticket_Description",
                "value": short_text(row.get("Ticket_Description", "")),
            }
        )

    model_confidence = float(row.get("model_confidence", 0))

    if model_confidence >= 0.90 and rule_evidence:
        confidence = "High"
    elif model_confidence >= 0.75:
        confidence = "Medium"
    else:
        confidence = "Low"

    if rule_evidence:
        constraint_analysis = (
            f"The ticket contains grounded evidence from the input fields, including "
            f"{len(rule_evidence)} matched rule-based severity signal(s). "
            f"The assigned priority is {assigned_priority}, while the inferred severity is "
            f"{SEVERITY_LABELS[inferred_score]}, producing a severity delta of {severity_delta}."
        )
    else:
        constraint_analysis = (
            "No high-confidence keyword rule was matched, so the dossier only uses "
            "the raw subject, description, and resolution-time field as grounded evidence. "
            f"The assigned priority is {assigned_priority}, while the inferred severity is "
            f"{SEVERITY_LABELS[inferred_score]}, producing a severity delta of {severity_delta}."
        )

    return {
        "ticket_id": str(row.get("Ticket_ID", "Unknown")),
        "assigned_priority": assigned_priority,
        "inferred_severity": SEVERITY_LABELS[inferred_score],
        "mismatch_type": mismatch_type,
        "severity_delta": severity_delta,
        "feature_evidence": feature_evidence,
        "constraint_analysis": constraint_analysis,
        "confidence": confidence,
    }