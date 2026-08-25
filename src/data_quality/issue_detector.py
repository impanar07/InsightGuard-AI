from __future__ import annotations

import pandas as pd


def classify_severity(
    issue_rate: float,
    thresholds: dict[str, float]
) -> str:
    """
    Convert issue rate into a severity label.
    """

    if issue_rate >= thresholds["critical"]:
        return "critical"

    if issue_rate >= thresholds["high"]:
        return "high"

    if issue_rate >= thresholds["medium"]:
        return "medium"

    if issue_rate > thresholds["low"]:
        return "low"

    return "none"


def detect_completeness_issues(
    completeness_report: pd.DataFrame,
    thresholds: dict[str, float]
) -> pd.DataFrame:

    if completeness_report.empty:
        return pd.DataFrame()

    issues = completeness_report.copy()

    issues["issue_rate"] = (
        1
        - issues["completeness_score"] / 100
    )

    issues = issues[
        issues["missing_count"] > 0
    ].copy()

    issues["issue_type"] = "missing_values"

    issues["severity"] = issues[
        "issue_rate"
    ].apply(
        lambda value: classify_severity(
            value,
            thresholds
        )
    )

    issues["description"] = (
        issues["missing_count"]
        .astype(str)
        + " missing values detected in "
        + issues["column"]
    )

    return issues[
        [
            "dataset",
            "column",
            "issue_type",
            "severity",
            "issue_rate",
            "description",
        ]
    ]


def detect_validity_issues(
    validity_report: pd.DataFrame,
    thresholds: dict[str, float]
) -> pd.DataFrame:

    if validity_report.empty:
        return pd.DataFrame()

    issues = validity_report[
        validity_report["invalid_count"] > 0
    ].copy()

    issues["issue_rate"] = (
        issues["invalid_percentage"] / 100
    )

    issues["issue_type"] = "invalid_values"

    issues["severity"] = issues[
        "issue_rate"
    ].apply(
        lambda value: classify_severity(
            value,
            thresholds
        )
    )

    issues["description"] = (
        issues["invalid_count"]
        .astype(str)
        + " invalid records violate rule: "
        + issues["rule_name"]
    )

    issues["column"] = issues[
        "column"
    ].fillna("")

    return issues[
        [
            "dataset",
            "column",
            "issue_type",
            "severity",
            "issue_rate",
            "description",
        ]
    ]


def detect_consistency_issues(
    consistency_report: pd.DataFrame,
    thresholds: dict[str, float]
) -> pd.DataFrame:

    if consistency_report.empty:
        return pd.DataFrame()

    issues = consistency_report[
        consistency_report["invalid_count"] > 0
    ].copy()

    issues["issue_rate"] = (
        issues["invalid_percentage"] / 100
    )

    issues["issue_type"] = "consistency_violation"

    issues["severity"] = issues[
        "issue_rate"
    ].apply(
        lambda value: classify_severity(
            value,
            thresholds
        )
    )

    issues["description"] = (
        issues["invalid_count"]
        .astype(str)
        + " records violate consistency rule: "
        + issues["rule_name"]
    )

    issues["column"] = (
        issues["left_column"]
        + " -> "
        + issues["right_column"]
    )

    return issues[
        [
            "dataset",
            "column",
            "issue_type",
            "severity",
            "issue_rate",
            "description",
        ]
    ]


def detect_integrity_issues(
    integrity_report: pd.DataFrame,
    thresholds: dict[str, float]
) -> pd.DataFrame:

    if integrity_report.empty:
        return pd.DataFrame()

    issues = integrity_report[
        integrity_report[
            "unmatched_records"
        ] > 0
    ].copy()

    issues["issue_rate"] = (
        issues["unmatched_percentage"] / 100
    )

    issues["issue_type"] = (
        "referential_integrity_violation"
    )

    issues["severity"] = issues[
        "issue_rate"
    ].apply(
        lambda value: classify_severity(
            value,
            thresholds
        )
    )

    issues["description"] = (
        issues["unmatched_records"]
        .astype(str)
        + " records do not match parent relationship: "
        + issues["relationship"]
    )

    issues["column"] = issues[
        "child_column"
    ]

    issues["dataset"] = issues[
        "child_dataset"
    ]

    return issues[
        [
            "dataset",
            "column",
            "issue_type",
            "severity",
            "issue_rate",
            "description",
        ]
    ]


def combine_detected_issues(
    issue_reports: list[pd.DataFrame]
) -> pd.DataFrame:
    """
    Combine all issue categories.
    """

    valid_reports = [
        report
        for report in issue_reports
        if not report.empty
    ]

    if not valid_reports:

        return pd.DataFrame(
            columns=[
                "dataset",
                "column",
                "issue_type",
                "severity",
                "issue_rate",
                "description",
            ]
        )

    issues = pd.concat(
        valid_reports,
        ignore_index=True
    )

    severity_order = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
        "none": 0,
    }

    issues["severity_rank"] = (
        issues["severity"]
        .map(severity_order)
    )

    return (
        issues
        .sort_values(
            [
                "severity_rank",
                "issue_rate",
            ],
            ascending=[
                False,
                False,
            ]
        )
        .drop(
            columns="severity_rank"
        )
        .reset_index(drop=True)
    )