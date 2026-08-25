from __future__ import annotations

from typing import Dict, List

import pandas as pd


def evaluate_consistency_rule(
    dataframe: pd.DataFrame,
    rule: Dict
) -> pd.Series:
    """
    Return True for inconsistent rows.
    """

    left_column = rule["left_column"]
    right_column = rule["right_column"]

    left = pd.to_datetime(
        dataframe[left_column],
        errors="coerce"
    )

    right = pd.to_datetime(
        dataframe[right_column],
        errors="coerce"
    )

    comparable = (
        left.notna()
        & right.notna()
    )

    if rule["operator"] == "greater_than_or_equal":

        invalid_mask = (
            comparable
            & (left < right)
        )

    else:
        raise ValueError(
            f"Unsupported consistency operator: "
            f"{rule['operator']}"
        )

    return invalid_mask


def calculate_consistency(
    dataset_name: str,
    dataframe: pd.DataFrame,
    rules: List[Dict]
) -> tuple[float, pd.DataFrame]:
    """
    Calculate consistency score using configured
    cross-column business rules.
    """

    dataset_rules = [
        rule
        for rule in rules
        if rule["dataset"] == dataset_name
    ]

    if not dataset_rules:

        return 100.0, pd.DataFrame()

    reports = []

    total_invalid = 0
    total_evaluated = 0

    for rule in dataset_rules:

        required_columns = [
            rule["left_column"],
            rule["right_column"],
        ]

        if not all(
            column in dataframe.columns
            for column in required_columns
        ):
            continue

        left = dataframe[
            rule["left_column"]
        ]

        right = dataframe[
            rule["right_column"]
        ]

        evaluated_mask = (
            left.notna()
            & right.notna()
        )

        evaluated_count = int(
            evaluated_mask.sum()
        )

        invalid_mask = (
            evaluate_consistency_rule(
                dataframe,
                rule
            )
        )

        invalid_count = int(
            invalid_mask.sum()
        )

        consistency_score = (
            1
            - invalid_count / evaluated_count
        ) * 100 if evaluated_count > 0 else 100.0

        total_invalid += invalid_count
        total_evaluated += evaluated_count

        reports.append({
            "dataset": dataset_name,
            "rule_name": rule["rule_name"],
            "left_column": rule["left_column"],
            "right_column": rule["right_column"],
            "invalid_count": invalid_count,
            "evaluated_count": evaluated_count,
            "invalid_percentage": round(
                invalid_count
                / evaluated_count
                * 100,
                4
            ) if evaluated_count > 0 else 0.0,
            "consistency_score": round(
                consistency_score,
                4
            ),
        })

    dataset_score = (
        1
        - total_invalid / total_evaluated
    ) * 100 if total_evaluated > 0 else 100.0

    return (
        round(dataset_score, 4),
        pd.DataFrame(reports)
    )


def calculate_all_consistency(
    datasets: Dict[str, pd.DataFrame],
    consistency_rules: List[Dict]
) -> tuple[pd.DataFrame, pd.DataFrame]:

    dataset_results = []
    reports = []

    for dataset_name, dataframe in datasets.items():

        score, report = calculate_consistency(
            dataset_name=dataset_name,
            dataframe=dataframe,
            rules=consistency_rules
        )

        dataset_results.append({
            "dataset": dataset_name,
            "consistency_score": score,
        })

        if not report.empty:
            reports.append(report)

    return (
        pd.DataFrame(dataset_results),
        pd.concat(
            reports,
            ignore_index=True
        )
        if reports
        else pd.DataFrame()
    )