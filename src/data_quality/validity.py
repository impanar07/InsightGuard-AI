from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def evaluate_validity_rule(
    series: pd.Series,
    rule: Dict[str, Any]
) -> pd.Series:
    """
    Return a Boolean mask where True means invalid.

    Missing values are excluded here because
    completeness handles missingness separately.
    """

    non_null_mask = series.notna()

    invalid_mask = pd.Series(
        False,
        index=series.index
    )

    rule_type = rule["rule_type"]

    if rule_type == "minimum":

        invalid_mask = (
            non_null_mask
            & (series < rule["minimum"])
        )

    elif rule_type == "maximum":

        invalid_mask = (
            non_null_mask
            & (series > rule["maximum"])
        )

    elif rule_type == "range":

        invalid_mask = (
            non_null_mask
            & (
                (series < rule["minimum"])
                | (series > rule["maximum"])
            )
        )

    elif rule_type == "allowed_values":

        invalid_mask = (
            non_null_mask
            & ~series.isin(
                rule["allowed_values"]
            )
        )

    else:
        raise ValueError(
            f"Unsupported rule type: {rule_type}"
        )

    return invalid_mask


def calculate_validity(
    dataset_name: str,
    dataframe: pd.DataFrame,
    rules: list[dict[str, Any]]
) -> tuple[float, pd.DataFrame]:
    """
    Calculate validity score for configured rules.
    """

    if not rules:
        return 100.0, pd.DataFrame(
            columns=[
                "dataset",
                "column",
                "rule_name",
                "invalid_count",
                "evaluated_count",
                "invalid_percentage",
                "validity_score",
            ]
        )

    reports = []

    total_invalid = 0
    total_evaluated = 0

    for rule in rules:

        column = rule["column"]

        if column not in dataframe.columns:
            continue

        series = dataframe[column]

        invalid_mask = evaluate_validity_rule(
            series=series,
            rule=rule
        )

        invalid_count = int(
            invalid_mask.sum()
        )

        evaluated_count = int(
            series.notna().sum()
        )

        validity_score = (
            1 - invalid_count / evaluated_count
        ) * 100 if evaluated_count > 0 else 100.0

        total_invalid += invalid_count
        total_evaluated += evaluated_count

        reports.append({
            "dataset": dataset_name,
            "column": column,
            "rule_name": rule["rule_name"],
            "invalid_count": invalid_count,
            "evaluated_count": evaluated_count,
            "invalid_percentage": round(
                invalid_count
                / evaluated_count
                * 100,
                4
            ) if evaluated_count > 0 else 0.0,
            "validity_score": round(
                validity_score,
                4
            ),
        })

    dataset_score = (
        1 - total_invalid / total_evaluated
    ) * 100 if total_evaluated > 0 else 100.0

    return (
        round(dataset_score, 4),
        pd.DataFrame(reports)
    )


def calculate_all_validity(
    datasets: Dict[str, pd.DataFrame],
    validity_rules: Dict[
        str,
        list[dict[str, Any]]
    ]
) -> tuple[pd.DataFrame, pd.DataFrame]:

    dataset_results = []
    reports = []

    for dataset_name, dataframe in datasets.items():

        rules = validity_rules.get(
            dataset_name,
            []
        )

        score, report = calculate_validity(
            dataset_name=dataset_name,
            dataframe=dataframe,
            rules=rules
        )

        dataset_results.append({
            "dataset": dataset_name,
            "validity_score": score,
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