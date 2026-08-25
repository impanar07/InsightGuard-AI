from __future__ import annotations

from typing import Dict, List

import pandas as pd


def calculate_referential_integrity(
    datasets: Dict[str, pd.DataFrame],
    rules: List[Dict]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate foreign-key relationships.

    Referential Integrity Score:
        100 * (1 - unmatched child records /
               non-null child records)
    """

    relationship_reports = []
    dataset_issues = {}

    for rule in rules:

        child_dataset = rule["child_dataset"]
        child_column = rule["child_column"]

        parent_dataset = rule["parent_dataset"]
        parent_column = rule["parent_column"]

        if (
            child_dataset not in datasets
            or parent_dataset not in datasets
        ):
            continue

        child_df = datasets[child_dataset]
        parent_df = datasets[parent_dataset]

        if (
            child_column not in child_df.columns
            or parent_column not in parent_df.columns
        ):
            continue

        child_values = child_df[
            child_column
        ].dropna()

        parent_values = set(
            parent_df[
                parent_column
            ].dropna()
        )

        unmatched_mask = (
            ~child_values.isin(parent_values)
        )

        unmatched_count = int(
            unmatched_mask.sum()
        )

        evaluated_count = len(
            child_values
        )

        integrity_score = (
            1
            - unmatched_count
            / evaluated_count
        ) * 100 if evaluated_count > 0 else 100.0

        relationship_reports.append({
            "relationship": rule["name"],
            "child_dataset": child_dataset,
            "child_column": child_column,
            "parent_dataset": parent_dataset,
            "parent_column": parent_column,
            "evaluated_records": evaluated_count,
            "unmatched_records": unmatched_count,
            "unmatched_percentage": round(
                unmatched_count
                / evaluated_count
                * 100,
                4
            ) if evaluated_count > 0 else 0.0,
            "referential_integrity_score": round(
                integrity_score,
                4
            ),
        })

        dataset_issues.setdefault(
            child_dataset,
            []
        ).append(
            (
                unmatched_count,
                evaluated_count
            )
        )

    dataset_results = []

    for dataset_name in datasets:

        relationships = dataset_issues.get(
            dataset_name,
            []
        )

        if not relationships:

            score = 100.0

        else:

            total_unmatched = sum(
                item[0]
                for item in relationships
            )

            total_evaluated = sum(
                item[1]
                for item in relationships
            )

            score = (
                1
                - total_unmatched
                / total_evaluated
            ) * 100 if total_evaluated > 0 else 100.0

        dataset_results.append({
            "dataset": dataset_name,
            "referential_integrity_score": round(
                score,
                4
            ),
        })

    return (
        pd.DataFrame(dataset_results),
        pd.DataFrame(relationship_reports)
    )