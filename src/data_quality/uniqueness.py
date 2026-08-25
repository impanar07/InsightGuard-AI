from __future__ import annotations

from typing import Dict, List

import pandas as pd


def calculate_uniqueness(
    dataset_name: str,
    dataframe: pd.DataFrame,
    key_columns: List[str] | None = None
) -> tuple[float, pd.DataFrame]:
    """
    Calculate uniqueness score.

    The score is based on:
    - Exact duplicate rows
    - Duplicate configured key combinations
    """

    total_rows = len(dataframe)

    exact_duplicate_count = int(
        dataframe.duplicated().sum()
    )

    exact_duplicate_rate = (
        exact_duplicate_count / total_rows
        if total_rows > 0
        else 0.0
    )

    key_duplicate_count = 0
    key_duplicate_rate = 0.0

    if key_columns:
        available_columns = [
            column
            for column in key_columns
            if column in dataframe.columns
        ]

        if len(available_columns) == len(key_columns):

            key_duplicate_count = int(
                dataframe.duplicated(
                    subset=key_columns
                ).sum()
            )

            key_duplicate_rate = (
                key_duplicate_count
                / total_rows
                if total_rows > 0
                else 0.0
            )

    combined_duplicate_rate = max(
        exact_duplicate_rate,
        key_duplicate_rate
    )

    uniqueness_score = (
        1 - combined_duplicate_rate
    ) * 100

    report = pd.DataFrame([{
        "dataset": dataset_name,
        "total_rows": total_rows,
        "exact_duplicate_rows": exact_duplicate_count,
        "exact_duplicate_rate": round(
            exact_duplicate_rate * 100,
            4
        ),
        "key_duplicate_rows": key_duplicate_count,
        "key_duplicate_rate": round(
            key_duplicate_rate * 100,
            4
        ),
        "uniqueness_score": round(
            uniqueness_score,
            4
        ),
    }])

    return (
        round(uniqueness_score, 4),
        report
    )


def calculate_all_uniqueness(
    datasets: Dict[str, pd.DataFrame],
    key_definitions: Dict[str, List[str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:

    dataset_results = []
    reports = []

    for dataset_name, dataframe in datasets.items():

        key_columns = key_definitions.get(
            dataset_name
        )

        score, report = calculate_uniqueness(
            dataset_name=dataset_name,
            dataframe=dataframe,
            key_columns=key_columns
        )

        dataset_results.append({
            "dataset": dataset_name,
            "uniqueness_score": score,
        })

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