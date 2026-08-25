from __future__ import annotations

from typing import Dict, List

import pandas as pd


def calculate_timeliness(
    dataset_name: str,
    dataframe: pd.DataFrame,
    date_columns: List[str]
) -> tuple[float, pd.DataFrame]:
    """
    Calculate timestamp availability and usability.

    Timeliness in this historical project means:
    - Timestamp can be parsed
    - Timestamp is present where available
    """

    if not date_columns:

        return 100.0, pd.DataFrame()

    reports = []

    total_missing_or_invalid = 0
    total_values = 0

    for column in date_columns:

        if column not in dataframe.columns:
            continue

        parsed_dates = pd.to_datetime(
            dataframe[column],
            errors="coerce"
        )

        invalid_or_missing = int(
            parsed_dates.isna().sum()
        )

        total_count = len(parsed_dates)

        score = (
            1
            - invalid_or_missing
            / total_count
        ) * 100 if total_count > 0 else 100.0

        total_missing_or_invalid += (
            invalid_or_missing
        )

        total_values += total_count

        reports.append({
            "dataset": dataset_name,
            "column": column,
            "invalid_or_missing_dates": (
                invalid_or_missing
            ),
            "total_values": total_count,
            "timeliness_score": round(
                score,
                4
            ),
        })

    dataset_score = (
        1
        - total_missing_or_invalid
        / total_values
    ) * 100 if total_values > 0 else 100.0

    return (
        round(dataset_score, 4),
        pd.DataFrame(reports)
    )


def calculate_all_timeliness(
    datasets: Dict[str, pd.DataFrame],
    date_columns: Dict[str, List[str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:

    dataset_results = []
    reports = []

    for dataset_name, dataframe in datasets.items():

        configured_columns = date_columns.get(
            dataset_name,
            []
        )

        score, report = calculate_timeliness(
            dataset_name=dataset_name,
            dataframe=dataframe,
            date_columns=configured_columns
        )

        dataset_results.append({
            "dataset": dataset_name,
            "timeliness_score": score,
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