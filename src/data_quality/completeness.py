from __future__ import annotations

from typing import Dict

import pandas as pd


def calculate_completeness(
    dataset_name: str,
    dataframe: pd.DataFrame
) -> tuple[float, pd.DataFrame]:
    """
    Calculate dataset and column-level completeness.

    Completeness Score:
        100 * (1 - missing values / total cells)
    """

    total_cells = (
        dataframe.shape[0]
        * dataframe.shape[1]
    )

    total_missing = int(
        dataframe.isna().sum().sum()
    )

    if total_cells == 0:
        dataset_score = 100.0
    else:
        dataset_score = (
            1 - total_missing / total_cells
        ) * 100

    records = []

    for column in dataframe.columns:

        total_rows = len(dataframe)

        missing_count = int(
            dataframe[column].isna().sum()
        )

        completeness_score = (
            1 - missing_count / total_rows
        ) * 100 if total_rows > 0 else 100.0

        records.append({
            "dataset": dataset_name,
            "column": column,
            "missing_count": missing_count,
            "completeness_score": round(
                completeness_score,
                4
            ),
        })

    return (
        round(dataset_score, 4),
        pd.DataFrame(records)
    )


def calculate_all_completeness(
    datasets: Dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:

    dataset_results = []
    column_results = []

    for dataset_name, dataframe in datasets.items():

        score, column_report = (
            calculate_completeness(
                dataset_name,
                dataframe
            )
        )

        dataset_results.append({
            "dataset": dataset_name,
            "completeness_score": score,
        })

        column_results.append(column_report)

    return (
        pd.DataFrame(dataset_results),
        pd.concat(
            column_results,
            ignore_index=True
        )
        if column_results
        else pd.DataFrame()
    )