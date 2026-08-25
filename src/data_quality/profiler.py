
from __future__ import annotations

from typing import Dict

import pandas as pd


def profile_dataset(
    dataset_name: str,
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a column-level profile for a dataset.
    """

    records = []

    total_rows = len(dataframe)

    for column in dataframe.columns:

        series = dataframe[column]

        null_count = int(series.isna().sum())

        non_null_count = int(series.notna().sum())

        unique_count = int(
            series.nunique(dropna=True)
        )

        unique_percentage = (
            unique_count / non_null_count * 100
            if non_null_count > 0
            else 0.0
        )

        records.append({
            "dataset": dataset_name,
            "column": column,
            "data_type": str(series.dtype),
            "total_rows": total_rows,
            "non_null_count": non_null_count,
            "null_count": null_count,
            "null_percentage": round(
                null_count / total_rows * 100,
                4
            ) if total_rows > 0 else 0.0,
            "unique_count": unique_count,
            "unique_percentage": round(
                unique_percentage,
                4
            ),
        })

    return pd.DataFrame(records)


def profile_all_datasets(
    datasets: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Create a complete profile for all datasets.
    """

    profiles = []

    for dataset_name, dataframe in datasets.items():

        profile = profile_dataset(
            dataset_name=dataset_name,
            dataframe=dataframe
        )

        profiles.append(profile)

    if not profiles:
        return pd.DataFrame()

    return pd.concat(
        profiles,
        ignore_index=True
    )