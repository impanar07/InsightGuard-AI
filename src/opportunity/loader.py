from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def load_datasets(
    data_path: Path,
    dataset_files: Dict[str, str]
) -> Dict[str, pd.DataFrame]:
    """
    Load all required cleaned datasets.

    Raises:
        FileNotFoundError:
            If a required dataset does not exist.

        ValueError:
            If a dataset is empty.
    """

    datasets = {}

    for dataset_name, filename in (
        dataset_files.items()
    ):

        file_path = data_path / filename

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required dataset not found: "
                f"{file_path}"
            )

        dataframe = pd.read_csv(
            file_path,
            low_memory=False
        )

        if dataframe.empty:

            raise ValueError(
                f"Dataset is empty: "
                f"{dataset_name}"
            )

        datasets[dataset_name] = dataframe

        print(
            f"Loaded {dataset_name}: "
            f"{len(dataframe):,} rows"
        )

    return datasets


def convert_datetime_columns(
    datasets: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Convert known timestamp columns into datetime.
    """

    datetime_columns = {
        "orders": [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
        "order_items": [
            "shipping_limit_date",
        ],
        "order_reviews": [
            "review_creation_date",
            "review_answer_timestamp",
        ],
    }

    for dataset_name, columns in (
        datetime_columns.items()
    ):

        if dataset_name not in datasets:
            continue

        dataframe = datasets[dataset_name].copy()

        for column in columns:

            if column in dataframe.columns:

                dataframe[column] = (
                    pd.to_datetime(
                        dataframe[column],
                        errors="coerce"
                    )
                )

        datasets[dataset_name] = dataframe

    return datasets