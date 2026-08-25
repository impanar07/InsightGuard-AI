from pathlib import Path
from typing import Dict

import pandas as pd


def save_clean_datasets(
    datasets: Dict[str, pd.DataFrame],
    output_path: Path
) -> None:
    """
    Save cleaned datasets as CSV files.
    """

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    for dataset_name, dataframe in datasets.items():

        output_file = (
            output_path
            / f"{dataset_name}_clean.csv"
        )

        dataframe.to_csv(
            output_file,
            index=False
        )

        print(
            f"Saved cleaned dataset: "
            f"{output_file.name}"
        )


def save_report(
    dataframe: pd.DataFrame,
    output_path: Path,
    filename: str
) -> None:
    """
    Save a DataFrame as a CSV report.
    """

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_path / filename
    )

    dataframe.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved report: "
        f"{output_file.name}"
    )