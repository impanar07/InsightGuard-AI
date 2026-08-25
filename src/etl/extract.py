from pathlib import Path
from typing import Dict

import pandas as pd


def validate_required_files(
    data_path: Path,
    dataset_files: Dict[str, str]
) -> None:
    """
    Validate that all required dataset files exist.
    """

    missing_files = []

    for dataset_name, filename in dataset_files.items():

        file_path = data_path / filename

        if not file_path.exists():

            missing_files.append(
                f"{dataset_name}: {filename}"
            )

    if missing_files:

        missing_message = "\n".join(
            missing_files
        )

        raise FileNotFoundError(
            "Required dataset files are missing:\n"
            f"{missing_message}"
        )


def load_dataset(
    file_path: Path
) -> pd.DataFrame:
    """
    Load a single CSV dataset.
    """

    try:

        dataframe = pd.read_csv(
            file_path,
            low_memory=False
        )

        return dataframe

    except Exception as error:

        raise RuntimeError(
            f"Failed to load dataset: {file_path.name}"
        ) from error


def extract_datasets(
    data_path: Path,
    dataset_files: Dict[str, str]
) -> Dict[str, pd.DataFrame]:
    """
    Load all configured datasets.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing all extracted datasets.
    """

    validate_required_files(
        data_path=data_path,
        dataset_files=dataset_files
    )

    datasets = {}

    for dataset_name, filename in dataset_files.items():

        file_path = data_path / filename

        print(
            f"Extracting {dataset_name} "
            f"from {filename}"
        )

        datasets[dataset_name] = load_dataset(
            file_path
        )

    return datasets