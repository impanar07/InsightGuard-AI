from typing import Dict, List

import numpy as np
import pandas as pd


def standardize_column_names(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Standardize column names.

    Operations:
    - Convert to lowercase
    - Remove leading/trailing spaces
    - Replace spaces with underscores
    """

    df = dataframe.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


def standardize_string_columns(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove leading and trailing whitespace
    from string columns.
    """

    df = dataframe.copy()

    string_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in string_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    return df


def convert_date_columns(
    dataframe: pd.DataFrame,
    date_columns: List[str]
) -> pd.DataFrame:
    """
    Convert specified columns to datetime.
    Invalid values become NaT.
    """

    df = dataframe.copy()

    for column in date_columns:

        if column in df.columns:

            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

    return df


def replace_infinite_values(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Replace positive and negative infinity
    with NaN.
    """

    df = dataframe.copy()

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    return df


def remove_exact_duplicates(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove exact duplicate rows.

    Only complete duplicate rows are removed.
    """

    df = dataframe.copy()

    return df.drop_duplicates().reset_index(
        drop=True
    )


def transform_orders(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply order-specific transformations.
    """

    df = dataframe.copy()

    if "order_status" in df.columns:

        df["order_status"] = (
            df["order_status"]
            .str.lower()
            .str.strip()
        )

    return df


def transform_order_items(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply transformations to order items.
    """

    df = dataframe.copy()

    numeric_columns = [
        "price",
        "freight_value",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


def transform_order_payments(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply transformations to payment data.
    """

    df = dataframe.copy()

    numeric_columns = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    if "payment_type" in df.columns:

        df["payment_type"] = (
            df["payment_type"]
            .str.lower()
            .str.strip()
        )

    return df


def transform_reviews(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply transformations to review data.
    """

    df = dataframe.copy()

    if "review_score" in df.columns:

        df["review_score"] = pd.to_numeric(
            df["review_score"],
            errors="coerce"
        )

    return df


def transform_products(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply transformations to product data.
    """

    df = dataframe.copy()

    numeric_columns = [
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


def transform_geolocation(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply transformations to geolocation data.
    """

    df = dataframe.copy()

    numeric_columns = [
        "geolocation_lat",
        "geolocation_lng",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


def apply_dataset_transformations(
    datasets: Dict[str, pd.DataFrame],
    date_columns: Dict[str, List[str]]
) -> Dict[str, pd.DataFrame]:
    """
    Apply common and dataset-specific
    transformations to all datasets.
    """

    transformed_datasets = {}

    for dataset_name, dataframe in datasets.items():

        print(
            f"Transforming dataset: "
            f"{dataset_name}"
        )

        df = dataframe.copy()

        # Common transformations
        df = standardize_column_names(df)

        df = standardize_string_columns(df)

        df = replace_infinite_values(df)

        df = remove_exact_duplicates(df)

        # Date conversion
        if dataset_name in date_columns:

            df = convert_date_columns(
                dataframe=df,
                date_columns=date_columns[
                    dataset_name
                ]
            )

        # Dataset-specific transformations
        if dataset_name == "orders":

            df = transform_orders(df)

        elif dataset_name == "order_items":

            df = transform_order_items(df)

        elif dataset_name == "order_payments":

            df = transform_order_payments(df)

        elif dataset_name == "order_reviews":

            df = transform_reviews(df)

        elif dataset_name == "products":

            df = transform_products(df)

        elif dataset_name == "geolocation":

            df = transform_geolocation(df)

        transformed_datasets[
            dataset_name
        ] = df

    return transformed_datasets