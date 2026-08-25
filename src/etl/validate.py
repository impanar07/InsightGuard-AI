from typing import Dict, List

import pandas as pd


def calculate_dataset_profile(
    datasets: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Generate high-level dataset statistics.
    """

    records = []

    for dataset_name, df in datasets.items():

        records.append({
            "dataset": dataset_name,
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": int(
                df.isna().sum().sum()
            ),
            "exact_duplicate_rows": int(
                df.duplicated().sum()
            )
        })

    return pd.DataFrame(records)


def validate_key_integrity(
    datasets: Dict[str, pd.DataFrame],
    key_definitions: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Validate primary and composite keys.
    """

    records = []

    for dataset_name, key_columns in (
        key_definitions.items()
    ):

        df = datasets[dataset_name]

        missing_keys = (
            df[key_columns]
            .isna()
            .any(axis=1)
            .sum()
        )

        duplicate_keys = (
            df.duplicated(
                subset=key_columns
            )
            .sum()
        )

        records.append({
            "dataset": dataset_name,
            "key_columns": ", ".join(
                key_columns
            ),
            "total_rows": len(df),
            "missing_key_rows": int(
                missing_keys
            ),
            "duplicate_key_rows": int(
                duplicate_keys
            ),
            "is_valid": (
                missing_keys == 0
                and duplicate_keys == 0
            )
        })

    return pd.DataFrame(records)


def validate_referential_integrity(
    datasets: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Validate important foreign key relationships.
    """

    relationships = [
        (
            "orders",
            "customer_id",
            "customers",
            "customer_id"
        ),
        (
            "order_items",
            "order_id",
            "orders",
            "order_id"
        ),
        (
            "order_items",
            "product_id",
            "products",
            "product_id"
        ),
        (
            "order_items",
            "seller_id",
            "sellers",
            "seller_id"
        ),
        (
            "order_payments",
            "order_id",
            "orders",
            "order_id"
        ),
        (
            "order_reviews",
            "order_id",
            "orders",
            "order_id"
        ),
    ]

    records = []

    for (
        child_dataset,
        child_column,
        parent_dataset,
        parent_column
    ) in relationships:

        child_df = datasets[
            child_dataset
        ]

        parent_df = datasets[
            parent_dataset
        ]

        child_values = (
            child_df[child_column]
            .dropna()
        )

        parent_values = set(
            parent_df[parent_column]
            .dropna()
        )

        unmatched_count = (
            ~child_values.isin(
                parent_values
            )
        ).sum()

        records.append({
            "relationship": (
                f"{child_dataset}.{child_column}"
                " -> "
                f"{parent_dataset}.{parent_column}"
            ),
            "total_child_records": len(
                child_values
            ),
            "unmatched_records": int(
                unmatched_count
            ),
            "integrity_valid": (
                unmatched_count == 0
            )
        })

    return pd.DataFrame(records)


def validate_business_rules(
    datasets: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Validate basic business rules.

    Invalid records are counted but not
    automatically deleted.
    """

    records = []

    # Order item prices

    order_items = datasets["order_items"]

    invalid_price = (
        order_items["price"] < 0
    ).sum()

    invalid_freight = (
        order_items["freight_value"] < 0
    ).sum()

    records.append({
        "dataset": "order_items",
        "rule": "price >= 0",
        "invalid_records": int(
            invalid_price
        )
    })

    records.append({
        "dataset": "order_items",
        "rule": "freight_value >= 0",
        "invalid_records": int(
            invalid_freight
        )
    })

    # Payment rules

    payments = datasets[
        "order_payments"
    ]

    invalid_payment_value = (
        payments["payment_value"] < 0
    ).sum()

    invalid_installments = (
        payments[
            "payment_installments"
        ] < 0
    ).sum()

    records.append({
        "dataset": "order_payments",
        "rule": "payment_value >= 0",
        "invalid_records": int(
            invalid_payment_value
        )
    })

    records.append({
        "dataset": "order_payments",
        "rule": (
            "payment_installments >= 0"
        ),
        "invalid_records": int(
            invalid_installments
        )
    })

    # Review rules

    reviews = datasets[
        "order_reviews"
    ]

    invalid_review_score = (
        ~reviews["review_score"]
        .isin([1, 2, 3, 4, 5])
        &
        reviews["review_score"].notna()
    ).sum()

    records.append({
        "dataset": "order_reviews",
        "rule": (
            "review_score between 1 and 5"
        ),
        "invalid_records": int(
            invalid_review_score
        )
    })

    return pd.DataFrame(records)


def create_missing_value_report(
    datasets: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Create a detailed missing value report.
    """

    records = []

    for dataset_name, df in datasets.items():

        for column in df.columns:

            missing_count = (
                df[column]
                .isna()
                .sum()
            )

            missing_percentage = (
                missing_count
                / len(df)
                * 100
                if len(df) > 0
                else 0
            )

            records.append({
                "dataset": dataset_name,
                "column": column,
                "missing_count": int(
                    missing_count
                ),
                "missing_percentage": round(
                    missing_percentage,
                    4
                )
            })

    return pd.DataFrame(records)