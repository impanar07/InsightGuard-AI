from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def build_order_item_fact(
    datasets: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Build an order-item-level analytical fact table.

    Combines:
    - Order items
    - Orders
    - Customers
    - Products
    - Sellers
    """

    order_items = datasets["order_items"].copy()
    orders = datasets["orders"].copy()
    customers = datasets["customers"].copy()
    products = datasets["products"].copy()
    sellers = datasets["sellers"].copy()

    # --------------------------------------------------
    # ORDERS
    # --------------------------------------------------

    order_columns = [
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    available_order_columns = [
        column
        for column in order_columns
        if column in orders.columns
    ]

    fact = order_items.merge(
        orders[available_order_columns],
        on="order_id",
        how="left"
    )

    # --------------------------------------------------
    # CUSTOMERS
    # --------------------------------------------------

    customer_columns = [
        "customer_id"
    ]

    optional_customer_columns = [
        "customer_unique_id",
        "customer_city",
        "customer_state",
    ]

    for column in optional_customer_columns:
        if column in customers.columns:
            customer_columns.append(column)

    fact = fact.merge(
        customers[customer_columns],
        on="customer_id",
        how="left"
    )

    # --------------------------------------------------
    # PRODUCTS
    # --------------------------------------------------

    product_columns = [
        "product_id"
    ]

    optional_product_columns = [
        "product_category_name",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]

    for column in optional_product_columns:
        if column in products.columns:
            product_columns.append(column)

    fact = fact.merge(
        products[product_columns],
        on="product_id",
        how="left"
    )

    # --------------------------------------------------
    # SELLERS
    # --------------------------------------------------

    seller_columns = [
        "seller_id"
    ]

    optional_seller_columns = [
        "seller_city",
        "seller_state",
    ]

    for column in optional_seller_columns:
        if column in sellers.columns:
            seller_columns.append(column)

    fact = fact.merge(
        sellers[seller_columns],
        on="seller_id",
        how="left"
    )

    # --------------------------------------------------
    # REVENUE
    # --------------------------------------------------

    if "price" not in fact.columns:
        raise KeyError(
            "Column 'price' is required in order_items."
        )

    if "freight_value" not in fact.columns:
        raise KeyError(
            "Column 'freight_value' is required in order_items."
        )

    fact["price"] = pd.to_numeric(
        fact["price"],
        errors="coerce"
    )

    fact["freight_value"] = pd.to_numeric(
        fact["freight_value"],
        errors="coerce"
    )

    fact["item_revenue"] = (
        fact["price"].fillna(0)
        + fact["freight_value"].fillna(0)
    )

    return fact


def build_order_fact(
    datasets: Dict[str, pd.DataFrame],
    order_item_fact: pd.DataFrame
) -> pd.DataFrame:
    """
    Build an order-level analytical fact table.

    Includes:
    - Revenue metrics
    - Item counts
    - Customer information
    - Delivery performance
    - Delay indicators
    """

    orders = datasets["orders"].copy()
    customers = datasets["customers"].copy()

    # --------------------------------------------------
    # ORDER REVENUE
    # --------------------------------------------------

    order_revenue = (
        order_item_fact
        .groupby(
            "order_id",
            as_index=False
        )
        .agg(
            total_revenue=(
                "item_revenue",
                "sum"
            ),
            total_product_value=(
                "price",
                "sum"
            ),
            total_freight_value=(
                "freight_value",
                "sum"
            ),
            total_items=(
                "order_item_id",
                "count"
            ),
        )
    )

    # --------------------------------------------------
    # MERGE REVENUE
    # --------------------------------------------------

    order_fact = orders.merge(
        order_revenue,
        on="order_id",
        how="left"
    )

    revenue_columns = [
        "total_revenue",
        "total_product_value",
        "total_freight_value",
        "total_items",
    ]

    for column in revenue_columns:
        if column in order_fact.columns:
            order_fact[column] = (
                order_fact[column].fillna(0)
            )

    # --------------------------------------------------
    # CUSTOMER INFORMATION
    # --------------------------------------------------

    customer_columns = [
        "customer_id"
    ]

    optional_customer_columns = [
        "customer_unique_id",
        "customer_city",
        "customer_state",
    ]

    for column in optional_customer_columns:
        if column in customers.columns:
            customer_columns.append(column)

    customer_data = (
        customers[customer_columns]
        .drop_duplicates(
            subset=["customer_id"]
        )
    )

    # Remove existing duplicate columns if present
    columns_to_remove = [
        column
        for column in optional_customer_columns
        if column in order_fact.columns
    ]

    if columns_to_remove:
        order_fact = order_fact.drop(
            columns=columns_to_remove
        )

    order_fact = order_fact.merge(
        customer_data,
        on="customer_id",
        how="left"
    )

    # --------------------------------------------------
    # DELIVERY TIME
    # --------------------------------------------------

    if (
        "order_delivered_customer_date"
        in order_fact.columns
        and "order_purchase_timestamp"
        in order_fact.columns
    ):

        order_fact["actual_delivery_days"] = (
            order_fact[
                "order_delivered_customer_date"
            ]
            - order_fact[
                "order_purchase_timestamp"
            ]
        ).dt.total_seconds() / 86400

    else:
        order_fact["actual_delivery_days"] = np.nan

    # --------------------------------------------------
    # DELIVERY DELAY
    # --------------------------------------------------

    if (
        "order_estimated_delivery_date"
        in order_fact.columns
        and "order_delivered_customer_date"
        in order_fact.columns
    ):

        order_fact["delivery_delay_days"] = (
            order_fact[
                "order_delivered_customer_date"
            ]
            - order_fact[
                "order_estimated_delivery_date"
            ]
        ).dt.total_seconds() / 86400

    else:
        order_fact["delivery_delay_days"] = np.nan

    # --------------------------------------------------
    # DELAY FLAG
    # --------------------------------------------------

    order_fact["is_delayed"] = (
        order_fact["delivery_delay_days"] > 0
    ).astype("Int64")

    return order_fact


def build_review_features(
    datasets: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Build order-level review features.
    """

    reviews = datasets["order_reviews"].copy()

    if "review_score" not in reviews.columns:

        return pd.DataFrame(
            columns=[
                "order_id",
                "average_review_score",
            ]
        )

    review_features = (
        reviews
        .groupby(
            "order_id",
            as_index=False
        )
        .agg(
            average_review_score=(
                "review_score",
                "mean"
            )
        )
    )

    return review_features


def add_review_features(
    order_fact: pd.DataFrame,
    review_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Add review information to the order fact table.
    """

    if review_features.empty:

        order_fact[
            "average_review_score"
        ] = np.nan

        return order_fact

    return order_fact.merge(
        review_features,
        on="order_id",
        how="left"
    )


def build_customer_features(
    order_fact: pd.DataFrame,
    customers: pd.DataFrame
) -> pd.DataFrame:
    """
    Build customer-level analytical features.

    Includes:
    - Total orders
    - Total revenue
    - Average order value
    - Recency
    - Customer lifetime duration
    """

    delivered_orders = (
        order_fact[
            order_fact["order_status"]
            .eq("delivered")
        ]
        .copy()
    )

    if delivered_orders.empty:
        raise ValueError(
            "No delivered orders available "
            "for customer feature engineering."
        )

    # Use customer_unique_id when available
    if "customer_unique_id" in delivered_orders.columns:
        customer_id_column = (
            "customer_unique_id"
        )
    else:
        customer_id_column = "customer_id"

    reference_date = (
        delivered_orders[
            "order_purchase_timestamp"
        ].max()
    )

    customer_features = (
        delivered_orders
        .groupby(
            customer_id_column,
            as_index=False
        )
        .agg(
            total_orders=(
                "order_id",
                "nunique"
            ),
            total_revenue=(
                "total_revenue",
                "sum"
            ),
            average_order_value=(
                "total_revenue",
                "mean"
            ),
            last_purchase_date=(
                "order_purchase_timestamp",
                "max"
            ),
            first_purchase_date=(
                "order_purchase_timestamp",
                "min"
            ),
        )
    )

    customer_features["recency_days"] = (
        reference_date
        - customer_features[
            "last_purchase_date"
        ]
    ).dt.days

    customer_features[
        "customer_lifetime_days"
    ] = (
        customer_features[
            "last_purchase_date"
        ]
        - customer_features[
            "first_purchase_date"
        ]
    ).dt.days

    return customer_features


def build_all_features(
    datasets: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Main feature engineering pipeline.

    Builds all analytical fact tables required
    by the Opportunity AI Engine.
    """

    # --------------------------------------------------
    # ORDER ITEM FACT
    # --------------------------------------------------

    print(
        "Building order-item fact table..."
    )

    order_item_fact = build_order_item_fact(
        datasets
    )

    # --------------------------------------------------
    # ORDER FACT
    # --------------------------------------------------

    print(
        "Building order fact table..."
    )

    order_fact = build_order_fact(
        datasets,
        order_item_fact
    )

    # --------------------------------------------------
    # REVIEW FEATURES
    # --------------------------------------------------

    print(
        "Building review features..."
    )

    review_features = build_review_features(
        datasets
    )

    order_fact = add_review_features(
        order_fact,
        review_features
    )

    # --------------------------------------------------
    # CUSTOMER FEATURES
    # --------------------------------------------------

    print(
        "Building customer features..."
    )

    customer_features = (
        build_customer_features(
            order_fact,
            datasets["customers"]
        )
    )

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    if "customer_state" not in order_fact.columns:
        raise KeyError(
            "Feature engineering failed: "
            "'customer_state' was not created "
            "in order_fact."
        )

    return {
        "order_item_fact": order_item_fact,
        "order_fact": order_fact,
        "customer_features": customer_features,
    }