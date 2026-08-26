from __future__ import annotations

import pandas as pd

from src.opportunity.scoring import (
    min_max_score,
    weighted_score,
    assign_opportunity_level,
)


def analyze_product_opportunities(
    order_item_fact: pd.DataFrame,
    opportunity_weights: dict[str, float],
    thresholds: dict[str, float]
) -> pd.DataFrame:
    """
    Identify product-category business opportunities.

    Opportunity signals include:

    - Revenue potential
    - Order demand
    - Purchase growth
    - Customer satisfaction
    - Revenue efficiency
    """

    required_columns = [
        "order_id",
        "product_category_name",
        "item_revenue",
        "price",
        "order_purchase_timestamp",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in order_item_fact.columns
    ]

    if missing_columns:

        raise KeyError(
            "Missing required product columns: "
            f"{missing_columns}"
        )

    data = order_item_fact.copy()

    data = data[
        data["product_category_name"]
        .notna()
    ].copy()

    data["purchase_month"] = (
        data[
            "order_purchase_timestamp"
        ]
        .dt.to_period("M")
    )

    category_summary = (
        data
        .groupby(
            "product_category_name",
            as_index=False
        )
        .agg(
            total_orders=(
                "order_id",
                "nunique"
            ),
            total_revenue=(
                "item_revenue",
                "sum"
            ),
            average_price=(
                "price",
                "mean"
            ),
            item_count=(
                "order_id",
                "count"
            ),
        )
    )

    if (
        "average_review_score"
        in data.columns
    ):

        review_summary = (
            data
            .groupby(
                "product_category_name",
                as_index=False
            )
            .agg(
                average_review_score=(
                    "average_review_score",
                    "mean"
                )
            )
        )

        category_summary = (
            category_summary.merge(
                review_summary,
                on="product_category_name",
                how="left"
            )
        )

    else:

        category_summary[
            "average_review_score"
        ] = 3.0

    monthly_orders = (
        data
        .groupby(
            [
                "product_category_name",
                "purchase_month",
            ],
            as_index=False
        )
        .agg(
            monthly_orders=(
                "order_id",
                "nunique"
            )
        )
    )

    growth_records = []

    for category, group in (
        monthly_orders.groupby(
            "product_category_name"
        )
    ):

        group = group.sort_values(
            "purchase_month"
        )

        if len(group) < 2:

            growth_rate = 0.0

        else:

            first_value = (
                group[
                    "monthly_orders"
                ].iloc[0]
            )

            last_value = (
                group[
                    "monthly_orders"
                ].iloc[-1]
            )

            growth_rate = (
                (
                    last_value
                    - first_value
                )
                / first_value
                * 100
                if first_value > 0
                else 0.0
            )

        growth_records.append({
            "product_category_name": category,
            "growth_rate": growth_rate,
        })

    growth_dataframe = pd.DataFrame(
        growth_records
    )

    category_summary = (
        category_summary.merge(
            growth_dataframe,
            on="product_category_name",
            how="left"
        )
    )

    category_summary[
        "revenue_potential"
    ] = min_max_score(
        category_summary["total_revenue"]
    )

    category_summary[
        "order_demand"
    ] = min_max_score(
        category_summary["total_orders"]
    )

    category_summary[
        "growth_potential"
    ] = min_max_score(
        category_summary["growth_rate"]
    )

    category_summary[
        "customer_satisfaction"
    ] = (
        category_summary[
            "average_review_score"
        ]
        / 5
        * 100
    ).fillna(50)

    category_summary[
        "profit_proxy"
    ] = min_max_score(
        category_summary[
            "average_price"
        ]
    )

    category_summary[
        "opportunity_score"
    ] = weighted_score(
        category_summary,
        opportunity_weights
    )

    category_summary[
        "opportunity_level"
    ] = category_summary[
        "opportunity_score"
    ].apply(
        lambda value: assign_opportunity_level(
            value,
            thresholds
        )
    )

    category_summary[
        "opportunity_type"
    ] = "product"

    return (
        category_summary
        .sort_values(
            "opportunity_score",
            ascending=False
        )
        .reset_index(drop=True)
    )