from __future__ import annotations

import pandas as pd

from src.opportunity.scoring import (
    min_max_score,
    weighted_score,
    assign_opportunity_level,
)


def analyze_seller_opportunities(
    order_item_fact: pd.DataFrame,
    opportunity_weights: dict[str, float],
    thresholds: dict[str, float]
) -> pd.DataFrame:
    """
    Identify seller-level business opportunities.
    """

    required_columns = [
        "seller_id",
        "order_id",
        "item_revenue",
        "order_purchase_timestamp",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in order_item_fact.columns
    ]

    if missing_columns:

        raise KeyError(
            "Missing required seller columns: "
            f"{missing_columns}"
        )

    data = order_item_fact.copy()

    seller_summary = (
        data
        .groupby(
            "seller_id",
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
            total_items=(
                "order_item_id",
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
                "seller_id",
                as_index=False
            )
            .agg(
                average_review_score=(
                    "average_review_score",
                    "mean"
                )
            )
        )

        seller_summary = (
            seller_summary.merge(
                review_summary,
                on="seller_id",
                how="left"
            )
        )

    else:

        seller_summary[
            "average_review_score"
        ] = 3.0

    if (
        "order_delivered_customer_date"
        in data.columns
        and "order_estimated_delivery_date"
        in data.columns
    ):

        delivery_data = (
            data
            .dropna(
                subset=[
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                ]
            )
            .copy()
        )

        delivery_data[
            "delivery_delay_days"
        ] = (
            delivery_data[
                "order_delivered_customer_date"
            ]
            - delivery_data[
                "order_estimated_delivery_date"
            ]
        ).dt.total_seconds() / 86400

        delivery_summary = (
            delivery_data
            .groupby(
                "seller_id",
                as_index=False
            )
            .agg(
                average_delivery_delay=(
                    "delivery_delay_days",
                    "mean"
                )
            )
        )

        seller_summary = (
            seller_summary.merge(
                delivery_summary,
                on="seller_id",
                how="left"
            )
        )

    else:

        seller_summary[
            "average_delivery_delay"
        ] = 0.0

    seller_summary[
        "average_delivery_delay"
    ] = (
        seller_summary[
            "average_delivery_delay"
        ]
        .fillna(0)
    )

    data["purchase_month"] = (
        data[
            "order_purchase_timestamp"
        ]
        .dt.to_period("M")
    )

    monthly_seller_orders = (
        data
        .groupby(
            [
                "seller_id",
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

    for seller_id, group in (
        monthly_seller_orders.groupby(
            "seller_id"
        )
    ):

        group = group.sort_values(
            "purchase_month"
        )

        if len(group) < 2:

            growth_rate = 0.0

        else:

            first_orders = (
                group[
                    "monthly_orders"
                ].iloc[0]
            )

            last_orders = (
                group[
                    "monthly_orders"
                ].iloc[-1]
            )

            growth_rate = (
                (
                    last_orders
                    - first_orders
                )
                / first_orders
                * 100
                if first_orders > 0
                else 0.0
            )

        growth_records.append({
            "seller_id": seller_id,
            "growth_rate": growth_rate,
        })

    growth_dataframe = pd.DataFrame(
        growth_records
    )

    seller_summary = (
        seller_summary.merge(
            growth_dataframe,
            on="seller_id",
            how="left"
        )
    )

    seller_summary[
        "revenue_potential"
    ] = min_max_score(
        seller_summary[
            "total_revenue"
        ]
    )

    seller_summary[
        "order_volume"
    ] = min_max_score(
        seller_summary[
            "total_orders"
        ]
    )

    seller_summary[
        "customer_satisfaction"
    ] = (
        seller_summary[
            "average_review_score"
        ]
        / 5
        * 100
    ).fillna(50)

    seller_summary[
        "delivery_performance"
    ] = min_max_score(
        seller_summary[
            "average_delivery_delay"
        ],
        higher_is_better=False
    )

    seller_summary[
        "growth_potential"
    ] = min_max_score(
        seller_summary[
            "growth_rate"
        ]
    )

    seller_summary[
        "opportunity_score"
    ] = weighted_score(
        seller_summary,
        opportunity_weights
    )

    seller_summary[
        "opportunity_level"
    ] = seller_summary[
        "opportunity_score"
    ].apply(
        lambda value: assign_opportunity_level(
            value,
            thresholds
        )
    )

    seller_summary[
        "opportunity_type"
    ] = "seller"

    return (
        seller_summary
        .sort_values(
            "opportunity_score",
            ascending=False
        )
        .reset_index(drop=True)
    )