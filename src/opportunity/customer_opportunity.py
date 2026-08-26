from __future__ import annotations

import pandas as pd

from src.opportunity.scoring import (
    min_max_score,
    weighted_score,
    assign_opportunity_level,
)


def analyze_customer_opportunities(
    customer_features: pd.DataFrame,
    opportunity_weights: dict[str, float],
    thresholds: dict[str, float]
) -> pd.DataFrame:
    """
    Rank customers based on business opportunity.

    The score considers:

    - Customer lifetime value proxy
    - Purchase frequency
    - Recency
    - Average order value
    """

    required_columns = [
        "total_orders",
        "total_revenue",
        "average_order_value",
        "recency_days",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in customer_features.columns
    ]

    if missing_columns:

        raise KeyError(
            "Missing customer feature columns: "
            f"{missing_columns}"
        )

    opportunities = (
        customer_features.copy()
    )

    opportunities[
        "customer_value"
    ] = min_max_score(
        opportunities[
            "total_revenue"
        ]
    )

    opportunities[
        "purchase_frequency"
    ] = min_max_score(
        opportunities[
            "total_orders"
        ]
    )

    opportunities[
        "recency"
    ] = min_max_score(
        opportunities[
            "recency_days"
        ],
        higher_is_better=False
    )

    opportunities[
        "average_order_value"
    ] = min_max_score(
        opportunities[
            "average_order_value"
        ]
    )

    opportunities[
        "opportunity_score"
    ] = weighted_score(
        opportunities,
        opportunity_weights
    )

    opportunities[
        "opportunity_level"
    ] = opportunities[
        "opportunity_score"
    ].apply(
        lambda value: assign_opportunity_level(
            value,
            thresholds
        )
    )

    opportunities[
        "opportunity_type"
    ] = "customer"

    return (
        opportunities
        .sort_values(
            "opportunity_score",
            ascending=False
        )
        .reset_index(drop=True)
    )