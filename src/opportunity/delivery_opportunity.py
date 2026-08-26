from __future__ import annotations

import pandas as pd

from src.opportunity.scoring import (
    min_max_score,
    weighted_score,
    assign_opportunity_level,
)


def analyze_delivery_opportunities(
    order_fact: pd.DataFrame,
    opportunity_weights: dict[str, float],
    thresholds: dict[str, float]
) -> pd.DataFrame:
    """
    Identify delivery improvement opportunities
    by customer state.

    Opportunity is determined using:

    - Delay rate
    - Average delay duration
    - Order volume
    - Customer impact

    Higher delay and higher customer impact
    indicate a stronger improvement opportunity.
    """

    # --------------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------

    required_columns = [
        "order_id",
        "customer_state",
        "delivery_delay_days",
        "average_review_score",
        "order_status",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in order_fact.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing required delivery columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------
    # COPY DATA
    # --------------------------------------------------

    data = order_fact.copy()

    # Keep only valid customer states
    data = data[
        data["customer_state"].notna()
    ].copy()

    # Only delivered orders can be evaluated
    delivered_data = data[
        data["order_status"].eq("delivered")
    ].copy()

    if delivered_data.empty:
        raise ValueError(
            "No delivered orders available "
            "for delivery opportunity analysis."
        )

    # --------------------------------------------------
    # CLEAN DELIVERY DELAY
    # --------------------------------------------------

    delivered_data[
        "delivery_delay_days"
    ] = pd.to_numeric(
        delivered_data[
            "delivery_delay_days"
        ],
        errors="coerce"
    )

    delivered_data[
        "average_review_score"
    ] = pd.to_numeric(
        delivered_data[
            "average_review_score"
        ],
        errors="coerce"
    )

    # --------------------------------------------------
    # CREATE DELAY FLAG
    # --------------------------------------------------

    delivered_data["is_delayed"] = (
        delivered_data[
            "delivery_delay_days"
        ] > 0
    ).astype(int)

    # --------------------------------------------------
    # STATE-LEVEL AGGREGATION
    # --------------------------------------------------

    delivery_summary = (
        delivered_data
        .groupby(
            "customer_state",
            as_index=False
        )
        .agg(
            total_orders=(
                "order_id",
                "nunique"
            ),
            delayed_orders=(
                "is_delayed",
                "sum"
            ),
            average_delay_days=(
                "delivery_delay_days",
                "mean"
            ),
            average_review_score=(
                "average_review_score",
                "mean"
            ),
        )
    )

    # --------------------------------------------------
    # CALCULATE ACTUAL DELAY RATE
    # --------------------------------------------------

    delivery_summary[
        "delay_rate_percentage"
    ] = (
        delivery_summary[
            "delayed_orders"
        ]
        / delivery_summary[
            "total_orders"
        ]
        * 100
    )

    # --------------------------------------------------
    # CUSTOMER IMPACT
    #
    # Lower review score means higher impact.
    # --------------------------------------------------

    delivery_summary[
        "customer_impact_score"
    ] = (
        (
            5
            - delivery_summary[
                "average_review_score"
            ]
        )
        / 5
        * 100
    )

    delivery_summary[
        "customer_impact_score"
    ] = (
        delivery_summary[
            "customer_impact_score"
        ]
        .fillna(50)
        .clip(
            lower=0,
            upper=100
        )
    )

    # --------------------------------------------------
    # CREATE NORMALIZED SCORING METRICS
    # --------------------------------------------------

    # Higher delay rate = greater opportunity
    delivery_summary[
        "delay_rate"
    ] = min_max_score(
        delivery_summary[
            "delay_rate_percentage"
        ],
        higher_is_better=True
    )

    # Higher average delay = greater opportunity
    delivery_summary[
        "delay_duration"
    ] = min_max_score(
        delivery_summary[
            "average_delay_days"
        ],
        higher_is_better=True
    )

    # Higher order volume = greater business impact
    delivery_summary[
        "order_volume"
    ] = min_max_score(
        delivery_summary[
            "total_orders"
        ],
        higher_is_better=True
    )

    # Higher customer dissatisfaction = greater opportunity
    delivery_summary[
        "customer_impact"
    ] = delivery_summary[
        "customer_impact_score"
    ]

    # --------------------------------------------------
    # VALIDATE SCORING COLUMNS
    # --------------------------------------------------

    scoring_columns = [
        "delay_rate",
        "delay_duration",
        "order_volume",
        "customer_impact",
    ]

    for column in scoring_columns:

        if column not in delivery_summary.columns:
            raise KeyError(
                f"Scoring column '{column}' "
                "was not created."
            )

        delivery_summary[column] = (
            pd.to_numeric(
                delivery_summary[column],
                errors="coerce"
            )
            .fillna(0)
        )

    # --------------------------------------------------
    # CALCULATE OPPORTUNITY SCORE
    # --------------------------------------------------

    delivery_summary[
        "opportunity_score"
    ] = weighted_score(
        dataframe=delivery_summary,
        score_weights=opportunity_weights
    )

    # --------------------------------------------------
    # ASSIGN OPPORTUNITY LEVEL
    # --------------------------------------------------

    delivery_summary[
        "opportunity_level"
    ] = (
        delivery_summary[
            "opportunity_score"
        ]
        .apply(
            lambda value: (
                assign_opportunity_level(
                    score=value,
                    thresholds=thresholds
                )
            )
        )
    )

    # --------------------------------------------------
    # ADD OPPORTUNITY TYPE
    # --------------------------------------------------

    delivery_summary[
        "opportunity_type"
    ] = "delivery"

    # --------------------------------------------------
    # SORT RESULTS
    # --------------------------------------------------

    delivery_summary = (
        delivery_summary
        .sort_values(
            by="opportunity_score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return delivery_summary