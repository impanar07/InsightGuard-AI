from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.opportunity.config import (
    CLEAN_DATA_PATH,
    REPORT_PATH,
    CLEAN_DATASET_FILES,
    PRODUCT_OPPORTUNITY_WEIGHTS,
    CUSTOMER_OPPORTUNITY_WEIGHTS,
    SELLER_OPPORTUNITY_WEIGHTS,
    DELIVERY_OPPORTUNITY_WEIGHTS,
    OPPORTUNITY_THRESHOLDS,
    CONFIDENCE_WEIGHTS,
    MINIMUM_SAMPLE_SIZE,
)

from src.opportunity.loader import (
    load_datasets,
    convert_datetime_columns,
)

from src.opportunity.feature_engineering import (
    build_all_features,
)

from src.opportunity.product_opportunity import (
    analyze_product_opportunities,
)

from src.opportunity.customer_opportunity import (
    analyze_customer_opportunities,
)

from src.opportunity.seller_opportunity import (
    analyze_seller_opportunities,
)

from src.opportunity.delivery_opportunity import (
    analyze_delivery_opportunities,
)

from src.opportunity.confidence import (
    calculate_opportunity_confidence,
)

from src.opportunity.ranking import (
    combine_opportunities,
)


def load_overall_data_quality_score() -> float:
    """
    Load the Phase 5 overall DataTrust score.

    A fallback value of 50 is used if the
    report does not exist.
    """

    quality_file = (
        Path(
            REPORT_PATH.parent
        )
        / "data_quality"
        / "overall_quality_score.csv"
    )

    if not quality_file.exists():

        print(
            "Warning: DataTrust score not found. "
            "Using default confidence of 50."
        )

        return 50.0

    quality_dataframe = pd.read_csv(
        quality_file
    )

    required_column = (
        "overall_data_quality_score"
    )

    if (
        required_column
        not in quality_dataframe.columns
    ):

        return 50.0

    return float(
        quality_dataframe[
            required_column
        ].iloc[0]
    )


def save_report(
    dataframe: pd.DataFrame,
    filename: str
) -> None:
    """
    Save an Opportunity AI report.
    """

    REPORT_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        REPORT_PATH / filename
    )

    dataframe.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path.name}"
    )


def build_executive_summary(
    product_opportunities: pd.DataFrame,
    customer_opportunities: pd.DataFrame,
    seller_opportunities: pd.DataFrame,
    delivery_opportunities: pd.DataFrame,
    overall_data_quality_score: float
) -> pd.DataFrame:
    """
    Build a high-level summary for dashboards
    and later AI explanation.
    """

    records = []

    opportunity_groups = {
        "Product": product_opportunities,
        "Customer": customer_opportunities,
        "Seller": seller_opportunities,
        "Delivery": delivery_opportunities,
    }

    for opportunity_name, dataframe in (
        opportunity_groups.items()
    ):

        if dataframe.empty:
            continue

        top_row = dataframe.iloc[0]

        records.append({
            "opportunity_category":
                opportunity_name,

            "top_opportunity_score":
                round(
                    float(
                        top_row[
                            "opportunity_score"
                        ]
                    ),
                    2
                ),

            "top_opportunity":
                str(
                    top_row[
                        "opportunity_type"
                    ]
                ),

            "high_opportunity_count":
                int(
                    dataframe[
                        "opportunity_level"
                    ]
                    .eq("High")
                    .sum()
                ),

            "total_opportunities":
                int(
                    len(dataframe)
                ),

            "data_quality_score":
                round(
                    overall_data_quality_score,
                    2
                ),
        })

    return pd.DataFrame(
        records
    )


def run_opportunity_engine() -> None:
    """
    Run the complete Opportunity AI engine.
    """

    print("=" * 80)
    print("INSIGHTGUARD AI")
    print("PHASE 6 — OPPORTUNITY AI ENGINE")
    print("=" * 80)

    # --------------------------------------------------
    # STEP 1 — LOAD DATA
    # --------------------------------------------------

    print(
        "\n[1/6] LOADING CLEAN DATASETS"
    )

    datasets = load_datasets(
        data_path=CLEAN_DATA_PATH,
        dataset_files=CLEAN_DATASET_FILES
    )

    datasets = convert_datetime_columns(
        datasets
    )

    # --------------------------------------------------
    # STEP 2 — FEATURE ENGINEERING
    # --------------------------------------------------

    print(
        "\n[2/6] BUILDING ANALYTICAL FEATURES"
    )

    features = build_all_features(
        datasets
    )

    order_item_fact = features[
        "order_item_fact"
    ]

    order_fact = features[
        "order_fact"
    ]

    customer_features = features[
        "customer_features"
    ]

    print(
        f"Order item fact: "
        f"{len(order_item_fact):,} rows"
    )

    print(
        f"Order fact: "
        f"{len(order_fact):,} rows"
    )

    print(
        f"Customer features: "
        f"{len(customer_features):,} rows"
    )

    # --------------------------------------------------
    # STEP 3 — LOAD DATATRUST SCORE
    # --------------------------------------------------

    print(
        "\n[3/6] LOADING DATATRUST SCORE"
    )

    overall_data_quality_score = (
        load_overall_data_quality_score()
    )

    print(
        f"DataTrust Score: "
        f"{overall_data_quality_score:.2f}/100"
    )

    # --------------------------------------------------
    # STEP 4 — ANALYZE OPPORTUNITIES
    # --------------------------------------------------

    print(
        "\n[4/6] ANALYZING BUSINESS OPPORTUNITIES"
    )

    product_opportunities = (
        analyze_product_opportunities(
            order_item_fact=order_item_fact,
            opportunity_weights=(
                PRODUCT_OPPORTUNITY_WEIGHTS
            ),
            thresholds=OPPORTUNITY_THRESHOLDS
        )
    )

    customer_opportunities = (
        analyze_customer_opportunities(
            customer_features=customer_features,
            opportunity_weights=(
                CUSTOMER_OPPORTUNITY_WEIGHTS
            ),
            thresholds=OPPORTUNITY_THRESHOLDS
        )
    )

    seller_opportunities = (
        analyze_seller_opportunities(
            order_item_fact=order_item_fact,
            opportunity_weights=(
                SELLER_OPPORTUNITY_WEIGHTS
            ),
            thresholds=OPPORTUNITY_THRESHOLDS
        )
    )

    delivery_opportunities = (
        analyze_delivery_opportunities(
            order_fact=order_fact,
            opportunity_weights=(
                DELIVERY_OPPORTUNITY_WEIGHTS
            ),
            thresholds=OPPORTUNITY_THRESHOLDS
        )
    )

    # --------------------------------------------------
    # STEP 5 — CALCULATE CONFIDENCE
    # --------------------------------------------------

    print(
        "\n[5/6] CALCULATING OPPORTUNITY "
        "CONFIDENCE"
    )

    product_opportunities = (
        calculate_opportunity_confidence(
            dataframe=product_opportunities,
            data_quality_score=(
                overall_data_quality_score
            ),
            sample_size_column="total_orders",
            minimum_sample_size=(
                MINIMUM_SAMPLE_SIZE
            ),
            confidence_weights=(
                CONFIDENCE_WEIGHTS
            )
        )
    )

    customer_opportunities = (
        calculate_opportunity_confidence(
            dataframe=customer_opportunities,
            data_quality_score=(
                overall_data_quality_score
            ),
            sample_size_column="total_orders",
            minimum_sample_size=(
                MINIMUM_SAMPLE_SIZE
            ),
            confidence_weights=(
                CONFIDENCE_WEIGHTS
            )
        )
    )

    seller_opportunities = (
        calculate_opportunity_confidence(
            dataframe=seller_opportunities,
            data_quality_score=(
                overall_data_quality_score
            ),
            sample_size_column="total_orders",
            minimum_sample_size=(
                MINIMUM_SAMPLE_SIZE
            ),
            confidence_weights=(
                CONFIDENCE_WEIGHTS
            )
        )
    )

    delivery_opportunities = (
        calculate_opportunity_confidence(
            dataframe=delivery_opportunities,
            data_quality_score=(
                overall_data_quality_score
            ),
            sample_size_column="total_orders",
            minimum_sample_size=(
                MINIMUM_SAMPLE_SIZE
            ),
            confidence_weights=(
                CONFIDENCE_WEIGHTS
            )
        )
    )

    # --------------------------------------------------
    # STEP 6 — RANK AND SAVE
    # --------------------------------------------------

    print(
        "\n[6/6] RANKING AND SAVING RESULTS"
    )

    all_opportunities = (
        combine_opportunities(
            product_opportunities=(
                product_opportunities
            ),
            customer_opportunities=(
                customer_opportunities
            ),
            seller_opportunities=(
                seller_opportunities
            ),
            delivery_opportunities=(
                delivery_opportunities
            )
        )
    )

    executive_summary = (
        build_executive_summary(
            product_opportunities=(
                product_opportunities
            ),
            customer_opportunities=(
                customer_opportunities
            ),
            seller_opportunities=(
                seller_opportunities
            ),
            delivery_opportunities=(
                delivery_opportunities
            ),
            overall_data_quality_score=(
                overall_data_quality_score
            )
        )
    )

    save_report(
        product_opportunities,
        "product_opportunities.csv"
    )

    save_report(
        customer_opportunities,
        "customer_opportunities.csv"
    )

    save_report(
        seller_opportunities,
        "seller_opportunities.csv"
    )

    save_report(
        delivery_opportunities,
        "delivery_opportunities.csv"
    )

    save_report(
        all_opportunities,
        "all_opportunities_ranked.csv"
    )

    save_report(
        executive_summary,
        "opportunity_executive_summary.csv"
    )

    save_report(
        order_item_fact,
        "order_item_fact.csv"
    )

    save_report(
        order_fact,
        "order_fact.csv"
    )

    save_report(
        customer_features,
        "customer_features.csv"
    )

    print("\n" + "=" * 80)
    print("OPPORTUNITY AI ENGINE COMPLETED")
    print("=" * 80)

    print(
        f"\nDataTrust Score Used: "
        f"{overall_data_quality_score:.2f}/100"
    )

    print(
        "\nTop 10 Opportunities:"
    )

    print(
        all_opportunities
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        f"\nReports saved to:\n"
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    run_opportunity_engine()