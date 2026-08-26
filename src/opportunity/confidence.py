from __future__ import annotations

import pandas as pd

from src.opportunity.scoring import (
    calculate_sample_size_score,
)


def calculate_data_quality_confidence(
    quality_score: float
) -> float:
    """
    Convert the overall DataTrust score into
    a confidence component between 0 and 100.
    """

    if pd.isna(quality_score):
        return 50.0

    return float(
        max(
            0,
            min(
                100,
                quality_score
            )
        )
    )


def calculate_opportunity_confidence(
    dataframe: pd.DataFrame,
    data_quality_score: float,
    sample_size_column: str,
    minimum_sample_size: int,
    confidence_weights: dict[str, float]
) -> pd.DataFrame:
    """
    Calculate confidence scores for opportunities.

    Confidence is based on:

    1. Overall data quality
    2. Sample size
    3. Metric completeness
    """

    result = dataframe.copy()

    # -----------------------------------------
    # 1. DATA QUALITY CONFIDENCE
    # -----------------------------------------

    data_quality_component = (
        calculate_data_quality_confidence(
            data_quality_score
        )
    )

    result[
        "data_quality_confidence"
    ] = data_quality_component

    # -----------------------------------------
    # 2. SAMPLE SIZE CONFIDENCE
    # -----------------------------------------

    if sample_size_column in result.columns:

        result[
            "sample_size_confidence"
        ] = calculate_sample_size_score(
            sample_size=result[
                sample_size_column
            ],
            minimum_sample_size=(
                minimum_sample_size
            )
        )

    else:

        result[
            "sample_size_confidence"
        ] = 50.0

    # -----------------------------------------
    # 3. METRIC COMPLETENESS
    # -----------------------------------------

    result[
        "metric_completeness"
    ] = (
        1
        - result.isna().mean(axis=1)
    ) * 100

    result[
        "metric_completeness"
    ] = (
        result[
            "metric_completeness"
        ]
        .clip(
            lower=0,
            upper=100
        )
    )

    # -----------------------------------------
    # 4. FINAL CONFIDENCE SCORE
    # -----------------------------------------

    result[
        "confidence_score"
    ] = (
        result[
            "data_quality_confidence"
        ]
        * confidence_weights[
            "data_quality"
        ]
        +
        result[
            "sample_size_confidence"
        ]
        * confidence_weights[
            "sample_size"
        ]
        +
        result[
            "metric_completeness"
        ]
        * confidence_weights[
            "metric_completeness"
        ]
    )

    result[
        "confidence_score"
    ] = (
        result[
            "confidence_score"
        ]
        .clip(
            lower=0,
            upper=100
        )
        .round(2)
    )

    return result