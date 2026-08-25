from __future__ import annotations

from typing import Dict

import pandas as pd


def validate_weights(
    quality_weights: Dict[str, float]
) -> None:
    """
    Ensure all weights sum to 1.
    """

    total_weight = sum(
        quality_weights.values()
    )

    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError(
            "Quality weights must sum to 1. "
            f"Current total: {total_weight}"
        )


def build_quality_scorecard(
    completeness: pd.DataFrame,
    uniqueness: pd.DataFrame,
    validity: pd.DataFrame,
    consistency: pd.DataFrame,
    integrity: pd.DataFrame,
    timeliness: pd.DataFrame,
    quality_weights: Dict[str, float]
) -> pd.DataFrame:
    """
    Combine all quality dimensions into one
    weighted Data Quality Score.
    """

    validate_weights(quality_weights)

    scorecard = completeness.copy()

    scorecard = scorecard.merge(
        uniqueness,
        on="dataset",
        how="outer"
    )

    scorecard = scorecard.merge(
        validity,
        on="dataset",
        how="outer"
    )

    scorecard = scorecard.merge(
        consistency,
        on="dataset",
        how="outer"
    )

    scorecard = scorecard.merge(
        integrity,
        on="dataset",
        how="outer"
    )

    scorecard = scorecard.merge(
        timeliness,
        on="dataset",
        how="outer"
    )

    score_columns = [
        "completeness_score",
        "uniqueness_score",
        "validity_score",
        "consistency_score",
        "referential_integrity_score",
        "timeliness_score",
    ]

    for column in score_columns:

        if column not in scorecard.columns:

            scorecard[column] = 100.0

        scorecard[column] = (
            scorecard[column]
            .fillna(100.0)
        )

    scorecard["data_quality_score"] = (
        scorecard["completeness_score"]
        * quality_weights["completeness"]

        + scorecard["uniqueness_score"]
        * quality_weights["uniqueness"]

        + scorecard["validity_score"]
        * quality_weights["validity"]

        + scorecard["consistency_score"]
        * quality_weights["consistency"]

        + scorecard[
            "referential_integrity_score"
        ]
        * quality_weights[
            "referential_integrity"
        ]

        + scorecard["timeliness_score"]
        * quality_weights["timeliness"]
    )

    scorecard["data_quality_score"] = (
        scorecard[
            "data_quality_score"
        ].round(4)
    )

    return scorecard.sort_values(
        "data_quality_score",
        ascending=True
    ).reset_index(drop=True)


def calculate_overall_quality_score(
    scorecard: pd.DataFrame
) -> float:
    """
    Calculate overall project quality score.
    """

    if scorecard.empty:
        return 100.0

    return round(
        scorecard[
            "data_quality_score"
        ].mean(),
        4
    )