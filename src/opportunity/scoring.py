from __future__ import annotations

import numpy as np
import pandas as pd


def min_max_score(
    series: pd.Series,
    higher_is_better: bool = True
) -> pd.Series:
    """
    Normalize values into a 0-100 score.

    Handles missing values and constant series safely.
    """

    numeric_series = pd.to_numeric(
        series,
        errors="coerce"
    )

    minimum = numeric_series.min()
    maximum = numeric_series.max()

    if pd.isna(minimum) or pd.isna(maximum):

        return pd.Series(
            50.0,
            index=series.index
        )

    if maximum == minimum:

        return pd.Series(
            100.0,
            index=series.index
        )

    normalized = (
        (
            numeric_series - minimum
        )
        / (
            maximum - minimum
        )
    ) * 100

    normalized = normalized.fillna(0)

    if not higher_is_better:

        normalized = 100 - normalized

    return normalized.clip(
        lower=0,
        upper=100
    )


def weighted_score(
    dataframe: pd.DataFrame,
    score_weights: dict[str, float]
) -> pd.Series:
    """
    Calculate weighted opportunity score.
    """

    total_weight = sum(
        score_weights.values()
    )

    if not np.isclose(
        total_weight,
        1.0
    ):
        raise ValueError(
            "Score weights must sum to 1."
        )

    result = pd.Series(
        0.0,
        index=dataframe.index
    )

    for column, weight in (
        score_weights.items()
    ):

        if column not in dataframe.columns:

            raise KeyError(
                f"Missing scoring column: "
                f"{column}"
            )

        result += (
            dataframe[column]
            * weight
        )

    return result.clip(
        lower=0,
        upper=100
    )


def assign_opportunity_level(
    score: float,
    thresholds: dict[str, float]
) -> str:
    """
    Assign High, Medium, or Low opportunity.
    """

    if score >= thresholds["high"]:
        return "High"

    if score >= thresholds["medium"]:
        return "Medium"

    return "Low"


def calculate_sample_size_score(
    sample_size: pd.Series,
    minimum_sample_size: int
) -> pd.Series:
    """
    Calculate confidence contribution from sample size.
    """

    score = (
        sample_size
        / minimum_sample_size
        * 100
    )

    return score.clip(
        lower=0,
        upper=100
    )