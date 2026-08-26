from __future__ import annotations

import pandas as pd


def create_opportunity_identifier(
    dataframe: pd.DataFrame,
    identifier_column: str,
    name: str
) -> pd.DataFrame:
    """
    Convert a specific opportunity DataFrame into
    a common structure for unified ranking.
    """

    result = pd.DataFrame()

    # Unique opportunity ID
    result["opportunity_id"] = (
        name.lower()
        + "_"
        + dataframe[
            identifier_column
        ].astype(str)
    )

    # Opportunity name
    result["opportunity_name"] = (
        dataframe[
            identifier_column
        ].astype(str)
    )

    # Opportunity type
    result["opportunity_type"] = (
        dataframe[
            "opportunity_type"
        ]
    )

    # Main opportunity score
    result["opportunity_score"] = (
        dataframe[
            "opportunity_score"
        ]
    )

    # Opportunity level
    result["opportunity_level"] = (
        dataframe[
            "opportunity_level"
        ]
    )

    # Confidence score
    if "confidence_score" in dataframe.columns:

        result["confidence_score"] = (
            dataframe[
                "confidence_score"
            ]
        )

    else:

        result["confidence_score"] = 50.0

    return result


def combine_opportunities(
    product_opportunities: pd.DataFrame,
    customer_opportunities: pd.DataFrame,
    seller_opportunities: pd.DataFrame,
    delivery_opportunities: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine product, customer, seller and delivery
    opportunities into one unified ranking.

    Returns a DataFrame ranked by opportunity score.
    """

    opportunity_frames = []

    # ---------------------------------------------
    # PRODUCT OPPORTUNITIES
    # ---------------------------------------------

    if (
        product_opportunities is not None
        and not product_opportunities.empty
    ):

        product_result = (
            create_opportunity_identifier(
                dataframe=product_opportunities,
                identifier_column=(
                    "product_category_name"
                ),
                name="product"
            )
        )

        opportunity_frames.append(
            product_result
        )

    # ---------------------------------------------
    # CUSTOMER OPPORTUNITIES
    # ---------------------------------------------

    if (
        customer_opportunities is not None
        and not customer_opportunities.empty
    ):

        if (
            "customer_unique_id"
            in customer_opportunities.columns
        ):

            customer_identifier = (
                "customer_unique_id"
            )

        elif (
            "customer_id"
            in customer_opportunities.columns
        ):

            customer_identifier = (
                "customer_id"
            )

        else:

            raise KeyError(
                "Customer opportunity DataFrame "
                "must contain either "
                "'customer_unique_id' or "
                "'customer_id'."
            )

        customer_result = (
            create_opportunity_identifier(
                dataframe=customer_opportunities,
                identifier_column=(
                    customer_identifier
                ),
                name="customer"
            )
        )

        opportunity_frames.append(
            customer_result
        )

    # ---------------------------------------------
    # SELLER OPPORTUNITIES
    # ---------------------------------------------

    if (
        seller_opportunities is not None
        and not seller_opportunities.empty
    ):

        seller_result = (
            create_opportunity_identifier(
                dataframe=seller_opportunities,
                identifier_column=(
                    "seller_id"
                ),
                name="seller"
            )
        )

        opportunity_frames.append(
            seller_result
        )

    # ---------------------------------------------
    # DELIVERY OPPORTUNITIES
    # ---------------------------------------------

    if (
        delivery_opportunities is not None
        and not delivery_opportunities.empty
    ):

        delivery_result = (
            create_opportunity_identifier(
                dataframe=delivery_opportunities,
                identifier_column=(
                    "customer_state"
                ),
                name="delivery"
            )
        )

        opportunity_frames.append(
            delivery_result
        )

    # ---------------------------------------------
    # NO OPPORTUNITIES FOUND
    # ---------------------------------------------

    if not opportunity_frames:

        return pd.DataFrame(
            columns=[
                "rank",
                "opportunity_id",
                "opportunity_name",
                "opportunity_type",
                "opportunity_score",
                "opportunity_level",
                "confidence_score",
            ]
        )

    # ---------------------------------------------
    # COMBINE ALL OPPORTUNITIES
    # ---------------------------------------------

    combined = pd.concat(
        opportunity_frames,
        ignore_index=True
    )

    # Ensure numeric values
    combined[
        "opportunity_score"
    ] = pd.to_numeric(
        combined[
            "opportunity_score"
        ],
        errors="coerce"
    )

    combined[
        "confidence_score"
    ] = pd.to_numeric(
        combined[
            "confidence_score"
        ],
        errors="coerce"
    )

    # Sort by opportunity score first,
    # then confidence score
    combined = (
        combined
        .sort_values(
            by=[
                "opportunity_score",
                "confidence_score",
            ],
            ascending=[
                False,
                False,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # Create final rank
    combined.insert(
        0,
        "rank",
        range(
            1,
            len(combined) + 1
        )
    )

    return combined