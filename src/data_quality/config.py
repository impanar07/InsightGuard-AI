from __future__ import annotations

from pathlib import Path
from typing import Any


def find_project_root(start_path: Path | None = None) -> Path:
    """
    Locate the InsightGuard-AI project root.
    """

    if start_path is None:
        start_path = Path.cwd()

    start_path = start_path.resolve()

    for path in [start_path, *start_path.parents]:
        if (path / "data").exists() and (path / "src").exists():
            return path

    raise FileNotFoundError(
        "Project root could not be located. "
        "Expected project directories: data/ and src/."
    )


PROJECT_ROOT = find_project_root()

CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "clean"
REPORT_PATH = PROJECT_ROOT / "reports" / "data_quality"

CLEAN_DATA_PATH.mkdir(parents=True, exist_ok=True)
REPORT_PATH.mkdir(parents=True, exist_ok=True)


CLEAN_DATASET_FILES = {
    "customers": "customers_clean.csv",
    "geolocation": "geolocation_clean.csv",
    "order_items": "order_items_clean.csv",
    "order_payments": "order_payments_clean.csv",
    "order_reviews": "order_reviews_clean.csv",
    "orders": "orders_clean.csv",
    "products": "products_clean.csv",
    "sellers": "sellers_clean.csv",
    "category_translation": "category_translation_clean.csv",
}


QUALITY_WEIGHTS = {
    "completeness": 0.25,
    "uniqueness": 0.15,
    "validity": 0.20,
    "consistency": 0.15,
    "referential_integrity": 0.15,
    "timeliness": 0.10,
}


KEY_DEFINITIONS = {
    "customers": ["customer_id"],
    "orders": ["order_id"],
    "products": ["product_id"],
    "sellers": ["seller_id"],
    "order_items": [
        "order_id",
        "order_item_id",
    ],
    "order_payments": [
        "order_id",
        "payment_sequential",
    ],
}


REFERENTIAL_RULES = [
    {
        "name": "orders_customer_integrity",
        "child_dataset": "orders",
        "child_column": "customer_id",
        "parent_dataset": "customers",
        "parent_column": "customer_id",
    },
    {
        "name": "order_items_order_integrity",
        "child_dataset": "order_items",
        "child_column": "order_id",
        "parent_dataset": "orders",
        "parent_column": "order_id",
    },
    {
        "name": "order_items_product_integrity",
        "child_dataset": "order_items",
        "child_column": "product_id",
        "parent_dataset": "products",
        "parent_column": "product_id",
    },
    {
        "name": "order_items_seller_integrity",
        "child_dataset": "order_items",
        "child_column": "seller_id",
        "parent_dataset": "sellers",
        "parent_column": "seller_id",
    },
    {
        "name": "order_payments_order_integrity",
        "child_dataset": "order_payments",
        "child_column": "order_id",
        "parent_dataset": "orders",
        "parent_column": "order_id",
    },
    {
        "name": "order_reviews_order_integrity",
        "child_dataset": "order_reviews",
        "child_column": "order_id",
        "parent_dataset": "orders",
        "parent_column": "order_id",
    },
]


VALIDITY_RULES: dict[str, list[dict[str, Any]]] = {
    "orders": [
        {
            "column": "order_status",
            "rule_name": "valid_order_status",
            "rule_type": "allowed_values",
            "allowed_values": [
                "created",
                "approved",
                "invoiced",
                "processing",
                "shipped",
                "delivered",
                "unavailable",
                "canceled",
            ],
        },
    ],
    "order_items": [
        {
            "column": "price",
            "rule_name": "non_negative_price",
            "rule_type": "minimum",
            "minimum": 0,
        },
        {
            "column": "freight_value",
            "rule_name": "non_negative_freight",
            "rule_type": "minimum",
            "minimum": 0,
        },
        {
            "column": "order_item_id",
            "rule_name": "positive_order_item_id",
            "rule_type": "minimum",
            "minimum": 1,
        },
    ],
    "order_payments": [
        {
            "column": "payment_value",
            "rule_name": "non_negative_payment",
            "rule_type": "minimum",
            "minimum": 0,
        },
        {
            "column": "payment_installments",
            "rule_name": "non_negative_installments",
            "rule_type": "minimum",
            "minimum": 0,
        },
        {
            "column": "payment_sequential",
            "rule_name": "positive_payment_sequence",
            "rule_type": "minimum",
            "minimum": 1,
        },
    ],
    "order_reviews": [
        {
            "column": "review_score",
            "rule_name": "valid_review_score",
            "rule_type": "range",
            "minimum": 1,
            "maximum": 5,
        },
    ],
    "products": [
        {
            "column": "product_weight_g",
            "rule_name": "non_negative_product_weight",
            "rule_type": "minimum",
            "minimum": 0,
        },
        {
            "column": "product_length_cm",
            "rule_name": "non_negative_product_length",
            "rule_type": "minimum",
            "minimum": 0,
        },
        {
            "column": "product_height_cm",
            "rule_name": "non_negative_product_height",
            "rule_type": "minimum",
            "minimum": 0,
        },
        {
            "column": "product_width_cm",
            "rule_name": "non_negative_product_width",
            "rule_type": "minimum",
            "minimum": 0,
        },
    ],
}


DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": [
        "shipping_limit_date",
    ],
    "order_reviews": [
        "review_creation_date",
        "review_answer_timestamp",
    ],
}


CONSISTENCY_RULES = [
    {
        "dataset": "orders",
        "rule_name": "approval_after_purchase",
        "left_column": "order_approved_at",
        "right_column": "order_purchase_timestamp",
        "operator": "greater_than_or_equal",
    },
    {
        "dataset": "orders",
        "rule_name": "carrier_after_purchase",
        "left_column": "order_delivered_carrier_date",
        "right_column": "order_purchase_timestamp",
        "operator": "greater_than_or_equal",
    },
    {
        "dataset": "orders",
        "rule_name": "customer_delivery_after_purchase",
        "left_column": "order_delivered_customer_date",
        "right_column": "order_purchase_timestamp",
        "operator": "greater_than_or_equal",
    },
    {
        "dataset": "orders",
        "rule_name": "estimated_delivery_after_purchase",
        "left_column": "order_estimated_delivery_date",
        "right_column": "order_purchase_timestamp",
        "operator": "greater_than_or_equal",
    },
    {
        "dataset": "order_reviews",
        "rule_name": "review_answer_after_creation",
        "left_column": "review_answer_timestamp",
        "right_column": "review_creation_date",
        "operator": "greater_than_or_equal",
    },
]


TIMELINESS_RULES = [
    {
        "dataset": "orders",
        "date_column": "order_purchase_timestamp",
        "reference_dataset": "orders",
        "reference_column": "order_purchase_timestamp",
    },
]


SEVERITY_THRESHOLDS = {
    "critical": 0.20,
    "high": 0.10,
    "medium": 0.03,
    "low": 0.00,
}