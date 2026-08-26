from __future__ import annotations

from pathlib import Path


def find_project_root(
    start_path: Path | None = None
) -> Path:
    """
    Locate the project root by searching for
    the data and src directories.
    """

    if start_path is None:
        start_path = Path.cwd()

    start_path = start_path.resolve()

    for path in [start_path, *start_path.parents]:

        if (
            (path / "data").exists()
            and (path / "src").exists()
        ):
            return path

    raise FileNotFoundError(
        "InsightGuard-AI project root could not "
        "be located."
    )


PROJECT_ROOT = find_project_root()

CLEAN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "clean"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "opportunity"
)

REPORT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


CLEAN_DATASET_FILES = {
    "customers": "customers_clean.csv",
    "geolocation": "geolocation_clean.csv",
    "order_items": "order_items_clean.csv",
    "order_payments": "order_payments_clean.csv",
    "order_reviews": "order_reviews_clean.csv",
    "orders": "orders_clean.csv",
    "products": "products_clean.csv",
    "sellers": "sellers_clean.csv",
    "category_translation": (
        "category_translation_clean.csv"
    ),
}


PRODUCT_OPPORTUNITY_WEIGHTS = {
    "revenue_potential": 0.30,
    "order_demand": 0.25,
    "growth_potential": 0.25,
    "customer_satisfaction": 0.10,
    "profit_proxy": 0.10,
}


CUSTOMER_OPPORTUNITY_WEIGHTS = {
    "customer_value": 0.40,
    "purchase_frequency": 0.25,
    "recency": 0.20,
    "average_order_value": 0.15,
}


SELLER_OPPORTUNITY_WEIGHTS = {
    "revenue_potential": 0.30,
    "order_volume": 0.25,
    "customer_satisfaction": 0.20,
    "delivery_performance": 0.15,
    "growth_potential": 0.10,
}


DELIVERY_OPPORTUNITY_WEIGHTS = {
    "delay_rate": 0.40,
    "delay_duration": 0.25,
    "order_volume": 0.20,
    "customer_impact": 0.15,
}


OPPORTUNITY_THRESHOLDS = {
    "high": 75,
    "medium": 50,
    "low": 0,
}


CONFIDENCE_WEIGHTS = {
    "data_quality": 0.50,
    "sample_size": 0.30,
    "metric_completeness": 0.20,
}


MINIMUM_SAMPLE_SIZE = 30


OPPORTUNITY_TYPES = {
    "product": "Product / Category Opportunity",
    "customer": "Customer Opportunity",
    "seller": "Seller Opportunity",
    "delivery": "Delivery Improvement Opportunity",
}