from pathlib import Path


def find_project_root(start_path: Path | None = None) -> Path:
    """
    Locate the project root by searching for the
    InsightGuard-AI directory structure.
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
        "Project root could not be located. "
        "Expected directories: data/ and src/."
    )


PROJECT_ROOT = find_project_root()

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "clean"

REPORT_PATH = PROJECT_ROOT / "reports" / "etl"

RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
CLEAN_DATA_PATH.mkdir(parents=True, exist_ok=True)
REPORT_PATH.mkdir(parents=True, exist_ok=True)


DATASET_FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
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