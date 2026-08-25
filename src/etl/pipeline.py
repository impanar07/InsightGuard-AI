from datetime import datetime

from src.etl.config import (
    RAW_DATA_PATH,
    CLEAN_DATA_PATH,
    REPORT_PATH,
    DATASET_FILES,
    DATE_COLUMNS,
    KEY_DEFINITIONS,
)

from src.etl.extract import (
    extract_datasets,
)

from src.etl.transform import (
    apply_dataset_transformations,
)

from src.etl.validate import (
    calculate_dataset_profile,
    validate_key_integrity,
    validate_referential_integrity,
    validate_business_rules,
    create_missing_value_report,
)

from src.etl.load import (
    save_clean_datasets,
    save_report,
)


def run_etl_pipeline() -> None:
    """
    Execute the complete InsightGuard-AI
    Phase 4 ETL pipeline.
    """

    pipeline_start = datetime.now()

    print("=" * 80)
    print("INSIGHTGUARD-AI")
    print("PHASE 4 — ETL PIPELINE")
    print("=" * 80)

    # --------------------------------------------------
    # STEP 1 — EXTRACT
    # --------------------------------------------------

    print("\n[1/5] EXTRACTING RAW DATA")

    raw_datasets = extract_datasets(
        data_path=RAW_DATA_PATH,
        dataset_files=DATASET_FILES,
    )

    raw_profile = calculate_dataset_profile(
        raw_datasets
    )

    # --------------------------------------------------
    # STEP 2 — TRANSFORM
    # --------------------------------------------------

    print("\n[2/5] TRANSFORMING DATA")

    clean_datasets = (
        apply_dataset_transformations(
            datasets=raw_datasets,
            date_columns=DATE_COLUMNS,
        )
    )

    clean_profile = calculate_dataset_profile(
        clean_datasets
    )

    # --------------------------------------------------
    # STEP 3 — VALIDATE
    # --------------------------------------------------

    print("\n[3/5] VALIDATING DATA")

    key_validation = validate_key_integrity(
        datasets=clean_datasets,
        key_definitions=KEY_DEFINITIONS,
    )

    referential_integrity = (
        validate_referential_integrity(
            clean_datasets
        )
    )

    business_rule_validation = (
        validate_business_rules(
            clean_datasets
        )
    )

    missing_value_report = (
        create_missing_value_report(
            clean_datasets
        )
    )

    # --------------------------------------------------
    # STEP 4 — LOAD
    # --------------------------------------------------

    print("\n[4/5] SAVING CLEAN DATA")

    save_clean_datasets(
        datasets=clean_datasets,
        output_path=CLEAN_DATA_PATH,
    )

    # --------------------------------------------------
    # STEP 5 — REPORTING
    # --------------------------------------------------

    print("\n[5/5] GENERATING REPORTS")

    save_report(
        raw_profile,
        REPORT_PATH,
        "raw_dataset_profile.csv",
    )

    save_report(
        clean_profile,
        REPORT_PATH,
        "clean_dataset_profile.csv",
    )

    save_report(
        key_validation,
        REPORT_PATH,
        "key_validation_report.csv",
    )

    save_report(
        referential_integrity,
        REPORT_PATH,
        "referential_integrity_report.csv",
    )

    save_report(
        business_rule_validation,
        REPORT_PATH,
        "business_rule_validation.csv",
    )

    save_report(
        missing_value_report,
        REPORT_PATH,
        "missing_value_report.csv",
    )

    pipeline_end = datetime.now()

    duration = (
        pipeline_end - pipeline_start
    ).total_seconds()

    print("\n" + "=" * 80)
    print("ETL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print(
        f"Execution time: "
        f"{duration:.2f} seconds"
    )

    print(
        f"Clean data location: "
        f"{CLEAN_DATA_PATH}"
    )

    print(
        f"Report location: "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":

    run_etl_pipeline()