from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.data_quality.config import (
    CLEAN_DATA_PATH,
    REPORT_PATH,
    CLEAN_DATASET_FILES,
    QUALITY_WEIGHTS,
    KEY_DEFINITIONS,
    REFERENTIAL_RULES,
    VALIDITY_RULES,
    CONSISTENCY_RULES,
    DATE_COLUMNS,
    SEVERITY_THRESHOLDS,
)

from src.data_quality.profiler import (
    profile_all_datasets,
)

from src.data_quality.completeness import (
    calculate_all_completeness,
)

from src.data_quality.uniqueness import (
    calculate_all_uniqueness,
)

from src.data_quality.validity import (
    calculate_all_validity,
)

from src.data_quality.consistency import (
    calculate_all_consistency,
)

from src.data_quality.integrity import (
    calculate_referential_integrity,
)

from src.data_quality.timeliness import (
    calculate_all_timeliness,
)

from src.data_quality.scoring import (
    build_quality_scorecard,
    calculate_overall_quality_score,
)

from src.data_quality.issue_detector import (
    detect_completeness_issues,
    detect_validity_issues,
    detect_consistency_issues,
    detect_integrity_issues,
    combine_detected_issues,
)


def load_clean_datasets(
    data_path: Path,
    dataset_files: Dict[str, str]
) -> Dict[str, pd.DataFrame]:
    """
    Load all cleaned datasets.
    """

    datasets = {}

    for dataset_name, filename in (
        dataset_files.items()
    ):

        file_path = data_path / filename

        if not file_path.exists():

            raise FileNotFoundError(
                f"Clean dataset not found: "
                f"{file_path}"
            )

        print(
            f"Loading {dataset_name}: "
            f"{filename}"
        )

        datasets[dataset_name] = pd.read_csv(
            file_path,
            low_memory=False
        )

    return datasets


def save_dataframe(
    dataframe: pd.DataFrame,
    filename: str
) -> None:
    """
    Save a DataFrame to the data quality
    reports directory.
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
        f"Saved report: "
        f"{output_path.name}"
    )


def run_data_quality_engine() -> None:
    """
    Execute the complete DataTrust AI engine.
    """

    print("=" * 80)
    print("INSIGHTGUARD-AI")
    print("PHASE 5 — DATATRUST AI ENGINE")
    print("=" * 80)

    # --------------------------------------------------
    # STEP 1 — LOAD
    # --------------------------------------------------

    print("\n[1/8] LOADING CLEAN DATASETS")

    datasets = load_clean_datasets(
        data_path=CLEAN_DATA_PATH,
        dataset_files=CLEAN_DATASET_FILES
    )

    # --------------------------------------------------
    # STEP 2 — PROFILE
    # --------------------------------------------------

    print("\n[2/8] PROFILING DATASETS")

    profile_report = profile_all_datasets(
        datasets
    )

    # --------------------------------------------------
    # STEP 3 — COMPLETENESS
    # --------------------------------------------------

    print("\n[3/8] CALCULATING COMPLETENESS")

    (
        completeness_scores,
        completeness_report,
    ) = calculate_all_completeness(
        datasets
    )

    # --------------------------------------------------
    # STEP 4 — UNIQUENESS
    # --------------------------------------------------

    print("\n[4/8] CALCULATING UNIQUENESS")

    (
        uniqueness_scores,
        uniqueness_report,
    ) = calculate_all_uniqueness(
        datasets=datasets,
        key_definitions=KEY_DEFINITIONS
    )

    # --------------------------------------------------
    # STEP 5 — VALIDITY
    # --------------------------------------------------

    print("\n[5/8] VALIDATING BUSINESS RULES")

    (
        validity_scores,
        validity_report,
    ) = calculate_all_validity(
        datasets=datasets,
        validity_rules=VALIDITY_RULES
    )

    # --------------------------------------------------
    # STEP 6 — CONSISTENCY + INTEGRITY
    # --------------------------------------------------

    print(
        "\n[6/8] CHECKING CONSISTENCY "
        "AND REFERENTIAL INTEGRITY"
    )

    (
        consistency_scores,
        consistency_report,
    ) = calculate_all_consistency(
        datasets=datasets,
        consistency_rules=CONSISTENCY_RULES
    )

    (
        integrity_scores,
        integrity_report,
    ) = calculate_referential_integrity(
        datasets=datasets,
        rules=REFERENTIAL_RULES
    )

    # --------------------------------------------------
    # STEP 7 — TIMELINESS
    # --------------------------------------------------

    print("\n[7/8] CALCULATING TIMELINESS")

    (
        timeliness_scores,
        timeliness_report,
    ) = calculate_all_timeliness(
        datasets=datasets,
        date_columns=DATE_COLUMNS
    )

    # --------------------------------------------------
    # STEP 8 — SCORE + DETECT ISSUES
    # --------------------------------------------------

    print(
        "\n[8/8] BUILDING DATA QUALITY "
        "SCORECARD"
    )

    quality_scorecard = (
        build_quality_scorecard(
            completeness=completeness_scores,
            uniqueness=uniqueness_scores,
            validity=validity_scores,
            consistency=consistency_scores,
            integrity=integrity_scores,
            timeliness=timeliness_scores,
            quality_weights=QUALITY_WEIGHTS
        )
    )

    overall_quality_score = (
        calculate_overall_quality_score(
            quality_scorecard
        )
    )

    completeness_issues = (
        detect_completeness_issues(
            completeness_report,
            SEVERITY_THRESHOLDS
        )
    )

    validity_issues = (
        detect_validity_issues(
            validity_report,
            SEVERITY_THRESHOLDS
        )
    )

    consistency_issues = (
        detect_consistency_issues(
            consistency_report,
            SEVERITY_THRESHOLDS
        )
    )

    integrity_issues = (
        detect_integrity_issues(
            integrity_report,
            SEVERITY_THRESHOLDS
        )
    )

    detected_issues = combine_detected_issues(
        [
            completeness_issues,
            validity_issues,
            consistency_issues,
            integrity_issues,
        ]
    )

    # --------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------

    print("\nSaving DataTrust reports...")

    save_dataframe(
        profile_report,
        "dataset_profile.csv"
    )

    save_dataframe(
        completeness_report,
        "completeness_report.csv"
    )

    save_dataframe(
        uniqueness_report,
        "uniqueness_report.csv"
    )

    save_dataframe(
        validity_report,
        "validity_report.csv"
    )

    save_dataframe(
        consistency_report,
        "consistency_report.csv"
    )

    save_dataframe(
        integrity_report,
        "referential_integrity_report.csv"
    )

    save_dataframe(
        timeliness_report,
        "timeliness_report.csv"
    )

    save_dataframe(
        quality_scorecard,
        "data_quality_scorecard.csv"
    )

    save_dataframe(
        detected_issues,
        "detected_quality_issues.csv"
    )

    overall_score = pd.DataFrame(
        [{
            "overall_data_quality_score": (
                overall_quality_score
            )
        }]
    )

    save_dataframe(
        overall_score,
        "overall_quality_score.csv"
    )

    print("\n" + "=" * 80)
    print("DATATRUST AI ENGINE COMPLETED")
    print("=" * 80)

    print(
        f"Overall Data Quality Score: "
        f"{overall_quality_score:.2f}/100"
    )

    print(
        "\nLowest Quality Datasets:"
    )

    print(
        quality_scorecard[
            [
                "dataset",
                "data_quality_score",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    print(
        f"\nReports saved to:\n"
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    run_data_quality_engine()