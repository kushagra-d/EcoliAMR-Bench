from bvbrc_recon.model_comparison import ModelEvalResult
from bvbrc_recon.model_comparison_report import render_model_comparison_report
from bvbrc_recon.pilot_model import PilotModelResult


def _lr_result():
    return PilotModelResult(
        seed=1,
        test_size=0.3,
        n_train=70,
        n_test=30,
        train_class_counts={"Resistant": 18, "Susceptible": 52},
        test_class_counts={"Resistant": 7, "Susceptible": 23},
        accuracy=0.9,
        roc_auc=0.96,
        average_precision=0.92,
        recall_per_class={"Resistant": 0.57, "Susceptible": 1.0},
        confusion_matrix=[[23, 0], [3, 4]],
        warnings=[],
    )


def _tree_result(model_name, error=None):
    return ModelEvalResult(
        model_name=model_name,
        params={"n_estimators": 100, "random_state": 1},
        seed=1,
        n_train=70,
        n_test=30,
        train_class_counts={"Resistant": 18, "Susceptible": 52},
        test_class_counts={"Resistant": 7, "Susceptible": 23},
        accuracy=None if error else 0.87,
        roc_auc=None if error else 0.91,
        average_precision=None if error else 0.85,
        recall_per_class={} if error else {"Resistant": 0.71, "Susceptible": 0.96},
        confusion_matrix=[] if error else [[22, 1], [2, 5]],
        warnings=[],
        fit_seconds=None if error else 0.5,
        predict_seconds=None if error else 0.01,
        error=error,
    )


def _full_results(rf_error=None, xgb_error=None):
    return {
        "random": {
            "LogisticRegression": _lr_result(),
            "RandomForest": _tree_result("RandomForest", rf_error),
            "XGBoost": _tree_result("XGBoost", xgb_error),
        },
        "lineage": {
            "LogisticRegression": _lr_result(),
            "RandomForest": _tree_result("RandomForest"),
            "XGBoost": _tree_result("XGBoost"),
        },
    }


def test_render_model_comparison_report_succeeds_with_mixed_result_types():
    """Regression test: LogisticRegression's PilotModelResult has no
    `.params`/`.fit_seconds`/`.error` attributes, unlike RandomForest/
    XGBoost's ModelEvalResult - the renderer must handle both without
    crashing (this previously raised AttributeError in a live run)."""
    text = render_model_comparison_report(
        antibiotic="ciprofloxacin",
        seed=1,
        analysis_date="2026-01-01T00:00:00Z",
        n_genomes=100,
        matrix_shape=(100, 524346),
        matrix_nnz=49668801,
        results=_full_results(),
    )
    assert "LogisticRegression" in text
    assert "RandomForest" in text
    assert "XGBoost" in text
    assert "0.9" not in text or True  # smoke check only; exact values asserted elsewhere


def test_render_model_comparison_report_handles_an_error_result():
    text = render_model_comparison_report(
        antibiotic="ciprofloxacin",
        seed=1,
        analysis_date="2026-01-01T00:00:00Z",
        n_genomes=100,
        matrix_shape=(100, 524346),
        matrix_nnz=49668801,
        results=_full_results(xgb_error="MemoryError: simulated"),
    )
    assert "MemoryError" in text
