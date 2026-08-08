"""
Unit tests for the model pipeline: build, train, evaluate, save/load round-trip.
"""

from phishvpn.model import build_pipeline, evaluate_model, load_model, save_model, train_model
from phishvpn.synthetic_data import generate_synthetic


def test_build_pipeline_has_prep_and_clf_steps():
    pipeline = build_pipeline()
    names = [name for name, _ in pipeline.steps]
    assert names == ["prep", "clf"]


def test_train_model_produces_fitted_pipeline_with_expected_split():
    df = generate_synthetic(1000, seed=3)
    result = train_model(df, test_size=0.25, seed=3)
    assert len(result.x_test) == 250
    assert len(result.x_train) == 750
    # A fitted sklearn pipeline can predict without raising.
    result.model.predict(result.x_test)


def test_evaluate_model_reports_auc_and_classification_report():
    df = generate_synthetic(2000, seed=3)
    result = train_model(df, test_size=0.2, seed=3)
    metrics = evaluate_model(result.model, result.x_test, result.y_test)
    assert 0.5 <= metrics["roc_auc"] <= 1.0
    report = metrics["classification_report"]
    assert "0" in report and "1" in report
    assert "accuracy" in report


def test_model_achieves_strong_held_out_auc():
    # Regression guard: an earlier version of the synthetic label formula
    # combined with one-hot-encoding the near-unique `asn` column produced a
    # near-random held-out ROC-AUC (~0.55-0.58) despite ~0.96 in-sample AUC --
    # a textbook overfitting/high-cardinality-encoding bug. This should not
    # come back.
    df = generate_synthetic(20000, seed=11)
    result = train_model(df, test_size=0.2, seed=11)
    metrics = evaluate_model(result.model, result.x_test, result.y_test)
    assert metrics["roc_auc"] >= 0.8


def test_save_and_load_model_round_trip(tmp_path):
    df = generate_synthetic(500, seed=3)
    result = train_model(df, test_size=0.2, seed=3)

    model_path = tmp_path / "model.joblib"
    save_model(result.model, str(model_path))
    assert model_path.exists()

    loaded = load_model(str(model_path))
    original_preds = result.model.predict_proba(result.x_test)[:, 1]
    loaded_preds = loaded.predict_proba(result.x_test)[:, 1]
    assert (original_preds == loaded_preds).all()
