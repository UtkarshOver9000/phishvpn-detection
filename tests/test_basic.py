from phishvpn.model import evaluate_model, train_model
from phishvpn.synthetic_data import generate_synthetic


def test_train_and_evaluate():
    df = generate_synthetic(2000, seed=3)
    result = train_model(df, test_size=0.2, seed=3)
    metrics = evaluate_model(result.model, result.x_test, result.y_test)
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0
