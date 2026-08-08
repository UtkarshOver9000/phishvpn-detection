"""
Unit tests for the synthetic VPN session generator.
"""

from phishvpn.data_schema import ALL_COLUMNS
from phishvpn.synthetic_data import generate_synthetic


def test_generated_columns_match_schema():
    df = generate_synthetic(200, seed=1)
    assert list(df.columns) == ALL_COLUMNS
    assert len(df) == 200


def test_generation_is_deterministic_given_a_seed():
    df1 = generate_synthetic(300, seed=42)
    df2 = generate_synthetic(300, seed=42)
    assert df1.equals(df2)


def test_different_seeds_produce_different_data():
    df1 = generate_synthetic(300, seed=1)
    df2 = generate_synthetic(300, seed=2)
    assert not df1.equals(df2)


def test_label_is_binary_and_not_degenerate():
    df = generate_synthetic(5000, seed=7)
    labels = set(df["is_phishing"].unique().tolist())
    assert labels <= {0, 1}
    positive_rate = df["is_phishing"].mean()
    # Sanity: neither "nothing is ever phishing" nor "everything is".
    assert 0.01 < positive_rate < 0.5
