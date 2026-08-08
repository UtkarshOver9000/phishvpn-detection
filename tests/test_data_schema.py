"""
Sanity tests for the schema definition — guards against drift between
CATEGORICAL_COLUMNS / NUMERIC_COLUMNS and ALL_COLUMNS.
"""

from phishvpn.data_schema import (
    ALL_COLUMNS,
    CATEGORICAL_COLUMNS,
    IDENTIFIER_COLUMNS,
    NUMERIC_COLUMNS,
    SCHEMA,
    TARGET_COLUMN,
)


def test_all_columns_is_the_union_of_feature_and_target_columns():
    assert ALL_COLUMNS == IDENTIFIER_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [TARGET_COLUMN]


def test_no_overlap_between_categorical_and_numeric_columns():
    assert not set(CATEGORICAL_COLUMNS) & set(NUMERIC_COLUMNS)


def test_identifier_columns_are_excluded_from_model_features():
    # asn is high-cardinality/near-unique per session; it must stay out of
    # CATEGORICAL_COLUMNS or one-hot encoding it lets the model memorize
    # training rows instead of generalizing (see data_schema.py comment).
    assert not set(IDENTIFIER_COLUMNS) & set(CATEGORICAL_COLUMNS)
    assert not set(IDENTIFIER_COLUMNS) & set(NUMERIC_COLUMNS)


def test_schema_dataclass_matches_module_level_columns():
    assert SCHEMA.categorical == CATEGORICAL_COLUMNS
    assert SCHEMA.numeric == NUMERIC_COLUMNS
    assert SCHEMA.target == TARGET_COLUMN
