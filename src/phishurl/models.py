"""
The three models compared in this project, each trained with a real
validation set and a recorded loss curve.

* Logistic regression: linear baseline.
* Histogram gradient boosting: the shipped model. Trained for up to
  ``max_rounds`` boosting rounds; train/validation log loss is recorded every
  round and the round count with the lowest validation loss is kept.
* MLP: a small neural network trained epoch by epoch with early stopping on
  validation log loss.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _linear_prep() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("tld", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def _tree_prep() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "tld",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                    min_frequency=20,
                    max_categories=250,
                ),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )


def _ll(y, p) -> float:
    return float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))


def train_logistic(x_train, y_train, x_val, y_val) -> tuple[Pipeline, dict]:
    model = Pipeline([("prep", _linear_prep()), ("clf", LogisticRegression(max_iter=3000))])
    model.fit(x_train, y_train)
    clf = model.named_steps["clf"]
    history = {
        "solver_iterations": int(np.max(clf.n_iter_)),
        "train_log_loss": round(_ll(y_train, model.predict_proba(x_train)[:, 1]), 4),
        "val_log_loss": round(_ll(y_val, model.predict_proba(x_val)[:, 1]), 4),
    }
    return model, history


def train_boosting(x_train, y_train, x_val, y_val, max_rounds: int = 600, seed: int = 7) -> tuple[Pipeline, dict]:
    def make(rounds: int) -> Pipeline:
        clf = HistGradientBoostingClassifier(
            max_iter=rounds,
            learning_rate=0.08,
            max_leaf_nodes=63,
            l2_regularization=1.0,
            categorical_features=[0],  # tld is the first column out of _tree_prep
            early_stopping=False,
            random_state=seed,
        )
        return Pipeline([("prep", _tree_prep()), ("clf", clf)])

    full = make(max_rounds).fit(x_train, y_train)
    prep, clf = full.named_steps["prep"], full.named_steps["clf"]
    xt, xv = prep.transform(x_train), prep.transform(x_val)
    train_curve = [_ll(y_train, p[:, 1]) for p in clf.staged_predict_proba(xt)]
    val_curve = [_ll(y_val, p[:, 1]) for p in clf.staged_predict_proba(xv)]
    best_rounds = int(np.argmin(val_curve)) + 1

    model = make(best_rounds).fit(x_train, y_train)
    history = {
        "rounds_trained": max_rounds,
        "best_round": best_rounds,
        "train_log_loss_per_round": [round(v, 5) for v in train_curve],
        "val_log_loss_per_round": [round(v, 5) for v in val_curve],
    }
    return model, history


def train_mlp(x_train, y_train, x_val, y_val, max_epochs: int = 60, patience: int = 6, seed: int = 7):
    prep = _linear_prep().fit(x_train)
    xt, xv = prep.transform(x_train), prep.transform(x_val)
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512, learning_rate_init=1e-3, random_state=seed
    )

    train_curve, val_curve, val_acc = [], [], []
    best, best_state, bad_epochs = np.inf, None, 0
    for _ in range(max_epochs):
        clf.partial_fit(xt, y_train, classes=[0, 1])
        p_val = clf.predict_proba(xv)[:, 1]
        train_curve.append(_ll(y_train, clf.predict_proba(xt)[:, 1]))
        val_curve.append(_ll(y_val, p_val))
        val_acc.append(float(((p_val >= 0.5).astype(int) == np.asarray(y_val)).mean()))
        if val_curve[-1] < best - 1e-4:
            best, bad_epochs = val_curve[-1], 0
            best_state = ([w.copy() for w in clf.coefs_], [b.copy() for b in clf.intercepts_])
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    clf.coefs_, clf.intercepts_ = best_state
    model = Pipeline([("prep", prep), ("clf", clf)])
    history = {
        "epochs_run": len(val_curve),
        "best_epoch": int(np.argmin(val_curve)) + 1,
        "train_log_loss_per_epoch": [round(v, 5) for v in train_curve],
        "val_log_loss_per_epoch": [round(v, 5) for v in val_curve],
        "val_accuracy_per_epoch": [round(v, 4) for v in val_acc],
    }
    return model, history
