from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import logging
import pickle

import numpy as np

try:
    from catboost import CatBoostClassifier, EFstrType, Pool
except Exception:  # pragma: no cover - handled at runtime
    CatBoostClassifier = None
    EFstrType = None
    Pool = None


HYPERPARAMETER_GRID: dict[str, list[Any]] = {
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.1],
    "iterations": [200, 500],
    "l2_leaf_reg": [3, 5, 7],
}

log = logging.getLogger(__name__)


@dataclass
class CatBoostWrapper:
    params: dict[str, Any] | None = None
    early_stopping_rounds: int = 30
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed. Run: uv sync")

        default = {
            "depth": 6,
            "learning_rate": 0.08,
            "iterations": 300,
            "loss_function": "Logloss",
            "verbose": False,
            "random_seed": 42,
        }
        if self.params:
            default.update(self.params)
        self._base_params = default

    @property
    def hyperparameter_grid(self) -> dict[str, list[Any]]:
        return HYPERPARAMETER_GRID

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "CatBoostWrapper":
        if feature_names:
            self.feature_names = feature_names

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        log.info("scale_pos_weight=%.2f (imbalance %.1fx)", scale_pos_weight, scale_pos_weight)

        model_cls = CatBoostClassifier
        if model_cls is None:
            raise ImportError("catboost is not installed. Run: uv sync")

        self.model = model_cls(
            **self._base_params,
            class_weights=[1.0, scale_pos_weight],
        )
        self.model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            verbose=False,
            use_best_model=True,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X), dtype=float)

    def explain(self, X: np.ndarray) -> np.ndarray:
        if EFstrType is None:
            raise ImportError("catboost is not installed. Run: uv sync")
        if Pool is None:
            raise ImportError("catboost is not installed. Run: uv sync")
        pool = Pool(X, feature_names=self.feature_names or None)
        shap_values = np.asarray(
            self.model.get_feature_importance(type=EFstrType.ShapValues, data=pool),
            dtype=float,
        )
        if shap_values.ndim == 2 and shap_values.shape[1] > 1:
            return shap_values[:, :-1]
        return shap_values

    def feature_importance(self) -> np.ndarray:
        return np.asarray(self.model.get_feature_importance(), dtype=float)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "CatBoostWrapper":
        with open(Path(path), "rb") as f:
            return pickle.load(f)
