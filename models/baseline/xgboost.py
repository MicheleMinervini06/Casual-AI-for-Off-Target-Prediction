from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import pickle
import logging

import numpy as np

from xgboost import XGBClassifier
import shap

log = logging.getLogger(__name__)

HYPERPARAMETER_GRID: dict[str, list[Any]] = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.1],
    "n_estimators": [100, 300],
    "subsample": [0.8, 1.0],
}


@dataclass
class XGBoostWrapper:
    params: dict[str, Any] | None = None
    early_stopping_rounds: int = 30
    feature_names: list[str] = field(default_factory=list)
    _shap_explainer: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed.")

        default = {
            "max_depth": 5,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": 42,
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
        feature_names: Optional[list[str]] = None,
    ) -> "XGBoostWrapper":
        if feature_names:
            self.feature_names = feature_names

        # Gestione imbalance
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        log.info("scale_pos_weight=%.2f (imbalance %.1fx)", scale_pos_weight, scale_pos_weight)

        self.model = XGBClassifier(
            **self._base_params,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        self._shap_explainer = None  # reset cache
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Restituisce array [n_samples, 2] — uniforme con gli altri modelli."""
        return np.asarray(self.model.predict_proba(X), dtype=float)

    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        SHAP values per sample con TreeSHAP (esatto).
        Restituisce array [n_samples, n_features].
        """
        if shap is None:
            raise ImportError("shap is not installed.")
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.model)

        vals = self._shap_explainer.shap_values(X)
        if isinstance(vals, list):
            vals = vals[1]  # classe positiva
        return np.asarray(vals, dtype=float)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostWrapper":
        with open(Path(path), "rb") as f:
            return pickle.load(f)