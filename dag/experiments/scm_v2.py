from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from dag.nodes import CRISPRPairFeatures

# Aggiunte le forme per i modelli ad albero
EquationForm = Literal["linear", "multiplicative", "sigmoid", "tree_regressor", "tree_classifier"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _logit(p: float) -> float:
    clipped = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(clipped / (1.0 - clipped)))


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {context}: {missing}")


def _as_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    _require_columns(df, columns, "matrix extraction")
    return df[columns].to_numpy(dtype=float)


@dataclass(slots=True)
class StructuralEquation:
    """Single structural equation for one endogenous variable."""

    target: str
    parents: list[str]
    form: EquationForm
    coefficients: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    intercept: float = 0.0
    noise_std: float = 1.0
    fitted: bool = False
    model: Any = field(default=None)  # Contenitore per modelli sklearn (es. HistGradientBoosting)

    def fit(self, df: pd.DataFrame) -> "StructuralEquation":
        _require_columns(df, [self.target] + self.parents, f"equation {self.target}")

        y = df[self.target].to_numpy(dtype=float)
        X = _as_matrix(df, self.parents)

        if self.form == "multiplicative":
            if X.shape[1] != 1:
                raise ValueError(
                    f"Multiplicative equation for {self.target} requires exactly one parent"
                )
            x = X[:, 0]
            denom = float(np.dot(x, x)) + 1e-8
            coef = float(np.dot(x, y) / denom)
            self.intercept = 0.0
            self.coefficients = np.asarray([coef], dtype=float)
            y_hat = coef * x

        elif self.form == "linear":
            design = np.column_stack([np.ones(len(X), dtype=float), X])
            params, *_ = np.linalg.lstsq(design, y, rcond=None)
            self.intercept = float(params[0])
            self.coefficients = np.asarray(params[1:], dtype=float)
            y_hat = self.intercept + (X @ self.coefficients)

        elif self.form == "sigmoid":
            y_binary = (y > 0).astype(int)
            if np.unique(y_binary).size < 2:
                base_rate = float(np.clip(y_binary.mean(), 1e-6, 1.0 - 1e-6))
                self.intercept = _logit(base_rate)
                self.coefficients = np.zeros(X.shape[1], dtype=float)
                y_hat = np.full(len(y_binary), base_rate, dtype=float)
            else:
                clf = LogisticRegression(
                    C=1e6, fit_intercept=True, solver="lbfgs", max_iter=1000
                )
                clf.fit(X, y_binary)
                self.intercept = float(clf.intercept_[0])
                self.coefficients = np.asarray(clf.coef_[0], dtype=float)
                y_hat = clf.predict_proba(X)[:, 1]

        elif self.form == "tree_regressor":
            self.model = HistGradientBoostingRegressor(
                max_iter=100, max_depth=5, early_stopping=False, random_state=42
            )
            self.model.fit(X, y)
            y_hat = self.model.predict(X)
            # Dummy values per mantenere coerenza di interfaccia
            self.intercept = 0.0
            self.coefficients = np.zeros(X.shape[1], dtype=float)

        elif self.form == "tree_classifier":
            y_binary = (y > 0).astype(int)
            self.model = HistGradientBoostingClassifier(
                max_iter=100, max_depth=5, early_stopping=False, random_state=42
            )
            self.model.fit(X, y_binary)
            y_hat = self.model.predict_proba(X)[:, 1]
            self.intercept = 0.0
            self.coefficients = np.zeros(X.shape[1], dtype=float)

        else:
            raise ValueError(f"Unknown equation form: {self.form}")

        residual = y - y_hat
        self.noise_std = float(max(np.std(residual), 1e-8))
        self.fitted = True
        return self

    def linear_predictor(self, df: pd.DataFrame) -> np.ndarray:
        """Return pre-link linear predictor. Invalid for tree models."""
        if not self.fitted:
            raise RuntimeError(f"Equation {self.target} is not fitted")
        if self.form in ("tree_regressor", "tree_classifier"):
            raise TypeError(f"linear_predictor is undefined for tree-based form '{self.form}'")

        X = _as_matrix(df, self.parents)
        if self.form == "multiplicative":
            z = self.coefficients[0] * X[:, 0]
        else:
            z = self.intercept + (X @ self.coefficients)
        return np.asarray(z, dtype=float)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError(f"Equation {self.target} is not fitted")

        if self.form == "tree_regressor":
            return self.model.predict(_as_matrix(df, self.parents))
        if self.form == "tree_classifier":
            return self.model.predict_proba(_as_matrix(df, self.parents))[:, 1]

        z = self.linear_predictor(df)
        if self.form == "sigmoid":
            return _sigmoid(z)
        return z

    def sample_noise(self, df: pd.DataFrame, observed: np.ndarray | None = None) -> np.ndarray:
        if observed is None:
            _require_columns(df, [self.target], f"noise sampling {self.target}")
            observed = df[self.target].to_numpy(dtype=float)
        pred = self.predict(df)
        return np.asarray(observed, dtype=float) - pred


@dataclass(slots=True)
class CRISPRCausalModel:
    """Non-parametric SCM aligned with the engineered DAG concepts.

    Equations:
    1) pam_score <- alpha * node_A_pam                    (multiplicative)
    2) mismatch_rate <- tree(B, C, D, mean_energy)        (tree_regressor)
    3) label <- tree(pam, B, C, D)                        (tree_classifier)
    """

    pam_equation: StructuralEquation = field(
        default_factory=lambda: StructuralEquation(
            target="pam_score",
            parents=["node_A_pam"],
            form="multiplicative",
        )
    )
    mismatch_equation: StructuralEquation = field(
        default_factory=lambda: StructuralEquation(
            target="mismatch_rate",
            parents=["node_B_proximal", "node_C_seed_extension", "node_D_non_seed", "mean_energy_penalty"],
            form="tree_regressor",
        )
    )
    activity_equation: StructuralEquation = field(
        default_factory=lambda: StructuralEquation(
            target="label",
            parents=["pam_score", "node_B_proximal", "node_C_seed_extension", "node_D_non_seed"],
            form="tree_classifier",
        )
    )
    fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "CRISPRCausalModel":
        if len(df) == 0:
            raise ValueError("Cannot fit SCM on an empty dataframe")

        required = {
            "node_A_pam",
            "pam_score",
            "node_B_proximal",
            "node_C_seed_extension",
            "node_D_non_seed",
            "mean_energy_penalty",
            "mismatch_rate",
            "label",
        }
        _require_columns(df, sorted(required), "CRISPRCausalModel.fit")

        self.pam_equation.fit(df)
        self.mismatch_equation.fit(df)

        state = df.copy()
        state["pam_score"] = self.pam_equation.predict(state)
        self.activity_equation.fit(state)

        self.fitted = True
        return self

    def _pair_to_state(self, pair: CRISPRPairFeatures | Mapping[str, float] | tuple[str, str]) -> dict[str, float]:
        if isinstance(pair, CRISPRPairFeatures):
            raw = pair.to_feature_dict()
        elif isinstance(pair, tuple) and len(pair) == 2:
            raw = CRISPRPairFeatures(sgRNA_seq=pair[0], off_seq=pair[1]).to_feature_dict()
        elif isinstance(pair, Mapping):
            raw = dict(pair)
        else:
            raise TypeError("pair must be CRISPRPairFeatures, mapping, or (sgRNA_seq, off_seq) tuple")

        needed = {
            "node_A_pam",
            "pam_score",
            "node_B_proximal",
            "node_C_seed_extension",
            "node_D_non_seed",
            "mean_energy_penalty",
            "mismatch_rate",
        }
        missing = [name for name in sorted(needed) if name not in raw]
        if missing:
            raise ValueError(f"Missing pair fields required by SCM: {missing}")

        state: dict[str, float] = {}
        for key in needed:
            state[key] = float(raw[key])
        return state

    def predict(self, pair: CRISPRPairFeatures | Mapping[str, float] | tuple[str, str]) -> dict[str, float]:
        if not self.fitted:
            raise RuntimeError("SCM must be fitted before prediction")

        state = self._pair_to_state(pair)
        frame = pd.DataFrame([state])

        pam_score_hat = float(self.pam_equation.predict(frame)[0])
        state["pam_score"] = pam_score_hat
        frame = pd.DataFrame([state])

        mismatch_rate_hat = float(self.mismatch_equation.predict(frame)[0])
        state["mismatch_rate"] = mismatch_rate_hat
        frame = pd.DataFrame([state])

        activity_prob = float(self.activity_equation.predict(frame)[0])
        activity_label = int(activity_prob >= 0.5)

        return {
            "pam_score": pam_score_hat,
            "mismatch_rate": mismatch_rate_hat,
            "activity_probability": activity_prob,
            "activity_label": activity_label,
        }

    def sample_exogenous(
        self,
        pair: CRISPRPairFeatures | Mapping[str, float] | tuple[str, str],
        observed_activity: float,
    ) -> dict[str, float]:
        """Infer exogenous noises (abduction) non-parametrically.
        
        Logit residuals are approximated safely for tree models using _logit(P_hat).
        """
        if not self.fitted:
            raise RuntimeError("SCM must be fitted before abduction")

        observed_state = self._pair_to_state(pair)
        predicted = self.predict(pair)

        observed_activity = float(observed_activity)
        observed_activity_clipped = float(np.clip(observed_activity, 1e-6, 1.0 - 1e-6))

        frame_cf = pd.DataFrame([{**observed_state, "pam_score": predicted["pam_score"]}])
        
        # Abduzione sicura per modelli ad albero senza linear_predictor
        predicted_prob_cf = float(self.activity_equation.predict(frame_cf)[0])
        predicted_logit_cf = _logit(predicted_prob_cf)

        u_pam = observed_state["pam_score"] - predicted["pam_score"]
        u_mismatch = observed_state["mismatch_rate"] - predicted["mismatch_rate"]
        u_activity_probability = observed_activity - predicted["activity_probability"]
        u_activity_logit = _logit(observed_activity_clipped) - predicted_logit_cf

        return {
            "u_pam": float(u_pam),
            "u_mismatch_rate": float(u_mismatch),
            "u_activity_probability": float(u_activity_probability),
            "u_activity_logit": float(u_activity_logit),
        }


__all__ = [
    "CRISPRCausalModel",
    "StructuralEquation",
]