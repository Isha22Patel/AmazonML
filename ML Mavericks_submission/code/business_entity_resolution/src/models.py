"""Matching models module for Amazon ML Business Entity Resolution.

Contains implementations for all 5 evaluated model architectures:
- Model 0: Rule Baseline (Exact Name & Country Match)
- Model 1: Blocking-Only Baseline (All Candidates Admitted as Matches)
- Model 2: Logistic Regression (Scaled, L2-regularized, Class-Weighted)
- Model 3: LightGBM GBDT (Winning Model, Balanced Class Weights, 31 Leaves)
- Model 4: Random Forest (150 Trees, Class-Weighted)
"""

import sys
from pathlib import Path
_SRC_DIR = Path(__file__).resolve().parent
_PKG_DIR = _SRC_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.config import FEATURE_NAMES


class RuleBaselineModel:
    """Model 0: Exact normalized name and country match baseline."""

    def __init__(self) -> None:
        self.name_exact_idx = FEATURE_NAMES.index("name_exact")
        self.country_exact_idx = FEATURE_NAMES.index("country_exact")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RuleBaselineModel":
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        exact = (X[:, self.name_exact_idx] == 1.0) & (X[:, self.country_exact_idx] == 1.0)
        probs = np.zeros((X.shape[0], 2), dtype=np.float32)
        probs[exact, 1] = 1.0
        probs[~exact, 0] = 1.0
        probs[exact, 0] = 0.0
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class BlockingOnlyBaselineModel:
    """Model 1: Blocking-only baseline (predicts all candidates as matches)."""

    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BlockingOnlyBaselineModel":
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.zeros((X.shape[0], 2), dtype=np.float32)
        probs[:, 1] = 1.0
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return np.ones(X.shape[0], dtype=int)


class LogisticRegressionMatcher:
    """Model 2: Scaled Logistic Regression with balanced class weighting."""

    def __init__(self, random_state: int = 42, max_iter: int = 1000) -> None:
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            class_weight="balanced",
            random_state=random_state,
            max_iter=max_iter,
            solver="lbfgs",
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionMatcher":
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @property
    def feature_importances(self) -> np.ndarray:
        return self.clf.coef_[0]


class LightGBMMatcher:
    """Model 3: LightGBM Gradient Boosted Decision Tree (Winning Model)."""

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = 6,
        subsample: float = 0.8,
        subsample_freq: int = 1,
        random_state: int = 42,
    ) -> None:
        self.clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            subsample=subsample,
            subsample_freq=subsample_freq,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMMatcher":
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @property
    def feature_importances(self) -> np.ndarray:
        return self.clf.feature_importances_


class RandomForestMatcher:
    """Model 4: Random Forest Classifier for ensemble benchmark comparison."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: Optional[int] = 12,
        random_state: int = 42,
    ) -> None:
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestMatcher":
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @property
    def feature_importances(self) -> np.ndarray:
        return self.clf.feature_importances_
