"""
ML Models Module
================
Implements Logistic Regression and Naive Bayes classifiers,
combined via a linear weighted average of predicted probabilities.
"""

import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)


class HybridMLModel:
    """
    Hybrid ML model that combines Logistic Regression and Naive Bayes
    predictions via linear weighted average.

    Final score = alpha * P(LR) + (1 - alpha) * P(NB)
    """

    def __init__(self, alpha=0.5):
        """
        Args:
            alpha: Weight for Logistic Regression (0 to 1).
                   (1-alpha) is the weight for Naive Bayes.
        """
        self.alpha = alpha
        self.lr_model = LogisticRegression(
            max_iter=1000, random_state=42, C=1.0
        )
        self.nb_model = GaussianNB()
        self.is_trained = False
        self.feature_names = None
        self.training_metrics = {}

    def train(self, X, y, feature_names=None):
        """
        Train both models on the data.

        Args:
            X: Feature matrix (numpy array or pandas DataFrame).
            y: Labels (0 = reject, 1 = accept).
            feature_names: List of feature names for interpretability.
        """
        self.feature_names = feature_names or [
            f"feature_{i}" for i in range(X.shape[1])
        ]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train Logistic Regression
        self.lr_model.fit(X_train, y_train)
        lr_pred = self.lr_model.predict(X_test)

        # Train Naive Bayes
        self.nb_model.fit(X_train, y_train)
        nb_pred = self.nb_model.predict(X_test)

        # Hybrid predictions on test set
        lr_proba = self.lr_model.predict_proba(X_test)[:, 1]
        nb_proba = self.nb_model.predict_proba(X_test)[:, 1]
        hybrid_proba = self.alpha * lr_proba + (1 - self.alpha) * nb_proba
        hybrid_pred = (hybrid_proba >= 0.5).astype(int)

        # Store metrics
        self.training_metrics = {
            "logistic_regression": {
                "accuracy": round(accuracy_score(y_test, lr_pred), 4),
                "precision": round(precision_score(y_test, lr_pred, zero_division=0), 4),
                "recall": round(recall_score(y_test, lr_pred, zero_division=0), 4),
                "f1": round(f1_score(y_test, lr_pred, zero_division=0), 4),
            },
            "naive_bayes": {
                "accuracy": round(accuracy_score(y_test, nb_pred), 4),
                "precision": round(precision_score(y_test, nb_pred, zero_division=0), 4),
                "recall": round(recall_score(y_test, nb_pred, zero_division=0), 4),
                "f1": round(f1_score(y_test, nb_pred, zero_division=0), 4),
            },
            "hybrid": {
                "accuracy": round(accuracy_score(y_test, hybrid_pred), 4),
                "precision": round(precision_score(y_test, hybrid_pred, zero_division=0), 4),
                "recall": round(recall_score(y_test, hybrid_pred, zero_division=0), 4),
                "f1": round(f1_score(y_test, hybrid_pred, zero_division=0), 4),
                "alpha": self.alpha,
            },
        }

        self.is_trained = True
        return self.training_metrics

    def predict(self, X):
        """
        Make predictions using the hybrid model.

        Args:
            X: Feature matrix.

        Returns:
            dict: {
                "decision": "ACCEPT" or "REJECT",
                "confidence": float (0-1),
                "lr_probability": float,
                "nb_probability": float,
                "hybrid_probability": float,
                "feature_contributions": dict
            }
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        X = np.array(X).reshape(1, -1) if len(np.array(X).shape) == 1 else X

        lr_proba = self.lr_model.predict_proba(X)[:, 1][0]
        nb_proba = self.nb_model.predict_proba(X)[:, 1][0]
        hybrid_proba = self.alpha * lr_proba + (1 - self.alpha) * nb_proba

        decision = "ACCEPT" if hybrid_proba >= 0.5 else "REJECT"

        # Feature contributions from LR coefficients
        contributions = {}
        if self.feature_names and hasattr(self.lr_model, 'coef_'):
            coefs = self.lr_model.coef_[0]
            for name, coef, val in zip(
                self.feature_names, coefs, X.flatten()
            ):
                contributions[name] = {
                    "weight": round(float(coef), 4),
                    "value": round(float(val), 4),
                    "contribution": round(float(coef * val), 4),
                }

        return {
            "decision": decision,
            "confidence": round(float(hybrid_proba), 4),
            "lr_probability": round(float(lr_proba), 4),
            "nb_probability": round(float(nb_proba), 4),
            "hybrid_probability": round(float(hybrid_proba), 4),
            "feature_contributions": contributions,
        }

    def save(self, directory="models"):
        """Save trained models to disk."""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.lr_model, os.path.join(directory, "lr_model.pkl"))
        joblib.dump(self.nb_model, os.path.join(directory, "nb_model.pkl"))
        joblib.dump({
            "alpha": self.alpha,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
        }, os.path.join(directory, "model_meta.pkl"))

    def load(self, directory="models"):
        """Load trained models from disk."""
        self.lr_model = joblib.load(os.path.join(directory, "lr_model.pkl"))
        self.nb_model = joblib.load(os.path.join(directory, "nb_model.pkl"))
        meta = joblib.load(os.path.join(directory, "model_meta.pkl"))
        self.alpha = meta["alpha"]
        self.feature_names = meta["feature_names"]
        self.training_metrics = meta["training_metrics"]
        self.is_trained = True
