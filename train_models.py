"""
Model Trainer Module
====================
Trains the hybrid ML model using synthetic resume data.
Extracts features from resumes and trains Logistic Regression
and Naive Bayes models.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.feature_extractor import FeatureExtractor
from modules.ml_models import HybridMLModel
from data.prepare_data import generate_dataset, JOB_SKILL_POOLS


def train_model_for_category(df, category, models_dir="models"):
    """
    Train a hybrid model for a specific job category.

    Args:
        df: DataFrame with resume data.
        category: Job category key.
        models_dir: Directory to save models.
    """
    pool = JOB_SKILL_POOLS[category]
    category_df = df[df["job_category"] == category].copy()

    if len(category_df) < 10:
        print(f"  Skipping {category}: too few samples ({len(category_df)})")
        return None

    # Extract features for each resume
    extractor = FeatureExtractor(
        required_skills=pool["required"],
        min_experience=pool["min_experience"],
        min_education=pool["min_education"],
    )

    feature_vectors = []
    labels = []

    for _, row in category_df.iterrows():
        features = extractor.extract(row["resume_text"])
        fv = features["feature_vector"]
        feature_vectors.append([
            fv["skill_match_ratio"],
            fv["experience_years"],
            fv["education_score"],
            fv["total_skills_normalized"],
        ])
        labels.append(row["label"])

    X = np.array(feature_vectors)
    y = np.array(labels)

    feature_names = [
        "skill_match_ratio", "experience_years",
        "education_score", "total_skills"
    ]

    # Train hybrid model
    model = HybridMLModel(alpha=0.5)
    metrics = model.train(X, y, feature_names=feature_names)

    # Save model
    category_dir = os.path.join(models_dir, category)
    model.save(category_dir)

    return metrics


def train_all_models(num_samples=600):
    """Train models for all job categories."""
    print("=" * 60)
    print("TRAINING HYBRID ML MODELS")
    print("=" * 60)

    # Generate data
    print("\n1. Generating synthetic dataset...")
    df = generate_dataset(num_samples=num_samples)

    # Train per-category models
    print("\n2. Training models per job category...")
    all_metrics = {}

    for category in JOB_SKILL_POOLS:
        print(f"\n  Training: {category}")
        metrics = train_model_for_category(df, category)
        if metrics:
            all_metrics[category] = metrics
            print(f"    LR Accuracy:     {metrics['logistic_regression']['accuracy']:.4f}")
            print(f"    NB Accuracy:     {metrics['naive_bayes']['accuracy']:.4f}")
            print(f"    Hybrid Accuracy: {metrics['hybrid']['accuracy']:.4f}")
            print(f"    Hybrid F1:       {metrics['hybrid']['f1']:.4f}")

    # Save metrics summary
    os.makedirs("models", exist_ok=True)
    with open("models/training_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Models saved to models/ directory")
    print("=" * 60)

    return all_metrics


if __name__ == "__main__":
    train_all_models(num_samples=600)
