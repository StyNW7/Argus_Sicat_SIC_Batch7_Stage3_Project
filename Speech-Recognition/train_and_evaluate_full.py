#!/usr/bin/env python3
"""
train_and_evaluate_full.py

Usage:
python train_and_evaluate_full.py --csv audio_dataset.csv --out_dir models_output

python train_and_evaluate_full.py --csv audio_dataset_final.csv --out_dir models_output_final

What it does:
- Loads CSV having columns: timestamp,rms,zcr,spectral_centroid,label,mfcc_1..mfcc_13
- Preprocess (drop timestamp, scale features, encode labels)
- Stratified split into train/val/test
- Train multiple models (RandomForest, SVM, GradientBoosting)
- Evaluate on test set (accuracy, precision, recall, f1)
- Plot confusion matrix and model comparison charts
- Save best model, scaler, label encoder and plots into out_dir
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    # ensure label column present
    if 'label' not in df.columns:
        raise ValueError("CSV must contain 'label' column")
    X = df.drop(columns=['label']).values
    y = df['label'].values
    return X, y, df

def train_models(X_train, y_train):
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM-rbf": SVC(kernel='rbf', probability=True, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    fitted = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {}
    print("Training models (with 5-fold CV)...")
    for name, m in models.items():
        scores = cross_val_score(m, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        cv_scores[name] = scores
        print(f" {name}: CV f1_macro mean={scores.mean():.4f} std={scores.std():.4f}")
        m.fit(X_train, y_train)
        fitted[name] = m
    return fitted, cv_scores

def evaluate_model(model, X_test, y_test, label_encoder=None):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, target_names=(label_encoder.classes_ if label_encoder is not None else None))
    cm = confusion_matrix(y_test, y_pred)
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred
    }

def plot_confusion(cm, classes, outpath):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_model_comparison(cv_scores, outpath):
    # cv_scores: dict name -> array of scores
    names = list(cv_scores.keys())
    means = [cv_scores[n].mean() for n in names]
    plt.figure(figsize=(7,4))
    sns.barplot(x=names, y=means)
    plt.ylim(0,1)
    plt.ylabel("CV f1_macro (mean)")
    plt.title("Model comparison (5-fold CV f1_macro)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main(args):
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset:", csv_path)
    X, y, df_full = load_dataset(csv_path)

    # Label encode
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_
    print("Classes found:", classes)

    # Train/Val/Test split (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1764706, random_state=42, stratify=y_temp)
    # Explanation: second split ensures final splits ~ 70/15/15

    print(f"Sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Train models
    fitted_models, cv_scores = train_models(X_train_s, y_train)

    # Evaluate each on test set
    results = {}
    for name, model in fitted_models.items():
        print(f"\nEvaluating model: {name}")
        res = evaluate_model(model, X_test_s, y_test, label_encoder=le)
        results[name] = res
        print(f" Accuracy: {res['accuracy']:.4f}  F1-macro: {res['f1_macro']:.4f}")
        print("Classification Report:\n", res['report'])

        # Save confusion plot per model
        cm_path = out_dir / f"confusion_{name}.png"
        plot_confusion(res['confusion_matrix'], classes, cm_path)
        print(f"Saved confusion matrix to {cm_path}")

    # Model comparison plot
    comp_path = out_dir / "model_comparison_cv_f1.png"
    plot_model_comparison(cv_scores, comp_path)
    print(f"Saved model comparison to {comp_path}")

    # Choose best model by f1_macro on test set
    best_name = max(results.keys(), key=lambda k: results[k]['f1_macro'])
    best_model = fitted_models[best_name]
    best_metrics = results[best_name]
    print(f"\nSelected best model: {best_name} (test F1-macro={best_metrics['f1_macro']:.4f})")

    # Save model, scaler, label encoder
    model_out = out_dir / "best_model.joblib"
    scaler_out = out_dir / "scaler.joblib"
    label_out = out_dir / "label_encoder.joblib"
    joblib.dump(best_model, model_out)
    joblib.dump(scaler, scaler_out)
    joblib.dump(le, label_out)
    print(f"Saved best model to {model_out}, scaler to {scaler_out}, label encoder to {label_out}")

    # Save a summary CSV of test predictions
    y_pred = results[best_name]['y_pred']
    inv_true = le.inverse_transform(y_test)
    inv_pred = le.inverse_transform(y_pred)
    out_df = pd.DataFrame({"true_label": inv_true, "pred_label": inv_pred})
    out_df.to_csv(out_dir / "test_predictions.csv", index=False)
    print(f"Saved test predictions to {out_dir / 'test_predictions.csv'}")

    # Save final metrics summary
    summary = {
        "model": best_name,
        "accuracy": float(best_metrics['accuracy']),
        "f1_macro": float(best_metrics['f1_macro']),
        "precision_macro": float(best_metrics['precision_macro']),
        "recall_macro": float(best_metrics['recall_macro'])
    }
    pd.DataFrame([summary]).to_csv(out_dir / "metrics_summary.csv", index=False)
    print(f"Saved metrics summary to {out_dir / 'metrics_summary.csv'}")

    print("\nAll done. Plots and models saved in:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV dataset path")
    parser.add_argument("--out_dir", default="models_output", help="Output directory for models and plots")
    args = parser.parse_args()
    main(args)
