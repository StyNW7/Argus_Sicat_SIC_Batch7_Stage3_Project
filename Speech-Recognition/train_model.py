#!/usr/bin/env python3
"""
train_model.py
Train audio classifier from CSV output.

Usage:
python train_model.py --csv audio_dataset.csv --model_out model.joblib --scaler_out scaler.joblib
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns:
        X = df.drop(columns=['timestamp','label']).values
    else:
        X = df.drop(columns=['label']).values
    y = df['label'].values
    return X, y

def train_and_evaluate(csv_path, model_out, scaler_out):
    X, y = load_csv(csv_path)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    # stratify to keep class distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    # RandomForest baseline
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_s, y_train)
    y_pred = rf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print("RandomForest Accuracy:", acc)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    # SVM baseline
    svc = SVC(kernel='rbf', probability=True)
    svc.fit(X_train_s, y_train)
    y_pred2 = svc.predict(X_test_s)
    acc2 = accuracy_score(y_test, y_pred2)
    print("SVM Accuracy:", acc2)
    print(classification_report(y_test, y_pred2, target_names=le.classes_))
    # choose best
    if acc >= acc2:
        best = rf
        print("Selected RandomForest as best model")
    else:
        best = svc
        print("Selected SVM as best model")
    # save model & scaler & label encoder
    joblib.dump(best, model_out)
    joblib.dump(scaler, scaler_out)
    joblib.dump(le, model_out + '.labelenc')
    print(f"Saved model to {model_out}, scaler to {scaler_out}, labelencoder to {model_out+'.labelenc'}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Input CSV path')
    parser.add_argument('--model_out', default='model.joblib', help='Model output path')
    parser.add_argument('--scaler_out', default='scaler.joblib', help='Scaler output path')
    args = parser.parse_args()
    train_and_evaluate(args.csv, args.model_out, args.scaler_out)
