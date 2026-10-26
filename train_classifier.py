"""Train TalentLens' nine-category resume classifier from the Kaggle CSV.

Usage:
  python train_classifier.py "Resume dataset.csv"

The script removes exact normalized duplicate resume texts and excludes texts
that have conflicting category labels. It uses a grouped-by-text clean dataset,
then evaluates on a held-out stratified split before saving the model.
"""
from __future__ import annotations
import argparse, json, re, unicodedata
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def normalize_text(value: object) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).lower()
    return re.sub(r"\s+", " ", value).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to the downloaded resume CSV")
    parser.add_argument("--model-out", default="resume_category_model.joblib")
    parser.add_argument("--metrics-out", default="training_metrics.json")
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)
    required = {"category", "Text"}
    if not required.issubset(df.columns):
        raise SystemExit(f"CSV must contain columns {sorted(required)}; found {list(df.columns)}")
    df = df.dropna(subset=["category", "Text"]).copy()
    df["_norm"] = df["Text"].map(normalize_text)
    df = df[df["_norm"].str.len() > 0]
    conflict_counts = df.groupby("_norm")["category"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    clean = df[~df["_norm"].isin(conflicts)].drop_duplicates("_norm").copy()
    if clean["category"].nunique() < 2:
        raise SystemExit("Need at least two categories to train a classifier.")
    X = clean["Text"].astype(str).str.slice(0, 6000).tolist()
    y = clean["category"].astype(str).tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000,
            min_df=3, sublinear_tf=True, strip_accents="unicode")),
        ("clf", LinearSVC(class_weight="balanced", C=1.0)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "source_rows": int(len(df)),
        "clean_unique_resumes": int(len(clean)),
        "conflicting_texts_excluded": int(len(conflicts)),
        "exact_duplicate_rows_removed": int(len(df) - len(clean) - int(df["_norm"].isin(conflicts).sum())),
        "category_counts_after_cleaning": {str(k): int(v) for k, v in clean["category"].value_counts().items()},
        "test_rows": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "text_used_per_resume_char_limit": 6000,
        "random_state": 42,
    }
    joblib.dump(model, args.model_out, compress=3)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))
    print(f"Rows in source: {len(df):,}")
    print(f"Clean unique resumes: {len(clean):,}")
    print(f"Conflicting texts excluded: {len(conflicts):,}")
    print(f"Held-out accuracy: {metrics['accuracy']:.1%}")
    print(f"Held-out balanced accuracy: {metrics['balanced_accuracy']:.1%}")
    print(f"Saved model: {args.model_out}")
    print(f"Saved metrics: {args.metrics_out}")

if __name__ == "__main__":
    main()
