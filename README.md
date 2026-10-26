# TalentLens: Resume Skill Analyzer + Category Classifier

TalentLens is a Streamlit app that extracts text from PDF/DOCX/TXT resumes, detects skills, compares them with a job description, reports missing skills, checks basic resume structure, and predicts a resume category using a trained text classifier.

## What works in this version

- PDF, DOCX, and TXT upload; DOCX tables, headers, and footers are included.
- Boundary-aware skill matching and common aliases (for example `sklearn` → `scikit-learn`, `PowerBI` → `power bi`).
- A one-skill job description such as `Python` is valid and should return a 100% overlap when Python is detected in the resume.
- If no known target skills are detected, the app shows `N/A` rather than a misleading 0%.
- A separate TF-IDF + Linear SVM classifier predicts one of the nine categories in the supplied dataset.
- Automated unit tests for core matching behavior.
- No Transformers/PyTorch/GPU downloads required.

## Run locally

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_talentlens_core.py
python -m streamlit run app.py
```

For fish shell, activate with `source .venv/bin/activate.fish`. Open `http://localhost:8501`.

## Dataset and classifier

The committed `resume_category_model.joblib` was trained from the provided 9,000-row resume CSV after: (1) normalizing resume text for duplicate checks, (2) removing exact duplicate texts, and (3) excluding texts assigned to conflicting categories. This left 5,437 unique, non-conflicting resumes across nine categories. The model uses the first 6,000 characters of each resume, TF-IDF word unigrams/bigrams, and a class-weighted LinearSVC.

On a stratified 20% holdout split, the saved evaluation recorded **87.8% accuracy** and **87.1% balanced accuracy**. See `training_metrics.json` for details. This is a benchmark on the same dataset, not proof of generalization to all professions or a resume-to-job match accuracy score. The classifier only predicts the nine categories represented in the training data.

To retrain using your own downloaded CSV:

```bash
python train_classifier.py "Resume dataset.csv"
```

This writes `resume_category_model.joblib` and `training_metrics.json`. The CSV is intentionally not included in this repository because it is large and may contain personal information. Verify the Kaggle dataset license and usage terms before publicly deploying or redistributing derived artifacts.

## Deploy on Streamlit Community Cloud

1. Upload the project files to GitHub, including `talentlens_core.py`, `resume_category_model.joblib`, `requirements.txt`, and `app.py`.
2. Do not upload `.venv`, raw resumes, or the original dataset CSV.
3. Set the Streamlit main file to `app.py`.
4. Deploy.

## Important limitations

- The category model covers only nine categories and cannot accurately label every profession.
- The skill score is a transparent overlap metric, not a hiring decision or proof of competence.
- The skill vocabulary will miss terms not listed; OCR is not included for image-only scanned PDFs.
- ATS checks are simple structural signals, not a certification of compatibility with any employer's ATS.
- Test with fictional or consented resumes and avoid storing personal resume data in public repositories.
