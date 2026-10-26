# TalentLens — AI Resume Skill Analyzer

A polished CBSE Class 12 Artificial Intelligence capstone project that analyzes a resume against a target job description.

## Features

- PDF, DOCX and TXT resume upload
- Automatic skill extraction
- Target-job skill extraction
- Skill match percentage
- Missing-skill analysis
- Category-level skill breakdown
- ATS-style structural readiness score
- Action plan for improving the resume
- Built-in demo dataset
- Runs locally without sending resume content to an external API

## Run

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```text
resume_skill_analyzer/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── docs/
└── report/
```

## AI/NLP pipeline

Resume → text extraction → normalization → skill extraction → target skill extraction → matching → scoring → recommendations.

The current version uses a resume-specific BERT Named Entity Recognition (NER) model (`yashpwr/resume-ner-bert`) to extract entities, especially SKILLS, from resume/job text. A curated vocabulary remains as a transparent fallback if the model cannot be loaded. This is intentional for a school project: every prediction can be explained in a viva.

## Possible future improvements

- TF-IDF + machine-learning skill classification
- Named Entity Recognition (NER)
- semantic similarity using sentence embeddings
- job-role classification
- multilingual resume support
- learned ranking model
- anonymized benchmark dataset
