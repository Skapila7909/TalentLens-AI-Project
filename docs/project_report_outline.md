# TalentLens: AI Resume Skill Analyzer

## Abstract
TalentLens is an AI/NLP-based application designed to compare a candidate's resume with a target job description. It extracts skills from both texts, calculates a transparent skill-match score, identifies missing skills, evaluates basic ATS-oriented structure, and generates an improvement plan.

## Problem Statement
Job descriptions and resumes contain large amounts of unstructured text. Manually identifying the overlap between them is repetitive and error-prone. The project aims to automate this first-pass analysis.

## Objectives
1. Extract text from common resume formats.
2. Identify technical and professional skills.
3. Compare resume skills with target job requirements.
4. Quantify skill alignment.
5. Identify missing skills.
6. Provide actionable resume-improvement feedback.

## Methodology
Input → Text extraction → Preprocessing → Skill extraction → Matching → Scoring → Recommendations.

## Technologies
- Python
- Streamlit
- pdfplumber
- python-docx
- Regular expressions
- NLP vocabulary matching

## Evaluation
Test the system on resumes with known skills and compare extracted skills with manually verified labels. Report precision, recall and F1-score if a labeled test set is created.

## Limitations
The baseline uses a curated vocabulary and therefore cannot understand every synonym or contextual use of a skill. Presence of a skill does not prove proficiency.

## Future Scope
Semantic embeddings, NER, machine-learning classification, multilingual NLP, and a larger anonymized benchmark dataset.

## Conclusion
TalentLens demonstrates how AI and NLP can transform unstructured resume text into useful career-oriented insights while remaining transparent enough for educational use.
