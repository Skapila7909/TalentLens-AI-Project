# CBSE Viva Notes — TalentLens

## 1. What problem does the project solve?
Recruiters often compare a resume with the requirements of a job. TalentLens automates the first-pass skill comparison and highlights gaps.

## 2. What is AI doing here?
The system uses natural-language processing to extract meaningful skill terms from unstructured text and uses those terms to make a recommendation score.

## 3. What is NLP?
Natural Language Processing is a branch of AI that enables computers to process and analyze human language.

## 4. How is the score calculated?
Skill Match = (Matched target skills / Total target skills) × 100.

## 5. Why not just use an if-else program?
The project still contains rules, but the input is unstructured natural language. Text normalization, vocabulary-based entity/skill extraction and language analysis make it an NLP pipeline rather than a simple form-based calculator.

## 6. What are false positives?
A word may appear in a resume without proving strong expertise. A production system would use context, embeddings or an ML classifier to reduce this problem.

## 7. What are false negatives?
A resume may contain a synonym or related technology that is not in the vocabulary. Expanding the vocabulary or using semantic similarity would help.

## 8. What is an ATS?
Applicant Tracking System. It is software used by organizations to process and filter job applications.

## 9. What are limitations?
- Curated skill vocabulary
- No deep semantic understanding in the baseline
- Skill presence does not prove proficiency
- ATS score is an approximation, not a real recruiter's decision

## 10. Future scope
Use transformer-based embeddings, NER, a labeled resume/job dataset, semantic similarity and a learned ranking model.

## 11. Ethical considerations
The tool should support applicants rather than make high-stakes hiring decisions. It should avoid protected personal attributes and should not claim that a score guarantees employment.
