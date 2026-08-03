# Resume ATS System

An NLP pipeline that classifies resumes, extracts skills/education, and matches
resumes against job descriptions. Built on the
[Kaggle "Resume Dataset"](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
(2484 resumes, 24 job categories).

## Features

- **Category classification** - predicts one of 24 job categories (e.g. `HR`,
  `CHEF`, `INFORMATION-TECHNOLOGY`) using a tuned TF-IDF + Linear SVM
  model. 69% accuracy, 0.67 macro-F1 (random guessing would be ~4%).
- **Skills & education extraction** - matches a curated list of 566+
  skill/education terms against resume text with spaCy's `PhraseMatcher`.
  General-purpose NER doesn't have a "skill" category and misclassifies terms
  like "Machine Learning" as an organization, so keyword matching is used instead.
- **Job matching** - scores how well a resume fits a job description using two
  metrics: `cosine_similarity` (overall textual similarity) and
  `skill_coverage` (percentage of the job's required skills the candidate has).
  Both are reported together - relying on cosine similarity alone is misleading
  (see *Known limitations*).
- **Best-candidate search** - given a job description and a pool of resumes,
  returns the top matches, ranked by skill coverage first (with cosine
  similarity as a tiebreaker).
- **PDF ingestion** - accepts raw PDF resumes directly.

## What was explored but not shipped

Named Entity Recognition (candidate name, employer, location, dates) was built
and tested in `EDA.ipynb` but not added to `main.py`. Candidate identification
uses filenames instead - simpler, and NER-based name extraction fails on this
specific dataset anyway (see *Known limitations*).

## Repo structure

```
main.py                  - the shipped pipeline
skills_keywords.py        - skill/education keyword lists
model/resume_classifier.joblib  - trained model
dataset/                  - not included, see Setup
EDA.ipynb                 - full development notebook (methodology, experiments)
EDA_kaggle.ipynb           - portable variant for running on Kaggle
requirements.txt
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_md
   ```
2. Download the [dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
   and place it under `dataset/` (not included - third-party data, ~180MB).
3. `model/resume_classifier.joblib` is included, so `main.py` works immediately
   without retraining. To retrain, run `EDA.ipynb` top to bottom.

## Usage

```python
import main as ats

# Classify a resume
category = ats.predict_category_from_pdf("path/to/resume.pdf")

# Extract skills and education
profile = ats.extract_skills_and_education_from_pdf("path/to/resume.pdf")
# {'skills': ['Python', 'Machine Learning', ...], 'education': ['Bachelor of Science']}

# Rank a pool of resumes against a job description
ranking = ats.rank_resume_pdfs_for_job(job_description_text, {
    "candidate_a": "path/to/resume_a.pdf",
    "candidate_b": "path/to/resume_b.pdf",
})
# [{'candidate': 'candidate_a', 'cosine_similarity': 0.68, 'skill_coverage': 1.0}, ...]

# Find the best-matching candidates from a pool
best = ats.find_best_candidate_pdfs(job_description_text, {
    "candidate_a": "path/to/resume_a.pdf",
    "candidate_b": "path/to/resume_b.pdf",
}, top_n=3)
```

## Why TF-IDF + Linear SVM, not deep learning

- The dataset is small for DL (2481 resumes, some categories as few as 22-36
  examples) - not enough data per class to fine-tune a transformer reliably.
- Averaged word embeddings were tested directly and scored meaningfully worse
  than TF-IDF on this classification task.
- It's inspectable - every real diagnostic during development (e.g. why
  `CONSULTANT` gets confused with `BUSINESS-DEVELOPMENT`) was only possible
  because the model's decisions trace back to specific words.
- Logistic Regression, Random Forest, and Linear SVM were all tuned via
  `GridSearchCV` (5-fold stratified CV, `f1_macro`); SVM won on both
  cross-validated and held-out macro-F1 and is the model shipped here.

## Known limitations (honest, not hidden)

- **Smallest classes are weakest**: `BPO` (22 examples), `AUTOMOBILE` (36), and
  `AGRICULTURE` (63) score lowest and are unstable across experiments due to
  tiny test sets (as few as 4 test examples for `BPO`). A data-scarcity issue,
  not something a different model would fix.
- **Cosine similarity can penalize well-qualified candidates**: a candidate
  with *more* skills than a job requires scores *lower* on cosine similarity
  than one with only the exact required skills, even when both meet every
  requirement - extra content dilutes the similarity vector. This is why
  `skill_coverage` is reported alongside it.
- **Soft skills (Teamwork, Communication, etc.) are excluded from matching.**
  They appear on almost any resume regardless of domain, and were found to let
  unrelated candidates outscore genuinely qualified ones on `skill_coverage`
  purely by listing common buzzwords.
- **This dataset's resumes are anonymized** (company/city/state are literal
  placeholders, and candidate names appear stripped entirely). NER's
  `PERSON`/`GPE` extraction correctly finds nothing here, but works normally on
  real, non-anonymized resumes.
