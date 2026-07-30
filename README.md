# Resume ATS System

An NLP pipeline that classifies resumes into job categories, extracts structured
skills/education, and scores how well a resume matches a specific job description.
Built on the [Kaggle "Resume Dataset"](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
(2484 resumes across 24 job categories).

## What it does

- **Category classification** - given a resume, predicts one of 24 job categories
  (e.g. `INFORMATION-TECHNOLOGY`, `HR`, `CHEF`) using a tuned TF-IDF + Logistic
  Regression model.
- **Skills & education extraction** - pulls out a candidate's skills and education
  level using a curated keyword list (566+ terms across 17 domains) matched via
  spaCy's `PhraseMatcher`, since general-purpose NER has no "skill" category and
  reliably misclassifies terms like "Machine Learning" or "CNN" as organizations.
- **Job matching** - given a job description and a pool of resumes, ranks
  candidates two ways: `cosine_similarity` (overall textual similarity) and
  `skill_coverage` (fraction of the job's required skills the candidate actually
  has). Both are reported together deliberately - see *Known limitations* below
  for why relying on cosine similarity alone is misleading.
- **PDF ingestion** - accepts raw PDF resumes directly (`pdfplumber` extraction),
  not just pre-processed text.

## What it explored but does not ship

- **Named Entity Recognition** (candidate name, employers, locations, dates) was
  built and validated in `EDA.ipynb` using a hybrid of spaCy (DATE) and a
  HuggingFace transformer (`dslim/bert-base-NER`, scored, for PERSON/ORG/LOCATION)
  - but was deliberately **not wired into `main.py`**. Candidate identification
  in the shipped system uses filenames instead: simpler and more reliable than
  NER-based name extraction, which fails entirely on this dataset anyway (see
  below) and is a genuine risk of misfiring on any resume.

## Repo structure

```
main.py                 - the shipped pipeline (classify, extract, match)
skills_keywords.py       - curated skill/education keyword lists + canonicalization
model/
  resume_classifier.joblib  - trained TF-IDF + LogisticRegression pipeline
dataset/                 - NOT included, see Setup below
requirements.txt
```

The full development notebook (`EDA.ipynb`) documents the entire methodology
end to end: EDA, cleaning, tokenization, model comparison, hyperparameter
tuning, entity extraction experiments, and the job-matching metric fix.
`EDA_kaggle.ipynb` is a self-contained variant of the same notebook adapted
to run on Kaggle (no local file dependencies, dataset/model paths need the
Kaggle-side adjustments noted in comments in that file).

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_md
   ```
2. Download the [Kaggle "Resume Dataset"](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
   and place it under `dataset/` (not included in this repo - it's third-party
   data, and the CSV + PDFs together run ~180MB).
3. The trained model (`model/resume_classifier.joblib`) is included, so you can
   use `main.py` immediately without retraining. To retrain from scratch, run
   `EDA.ipynb` top to bottom.

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
# [{'candidate': 'candidate_a', 'cosine_similarity': 0.39, 'skill_coverage': 1.0}, ...]
```

## Methodology summary

TF-IDF + Logistic Regression (tuned via `GridSearchCV` with stratified k-fold
CV: `C=10, min_df=2, max_df=0.85`) was chosen over deep learning approaches
deliberately, not by default:

- The dataset is small for DL (2481 resumes, some categories as few as 22-36
  examples) - not enough data per class to fine-tune a transformer reliably.
- Averaged word embeddings were tested directly and scored meaningfully worse
  than TF-IDF for this exact classification task.
- TF-IDF + Logistic Regression is inspectable, which mattered throughout: every
  real diagnostic (e.g. discovering `CONSULTANT` gets confused with
  `BUSINESS-DEVELOPMENT` because of overlapping real-world vocabulary) was only
  possible because the model's decisions can be traced back to specific words.

**Final result:** 67% accuracy, 0.65 macro-F1 across 24 classes (random guessing
would be ~4%).

## Known limitations (honest, not hidden)

- **Smallest classes are the weakest**: `BPO` (22 examples), `AUTOMOBILE` (36),
  and `AGRICULTURE` (63) have the lowest per-class F1 scores, and their scores
  are noticeably unstable across experiments due to tiny test-set sizes (as few
  as 4 test examples for `BPO`). This is a data-scarcity limitation, not
  something a different model would fix.
- **Cosine similarity for job matching can be misleading on its own**: a
  candidate with *more* skills than a job requires scores *lower* via cosine
  similarity than one with only the exact required skills, even when both have
  every required skill - the "extra" content dilutes the similarity vector.
  Verified directly: a real CV with all 5 of a job's required skills (plus 38
  more) scored 0.389, while a hypothetical resume with only those exact 5
  skills scored a perfect 1.0. This is why `rank_resumes_for_job` reports
  `skill_coverage` alongside cosine similarity - coverage answers "does this
  candidate meet the requirements," unaffected by how many extra skills they
  also have.
- **This dataset's resumes are anonymized** - company names, cities, and states
  are replaced with literal placeholder text ("Company Name", "City", "State"),
  and candidate names appear to be stripped entirely. This is why NER's
  `PERSON`/`GPE` extraction fails completely on this dataset (there's nothing
  real to find) despite working correctly on real, non-anonymized resumes -
  confirmed directly by testing the same extraction on a real CV.
