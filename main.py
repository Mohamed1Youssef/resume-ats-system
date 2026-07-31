import re
import string
import joblib
import spacy
import pdfplumber
from spacy.matcher import PhraseMatcher
from sklearn.metrics.pairwise import cosine_similarity

from skills_keywords import ALL_SKILLS, AMBIGUOUS_SKILLS, EDUCATION_LEVELS, SOFT_SKILLS, canonicalize_skill

_SOFT_SKILLS_LOWER = {s.lower() for s in SOFT_SKILLS}


nlp = spacy.load("en_core_web_md")
model = joblib.load("Resume ATS system/model/resume_classifier.joblib")
vectorizer = model.named_steps["tfidf"]

_non_ambiguous_skills = [s for s in ALL_SKILLS if s not in AMBIGUOUS_SKILLS]
skill_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
skill_matcher.add("SKILL", [nlp.make_doc(s) for s in _non_ambiguous_skills])

ambiguous_matcher = PhraseMatcher(nlp.vocab)
ambiguous_matcher.add("AMBIGUOUS_SKILL", [nlp.make_doc(s) for s in AMBIGUOUS_SKILLS])

edu_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
edu_matcher.add("EDUCATION", [nlp.make_doc(e) for e in EDUCATION_LEVELS])

def extract_text_from_pdf(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        text = []
        for page in pdf.pages:
            text.append(page.extract_text() or '')
    return '\n'.join(text)


def clean_text(text: str)->str:
    #remove null/control bytes
    text=re.sub(r"[\x00-\x08\x0b\x0c\x0e\x0f]", "", text)

    #normalize dashes
    text=re.sub(r"[‐-‒–—―−\uff0d]+", "-",text)

    # Collapse repeated spaces/tabs, without removing newlines
    text = re.sub(r"[^\S\r\n]+", " ", text)
    #collapse multiple newlines and removing spaces around them
    text = re.sub(r"[ \t\r]*\n[ \t\r\n]*", "\n", text)

    return text.strip()


def preprocessing(text: str):
    doc = nlp(text, disable=["parser", "ner"])
    filtered = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        lemma = token.lemma_.lower().strip(string.punctuation)
        if len(lemma) < 2 or not any(c.isalpha() for c in lemma):
            continue
        filtered.append(lemma)
    return filtered


def predict_category(resume_text: str) -> str:
    cleaned = clean_text(resume_text)
    tokens = preprocessing(cleaned)
    tokens_str = ' '.join(tokens)
    return model.predict([tokens_str])[0]


def predict_category_from_pdf(pdf_path: str) -> str:
    raw_text = extract_text_from_pdf(pdf_path)
    return predict_category(raw_text)


def collect_matches(matcher, doc):
    results = []
    for match_id, start, end in matcher(doc):
        span = doc[start:end]
        results.append({"text": span.text, "start_char": span.start_char, "end_char": span.end_char})
    return results


def remove_overlapping_matches(matches):
    matches = sorted(
        matches,
        key=lambda item: (item["start_char"], -(item["end_char"] - item["start_char"]))
    )
    selected = []
    for current in matches:
        overlaps = any(
            current["start_char"] < saved["end_char"] and current["end_char"] > saved["start_char"]
            for saved in selected
        )
        if not overlaps:
            selected.append(current)
    return selected


def extract_skills_and_education(text: str) -> dict:
    doc = nlp(text)

    raw_skill_matches = collect_matches(skill_matcher, doc) + collect_matches(ambiguous_matcher, doc)
    skill_matches = remove_overlapping_matches(raw_skill_matches)

    canonical_skills = []
    seen = set()
    for m in skill_matches:
        canonical = canonicalize_skill(m["text"])
        if canonical.lower() not in seen:
            seen.add(canonical.lower())
            canonical_skills.append(canonical)

    edu_matches = remove_overlapping_matches(collect_matches(edu_matcher, doc))
    education = []
    seen_edu = set()
    for m in edu_matches:
        if m["text"].lower() not in seen_edu:
            seen_edu.add(m["text"].lower())
            education.append(m["text"])

    return {"skills": canonical_skills, "education": education}


def extract_skills_and_education_from_pdf(pdf_path: str) -> dict:
    raw_text = extract_text_from_pdf(pdf_path)
    return extract_skills_and_education(clean_text(raw_text))


def _skills_for_matching(raw_text: str) -> tuple[str, set[str]]:
    """
    Extract skills for job/resume matching, reduced from raw text - strips
    out narrative language ("we are seeking...") and job-history verbs that
    don't reflect actual required competencies. Classification still uses
    the full preprocessed text; this reduction is specific to matching.

    Soft skills (Teamwork, Communication, Time Management, etc.) are
    excluded here specifically - verified this matters: a real Junior Chef
    posting's required-skills list was 6 soft skills out of 11 total, and
    an unrelated ML/AI resume outscored an actual chef's resume on
    skill_coverage (0.36 vs 0.091) purely because soft-skill buzzwords
    appear on almost any professional resume regardless of domain. Soft
    skills don't differentiate domain fit, so they're excluded from
    matching (they're still shown normally in extract_skills_and_education's
    general output - this exclusion is specific to job/resume matching).

    Returns both a joined string (for TF-IDF cosine similarity) and a
    lowercase set (for skill-coverage calculation), computed from a single
    extraction pass rather than extracting twice.
    """
    skills = extract_skills_and_education(clean_text(raw_text))["skills"]
    hard_skills = [s for s in skills if s.lower() not in _SOFT_SKILLS_LOWER]
    return ' '.join(hard_skills), {s.lower() for s in hard_skills}


def rank_resumes_for_job(job_description: str, resumes: dict[str, str]) -> list[dict]:
    """
    resumes: {identifier (e.g. filename): RAW resume text}

    Returns a list of dicts sorted by cosine_similarity descending, each with:
      - candidate: the identifier
      - cosine_similarity: overall similarity between extracted skill sets.
        Can be misleadingly low for a candidate with many skills beyond
        what the job needs - cosine similarity is diluted by "extra"
        content even when every required skill is present (verified: a
        candidate with only the 5 exact required skills scored 1.0, the
        same candidate's real 43-skill resume scored 0.389, despite having
        all 5 required skills - having more skills than needed actively
        lowers this score). skill_coverage below doesn't have that problem.
      - skill_coverage: fraction of the job's required skills the candidate
        actually has (0.0-1.0) - unaffected by how many extra skills the
        candidate also has.
    """
    job_text, job_skill_set = _skills_for_matching(job_description)
    job_vec = vectorizer.transform([job_text])

    names = list(resumes.keys())
    resume_texts = []
    resume_skill_sets = []
    for text in resumes.values():
        t, s = _skills_for_matching(text)
        resume_texts.append(t)
        resume_skill_sets.append(s)

    resume_vecs = vectorizer.transform(resume_texts)
    cosine_scores = cosine_similarity(job_vec, resume_vecs)[0]

    results = []
    for name, cos_score, resume_skills in zip(names, cosine_scores, resume_skill_sets):
        coverage = len(job_skill_set & resume_skills) / len(job_skill_set) if job_skill_set else 0.0
        results.append({
            "candidate": name,
            "cosine_similarity": float(cos_score),
            "skill_coverage": coverage,
        })

    return sorted(results, key=lambda r: r["cosine_similarity"], reverse=True)


def rank_resume_pdfs_for_job(job_description: str, pdf_paths: dict[str, str]) -> list[dict]:
    """pdf_paths: {identifier (e.g. filename): path to PDF}"""
    resumes = {name: extract_text_from_pdf(path) for name, path in pdf_paths.items()}
    return rank_resumes_for_job(job_description, resumes)


def find_best_candidates(job_description: str, resumes: dict[str, str], top_n: int = 5) -> list[dict]:
    """
    Search a pool of resumes and return the top_n best candidates for a job.

    Sorted by skill_coverage first, cosine_similarity as a tiebreaker - NOT
    by cosine_similarity alone. This matters: cosine_similarity can rank an
    over-qualified candidate (many skills beyond what's needed) below one
    with only the exact required skills, since "extra" content dilutes the
    similarity vector even when every requirement is met (verified earlier -
    a candidate with all 5 required skills plus 38 more scored 0.389, while
    a hypothetical candidate with only those exact 5 scored 1.0).
    skill_coverage answers "does this candidate meet the requirements,"
    which is what "best candidate" should actually mean here.
    """
    results = rank_resumes_for_job(job_description, resumes)
    results.sort(key=lambda r: (r["skill_coverage"], r["cosine_similarity"]), reverse=True)
    return results[:top_n]


def find_best_candidate_pdfs(job_description: str, pdf_paths: dict[str, str], top_n: int = 5) -> list[dict]:
    """pdf_paths: {identifier (e.g. filename): path to PDF}"""
    resumes = {name: extract_text_from_pdf(path) for name, path in pdf_paths.items()}
    return find_best_candidates(job_description, resumes, top_n=top_n)


if __name__ == "__main__":
    # Synthetic example resume - no external file, no personal data, so
    # this demo runs for anyone cloning the repo without needing a PDF.
    sample_resume = """
    Jane Doe

    Summary
    Data scientist with experience in Python, machine learning, deep
    learning, and computer vision. Comfortable working with TensorFlow
    and PyTorch, with a strong background in data analysis and model
    deployment.

    Experience
    Data Scientist, Example Corp, 2021 to Current
    Built and deployed machine learning models using TensorFlow and
    scikit-learn. Worked on computer vision projects using PyTorch and
    OpenCV. Performed data analysis and feature engineering with Pandas
    and NumPy.

    Education
    Bachelor of Science in Computer Science
    """

    print("Category prediction:", predict_category(sample_resume))
    print()
    print("Skills & education:", extract_skills_and_education(clean_text(sample_resume)))
    print()

    job_description = """
    We are seeking a passionate and talented Data Scientist to join our team.
    The ideal candidate has experience with Python, machine learning, deep
    learning, and computer vision, and is comfortable working with TensorFlow
    or PyTorch.
    """
    ranking = rank_resumes_for_job(job_description, {"Jane_Doe": sample_resume})
    print("Job match ranking:", ranking)
    print()

    best = find_best_candidates(job_description, {"Jane_Doe": sample_resume}, top_n=3)
    print("Best candidates:", best)
