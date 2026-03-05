# RecruitAI — Resume Screening & Ranking System


<p align="center">
  An ML-powered system that automatically scores, ranks, and analyzes candidates against specific job roles — using 
  TF-IDF vectorization, cosine similarity, and a dynamic spaCy NLP skill extraction pipeline.
</p>

---

## Overview

Manually reviewing thousands of resumes is slow and inconsistent. RecruitAI solves this by converting unstructured resume text into machine-comparable vectors, ranking every candidate by relevance to a specific job role, and surfacing exactly which skills each candidate has — and which they are missing.

This version is specifically optimized to interface with a large-scale real-world dataset to provide high-accuracy benchmarking.

---

## Features

- **Dynamic Role Benchmarking**: Automatically pulls "Gold Standard" requirements from the preloaded dataset based on the job role you enter.

**Interactive Console Mode**: Screen multiple student resumes in real-time by entering names and pasting content.

**TF-IDF + Cosine Similarity**: Quantitative scoring (0–100%) that measures the mathematical "fit" of a candidate.

**NLP Skill Extraction**: Uses spaCy (en_core_web_sm) to intelligently identify technical skills using Part-of-Speech (POS) tagging.

**Skill Gap Analysis**: Compares candidate profiles against job requirements to highlight missing competencies.

**Ranked Leaderboard**: Outputs results in a clean, sorted DataFrame table for immediate decision-making.
---

## Tech Stack

Layer	                       Technology
Language	        Python 3.9+
ML / NLP	        scikit-learn (TF-IDF, Cosine Similarity), spaCy
Data Handling   	Pandas, NumPy
Dataset	          9,544-record real-world resume CSV

---

## How It Works

### 1 — Text Preprocessing
Each resume and the job description go through a cleaning pipeline: lowercasing → special character removal → stopword filtering. This ensures the model focuses on professional terminology rather than filler words.

### 2 — TF-IDF Vectorization
The vectorizer builds a weighted vocabulary where rare, specific technical terms (like "Hadoop" or "PyTorch") get boosted, while common resume words are down-weighted. This makes the ranking highly sensitive to technical expertise.

### 3 — Cosine Similarity Scoring

```
The system calculates the angle between the Job Requirement vector and the Candidate vector.
Score (%) = cosine_similarity(Requirement_Vector, Candidate_Vector) × 100
This method is document-length invariant, meaning a concise 1-page resume is compared fairly against a detailed 3-page resume.
```

### 4 — Skill Gap Identification
Using Python set logic combined with spaCy's POS tagging:

Missing Skills = {Benchmark Keywords} - {Candidate Keywords}
### 5 — Ranking & Categorization

| Score | Category |
|---|---|
| ≥ 75% | ✅ Strong Fit |
| 50 – 74% | 🟡 Moderate Fit |
| < 50% | 🔴 Weak Fit |

---

## Project Structure

```
recruit-ai/
│
├── main.py                     # Core execution script (Interactive Console)
├── utils/
│   ├── text_preprocessing.py    # Regex cleaning and normalization
│   └── skill_extraction.py      # spaCy NLP logic & gap analysis
│
├── data/
│   └── resume_data.csv         # 9,544-record master dataset
│
├── requirements.txt            # Project dependencies
└── README.md                   # System documentation
```

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/Mohith140306/
FUTURE_ML_03
cd
FUTURE_ML_03


# 2. Install dependencies
pip install pandas scikit-learn spacy
python -m spacy download en_core_web_sm

# 3. Run the app
python app.py
```

## Future Improvements

- [ ] Semantic Similarity: Implement BERT embeddings to recognize that "NLP" and "Natural Language Processing" are identical.

[ ] OCR Support: Add pdfminer.six to support direct PDF file uploads.

[ ] Web Dashboard: Transition the console UI to a Flask or Streamlit web application.

[ ] Section Weighting: Assign higher importance to the "Experience" and "Skills" sections than "Interests".

---
## Author
     Mohith Dappadi

Built as an ML Internship Project
