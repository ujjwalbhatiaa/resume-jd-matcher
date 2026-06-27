# Resume ↔ Job-Description Matcher

A dependency-free Python tool that compares a **resume** against a **job
description** and tells you exactly how well they line up: an overall match
score, a skill-by-skill gap analysis, and concrete, tailored suggestions for
closing the biggest gaps.

It's the kind of keyword/skill alignment that applicant-tracking systems (ATS)
and recruiters screen on — surfaced before you hit *Apply*.

> Built from scratch in **pure Python (3.8+) standard library only** — no numpy,
> no scikit-learn, no NLP libraries. The TF-IDF vectoriser and cosine
> similarity are implemented by hand.

## What it does

Given two text inputs it reports:

- **Overall match score (0–100)** — a weighted blend of skill coverage (70%)
  and TF-IDF text similarity (30%).
- **Matched skills** — required skills the JD asks for that your resume already
  shows.
- **Missing skills** — in the JD but absent from your resume (the gaps to close).
- **Extra skills** — on your resume but not requested (useful for trimming).
- **Tailored suggestions** — highest-impact missing skills first, each with a
  concrete action (add a project, containerise something, expose a REST API…).

Example, comparing the included sample resume against a sample ML-intern JD:

```
  Overall match score :  33.0 / 100   (Weak match — significant tailoring needed)
  Skill coverage      :  38.9 / 100   (7/18 JD skills present)
  Text similarity     :  19.4 / 100   (TF-IDF cosine)

  ✓ MATCHED SKILLS (7)
      Programming Languages: Python, SQL
      ML / AI: Machine Learning, Statistics
      ...
  ✗ MISSING (11)
      Frameworks & Libraries: PyTorch, TensorFlow
      ML / AI: Deep Learning, NLP
      ...
  ➜ SUGGESTIONS (highest-impact gaps first)
      • [Deep Learning] Build or fine-tune a small neural network ...
      • [NLP] Ship a text-processing project ...
```

## Quick start

```bash
# Human-readable report
python main.py --resume samples/resume_sample.txt --jd samples/jd_sample.txt

# Machine-readable JSON (pipe into jq, store, diff over time)
python main.py -r samples/resume_sample.txt -j samples/jd_sample.txt --json

# Paste a job description on stdin instead of a file
pbpaste | python main.py -r my_resume.txt --jd -
```

Run the test suite (19 tests):

```bash
python -m unittest -v
```

## How it works

1. **Skill detection (alias-aware).** `skills_db.py` holds a curated taxonomy of
   ~80 skills, each with its surface forms — so `JS` ↔ `JavaScript`,
   `k8s` ↔ `Kubernetes`, and `natural language processing` ↔ `NLP` all resolve
   to the same canonical skill. The analyzer builds every 1–3 word phrase in
   each document and looks them up, so multi-word skills are matched correctly.

2. **TF-IDF + cosine similarity (from scratch).** `tfidf.py` computes
   length-normalised term frequencies, smoothed inverse-document-frequencies
   (`idf(t) = ln((1+N)/(1+df)) + 1`), turns each document into a sparse weight
   vector, and measures the cosine of the angle between them. Identical texts
   score 1.0; disjoint texts score 0.0.

3. **Scoring & suggestions.** `analyzer.py` combines skill coverage and text
   similarity into one 0–100 score, then ranks the missing skills (technical
   skills before soft skills, since those are what filters screen on) and emits
   an actionable bullet for each.

## Project layout

| File               | Responsibility                                              |
|--------------------|-------------------------------------------------------------|
| `skills_db.py`     | Curated skill taxonomy with aliases and categories          |
| `text_utils.py`    | Tokenisation, stop-words, n-gram phrase extraction          |
| `tfidf.py`         | From-scratch TF-IDF vectoriser + cosine similarity          |
| `analyzer.py`      | Skill detection, scoring, gap analysis, suggestions, report |
| `main.py`          | Command-line interface                                       |
| `test_matcher.py`  | 19 unit + integration tests                                 |
| `samples/`         | A sample resume and job description                         |

## Design notes & honest limitations

- **Input is plain text.** PDF/DOCX parsing is intentionally out of scope to
  keep the project dependency-free; paste or export your resume to `.txt`
  first. PDF extraction is a natural next extension.
- **The skill taxonomy is curated, not learned.** That makes the tool
  transparent and easy to extend (add a row to `skills_db.py`), at the cost of
  only recognising skills it knows about.
- **TF-IDF is a bag-of-words measure** — it captures vocabulary overlap, not
  deep semantics. A future version could swap in sentence embeddings while
  keeping the same interface.

## License

MIT — see [LICENSE](LICENSE).
