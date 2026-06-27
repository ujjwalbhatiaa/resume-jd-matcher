"""
analyzer.py
===========
The heart of the tool. Given a resume and a job description (as text), it:

1. Detects which known skills each document mentions (alias-aware).
2. Computes a TF-IDF cosine similarity between the two texts.
3. Reports matched skills, missing skills (in the JD but not the resume), and
   "extra" resume skills the JD doesn't ask for.
4. Produces a single 0-100 match score blending skill coverage and text overlap.
5. Generates concrete, tailored bullet suggestions for the biggest gaps.

No third-party dependencies — standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from skills_db import SKILLS, CATEGORY_LABELS, build_alias_index, category_of
from text_utils import phrase_set
from tfidf import text_similarity

_ALIAS_INDEX = build_alias_index()

# Weighting of the two score components. Skill coverage is what recruiters and
# ATS keyword filters care about most, so it carries the majority of the score.
SKILL_WEIGHT = 0.7
TEXT_WEIGHT = 0.3


def detect_skills(text: str) -> set[str]:
    """Return the set of canonical skills mentioned anywhere in `text`.

    Works by building every 1-3 word phrase in the text and looking each up in
    the alias index, so "k8s", "k-nearest", "natural language processing" and
    "JS" all resolve to their canonical skill names.
    """
    phrases = phrase_set(text, max_n=3)
    found: set[str] = set()
    for phrase in phrases:
        canonical = _ALIAS_INDEX.get(phrase)
        if canonical:
            found.add(canonical)
    return found


# Hand-written, skill-specific bullet templates for the suggestion engine.
# Falls back to a generic template for skills without a bespoke line.
_BULLET_TEMPLATES: dict[str, str] = {
    "Python": "Strengthen a Python project (tests, typing, packaging) and lead the bullet with measurable impact.",
    "Machine Learning": "Add a small end-to-end ML project (train → evaluate → report a metric) to demonstrate the full workflow.",
    "Deep Learning": "Build or fine-tune a small neural network (e.g. PyTorch/TensorFlow) and quote the validation metric you achieved.",
    "NLP": "Ship a text-processing project (classification, search, or summarisation) and describe the dataset and result.",
    "SQL": "Add a data project that writes non-trivial SQL (joins, aggregations, window functions) against a real dataset.",
    "Docker": "Containerise one of your projects and mention the Dockerfile + one-command run in the README.",
    "Git": "Make your GitHub commit history visible — clean messages, a README, and a pinned repo signal Git fluency.",
    "Testing": "Add a unit-test suite to an existing project and state the test count / coverage in the bullet.",
    "React": "Build a small React front-end (even a single page) so the JD's UI requirement is concretely evidenced.",
    "REST API": "Expose one of your projects through a small REST API (Flask/FastAPI) and document the endpoints.",
    "AWS": "Deploy a project to a free-tier cloud service and note the platform; cloud exposure de-risks you for recruiters.",
}


@dataclass
class MatchReport:
    """Structured result of a resume↔JD comparison."""
    score: float                                   # 0-100 overall
    skill_coverage: float                          # 0-100, share of JD skills met
    text_similarity: float                         # 0-100, TF-IDF cosine
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)
    jd_skills: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        """JSON-serialisable view of the report."""
        return {
            "score": round(self.score, 1),
            "skill_coverage": round(self.skill_coverage, 1),
            "text_similarity": round(self.text_similarity, 1),
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "extra_skills": self.extra_skills,
            "jd_skills": self.jd_skills,
            "suggestions": self.suggestions,
        }


def _category_sort_key(skill: str) -> tuple[str, str]:
    return (category_of(skill), skill.lower())


def analyze(resume_text: str, jd_text: str, max_suggestions: int = 6) -> MatchReport:
    """Compare a resume against a job description and return a MatchReport."""
    resume_skills = detect_skills(resume_text)
    jd_skills = detect_skills(jd_text)

    matched = sorted(jd_skills & resume_skills, key=_category_sort_key)
    missing = sorted(jd_skills - resume_skills, key=_category_sort_key)
    extra = sorted(resume_skills - jd_skills, key=_category_sort_key)

    # Skill coverage: fraction of JD-required skills present in the resume.
    coverage = (len(matched) / len(jd_skills) * 100.0) if jd_skills else 0.0

    # Text overlap via from-scratch TF-IDF cosine similarity.
    sim = text_similarity(resume_text, jd_text) * 100.0

    overall = SKILL_WEIGHT * coverage + TEXT_WEIGHT * sim

    suggestions = _build_suggestions(missing, max_suggestions)

    return MatchReport(
        score=overall,
        skill_coverage=coverage,
        text_similarity=sim,
        matched_skills=matched,
        missing_skills=missing,
        extra_skills=extra,
        jd_skills=sorted(jd_skills, key=_category_sort_key),
        suggestions=suggestions,
    )


def _build_suggestions(missing: list[str], limit: int) -> list[str]:
    """Turn the most important missing skills into actionable advice.

    Technical skills (languages, ML, frameworks, infra) are prioritised over
    soft skills, since those are the ones ATS filters and recruiters screen on.
    """
    priority = {"language": 0, "ml_ai": 0, "framework": 1, "data": 1,
                "infra": 1, "cs": 2, "process": 3, "soft": 4, "other": 3}
    ranked = sorted(missing, key=lambda s: (priority.get(category_of(s), 3), s.lower()))

    out: list[str] = []
    for skill in ranked[:limit]:
        template = _BULLET_TEMPLATES.get(
            skill,
            f"The JD emphasises {skill} but it's absent from your resume — add a "
            f"project, course, or bullet that demonstrates {skill}.",
        )
        out.append(f"[{skill}] {template}")
    return out


def grade(score: float) -> str:
    """Human-friendly band for a 0-100 score."""
    if score >= 80:
        return "Strong match"
    if score >= 60:
        return "Good match — a few gaps to close"
    if score >= 40:
        return "Partial match — meaningful gaps"
    return "Weak match — significant tailoring needed"


def format_report(report: MatchReport) -> str:
    """Render a MatchReport as a readable plain-text block."""
    lines: list[str] = []
    bar = "=" * 60
    lines.append(bar)
    lines.append("  RESUME ↔ JOB DESCRIPTION  —  MATCH REPORT")
    lines.append(bar)
    lines.append("")
    lines.append(f"  Overall match score : {report.score:5.1f} / 100   ({grade(report.score)})")
    lines.append(f"  Skill coverage      : {report.skill_coverage:5.1f} / 100"
                 f"   ({len(report.matched_skills)}/{len(report.jd_skills)} JD skills present)")
    lines.append(f"  Text similarity     : {report.text_similarity:5.1f} / 100   (TF-IDF cosine)")
    lines.append("")

    lines.append(f"  ✓ MATCHED SKILLS ({len(report.matched_skills)})")
    lines.extend(_grouped_lines(report.matched_skills) or ["      (none)"])
    lines.append("")

    lines.append(f"  ✗ MISSING — in the JD, not in your resume ({len(report.missing_skills)})")
    lines.extend(_grouped_lines(report.missing_skills) or ["      (none — full coverage!)"])
    lines.append("")

    if report.extra_skills:
        lines.append(f"  + EXTRA — on your resume, not required ({len(report.extra_skills)})")
        lines.extend(_grouped_lines(report.extra_skills))
        lines.append("")

    if report.suggestions:
        lines.append("  ➜ SUGGESTIONS (highest-impact gaps first)")
        for s in report.suggestions:
            lines.append(f"      • {s}")
        lines.append("")

    lines.append(bar)
    return "\n".join(lines)


def _grouped_lines(skills: list[str]) -> list[str]:
    """Group skills by category for tidy printing."""
    if not skills:
        return []
    buckets: dict[str, list[str]] = {}
    for s in skills:
        buckets.setdefault(category_of(s), []).append(s)
    lines: list[str] = []
    for cat in sorted(buckets):
        label = CATEGORY_LABELS.get(cat, "Other")
        lines.append(f"      {label}: {', '.join(sorted(buckets[cat]))}")
    return lines
