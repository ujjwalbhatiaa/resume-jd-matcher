"""
skills_db.py
============
A curated taxonomy of technical and professional skills used to drive the
gap analysis. Every skill has a *canonical* name plus a list of surface
forms (aliases / abbreviations / spellings) that should all map back to it.

Keeping this as plain data (no dependencies) makes the matcher transparent and
easy to extend: add a row and the analyzer immediately understands the new
skill, its aliases, and its category.

The matcher is alias-aware, so a job description that says "JS" and a resume
that says "JavaScript" are correctly recognised as the *same* skill.
"""

from __future__ import annotations

# Each entry: canonical -> (category, [aliases/surface-forms])
# Aliases are matched case-insensitively as whole tokens or token phrases.
SKILLS: dict[str, tuple[str, list[str]]] = {
    # ---- Languages -----------------------------------------------------
    "Python":            ("language",  ["python", "py", "python3"]),
    "Java":              ("language",  ["java"]),
    "JavaScript":        ("language",  ["javascript", "js", "ecmascript"]),
    "TypeScript":        ("language",  ["typescript", "ts"]),
    "C":                 ("language",  ["c language", "c programming"]),
    "C++":               ("language",  ["c++", "cpp", "cplusplus"]),
    "C#":                ("language",  ["c#", "csharp", "c sharp"]),
    "Go":                ("language",  ["golang", "go lang"]),
    "Rust":              ("language",  ["rust"]),
    "Kotlin":            ("language",  ["kotlin"]),
    "Swift":             ("language",  ["swift"]),
    "R":                 ("language",  ["r language", "rlang"]),
    "SQL":               ("language",  ["sql"]),
    "Bash":              ("language",  ["bash", "shell scripting", "shell"]),
    "MATLAB":            ("language",  ["matlab"]),
    "Scala":             ("language",  ["scala"]),

    # ---- ML / AI -------------------------------------------------------
    "Machine Learning":  ("ml_ai",     ["machine learning", "ml", "supervised learning", "unsupervised learning"]),
    "Deep Learning":     ("ml_ai",     ["deep learning", "dl", "neural networks", "neural network"]),
    "NLP":               ("ml_ai",     ["nlp", "natural language processing", "text mining"]),
    "Computer Vision":   ("ml_ai",     ["computer vision", "cv", "image processing", "image recognition"]),
    "Reinforcement Learning": ("ml_ai", ["reinforcement learning", "rl"]),
    "Data Science":      ("ml_ai",     ["data science", "data scientist"]),
    "Data Analysis":     ("ml_ai",     ["data analysis", "data analytics", "exploratory data analysis", "eda"]),
    "Statistics":        ("ml_ai",     ["statistics", "statistical analysis", "statistical modeling"]),
    "Feature Engineering": ("ml_ai",   ["feature engineering", "feature extraction"]),
    "Model Deployment":  ("ml_ai",     ["model deployment", "mlops", "ml ops"]),

    # ---- ML / data frameworks -----------------------------------------
    "PyTorch":           ("framework", ["pytorch", "torch"]),
    "TensorFlow":        ("framework", ["tensorflow", "tf", "keras"]),
    "scikit-learn":      ("framework", ["scikit-learn", "scikit learn", "sklearn"]),
    "pandas":            ("framework", ["pandas"]),
    "NumPy":             ("framework", ["numpy"]),
    "Matplotlib":        ("framework", ["matplotlib", "seaborn", "plotly"]),
    "Hugging Face":      ("framework", ["hugging face", "huggingface", "transformers library"]),
    "OpenCV":            ("framework", ["opencv", "cv2"]),
    "Spark":             ("framework", ["spark", "apache spark", "pyspark"]),

    # ---- Web / app frameworks -----------------------------------------
    "React":             ("framework", ["react", "react.js", "reactjs"]),
    "Node.js":           ("framework", ["node", "node.js", "nodejs"]),
    "Django":            ("framework", ["django"]),
    "Flask":             ("framework", ["flask"]),
    "FastAPI":           ("framework", ["fastapi", "fast api"]),
    "Express":           ("framework", ["express", "express.js", "expressjs"]),
    "Spring":            ("framework", ["spring", "spring boot", "springboot"]),
    "HTML/CSS":          ("framework", ["html", "css", "html5", "css3", "tailwind", "bootstrap"]),

    # ---- Data / infra --------------------------------------------------
    "PostgreSQL":        ("data",      ["postgresql", "postgres", "psql"]),
    "MySQL":             ("data",      ["mysql"]),
    "MongoDB":           ("data",      ["mongodb", "mongo"]),
    "Redis":             ("data",      ["redis"]),
    "Docker":            ("infra",     ["docker", "containerization", "containers"]),
    "Kubernetes":        ("infra",     ["kubernetes", "k8s"]),
    "AWS":               ("infra",     ["aws", "amazon web services", "ec2", "s3", "lambda"]),
    "GCP":               ("infra",     ["gcp", "google cloud", "google cloud platform"]),
    "Azure":             ("infra",     ["azure", "microsoft azure"]),
    "Git":               ("infra",     ["git", "version control", "github", "gitlab"]),
    "CI/CD":             ("infra",     ["ci/cd", "cicd", "continuous integration", "continuous deployment"]),
    "Linux":             ("infra",     ["linux", "unix"]),
    "REST API":          ("infra",     ["rest", "rest api", "restful", "api design", "apis"]),
    "GraphQL":           ("infra",     ["graphql"]),

    # ---- CS fundamentals ----------------------------------------------
    "Data Structures":   ("cs",        ["data structures", "data structure"]),
    "Algorithms":        ("cs",        ["algorithms", "algorithm design", "dsa"]),
    "Object-Oriented Programming": ("cs", ["object-oriented programming", "oop", "object oriented"]),
    "Operating Systems": ("cs",        ["operating systems", "os"]),
    "Distributed Systems": ("cs",      ["distributed systems", "distributed computing"]),
    "Testing":           ("cs",        ["testing", "unit testing", "unit tests", "test-driven", "tdd", "pytest"]),
    "Agile":             ("process",   ["agile", "scrum", "kanban"]),

    # ---- Soft / professional ------------------------------------------
    "Communication":     ("soft",      ["communication", "communication skills", "written communication"]),
    "Teamwork":          ("soft",      ["teamwork", "collaboration", "cross-functional", "collaborate"]),
    "Problem Solving":   ("soft",      ["problem solving", "problem-solving", "analytical skills", "critical thinking"]),
    "Leadership":        ("soft",      ["leadership", "team lead", "mentoring", "mentorship"]),
}

# Pretty labels for the category buckets used in the report.
CATEGORY_LABELS: dict[str, str] = {
    "language":  "Programming Languages",
    "ml_ai":     "ML / AI",
    "framework":  "Frameworks & Libraries",
    "data":      "Databases",
    "infra":     "Tools & Infrastructure",
    "cs":        "CS Fundamentals",
    "process":   "Process",
    "soft":      "Professional Skills",
}


def build_alias_index() -> dict[str, str]:
    """Return a mapping {alias_lowercased: canonical_skill}.

    Longer aliases are intentionally kept; the analyzer matches multi-word
    phrases first so "natural language processing" wins over a stray "nlp".
    """
    index: dict[str, str] = {}
    for canonical, (_category, aliases) in SKILLS.items():
        for alias in aliases:
            index[alias.lower()] = canonical
    return index


def category_of(skill: str) -> str:
    """Category key for a canonical skill (defaults to 'other')."""
    entry = SKILLS.get(skill)
    return entry[0] if entry else "other"
