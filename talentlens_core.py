"""Core logic for TalentLens: skill extraction, scoring, and resume category prediction."""
from __future__ import annotations
import re
import unicodedata
from pathlib import Path

SKILL_GROUPS = {
    "Programming": ["python", "java", "c++", "c#", "javascript", "typescript", "golang", "rust", "kotlin", "swift", "php", "ruby", "matlab", "scala", "bash", "powershell", "sql", "html", "css"],
    "Web & App Development": ["react", "next.js", "node.js", "express", "angular", "vue.js", "django", "flask", "fastapi", "spring boot", "asp.net", ".net", "android", "flutter", "react native", "rest api", "graphql", "microservices", "api development"],
    "Data & AI": ["machine learning", "deep learning", "artificial intelligence", "natural language processing", "computer vision", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "opencv", "data analysis", "data visualization", "statistics", "power bi", "tableau", "excel", "generative ai", "large language models", "llm", "data science", "data mining", "predictive modeling", "etl", "data warehousing", "informatica", "spark", "pyspark", "hadoop", "airflow", "dbt"],
    "Cloud & DevOps": ["aws", "amazon web services", "azure", "microsoft azure", "gcp", "google cloud", "docker", "kubernetes", "git", "github", "gitlab", "ci/cd", "linux", "terraform", "jenkins", "ansible", "cloud computing", "devops", "site reliability engineering", "monitoring", "prometheus", "grafana"],
    "Databases": ["mysql", "postgresql", "mongodb", "sqlite", "redis", "oracle database", "firebase", "microsoft sql server", "nosql", "database design", "query optimization", "snowflake", "bigquery", "redshift"],
    "Finance & Accounting": ["financial analysis", "financial modeling", "accounting", "bookkeeping", "accounts payable", "accounts receivable", "taxation", "auditing", "budgeting", "forecasting", "variance analysis", "quickbooks", "tally", "sap fico", "financial reporting", "risk analysis", "valuation", "investment analysis", "compliance", "ifrs", "gaap"],
    "Business & Management": ["project management", "product management", "business analysis", "business strategy", "stakeholder management", "requirements gathering", "agile", "scrum", "kanban", "jira", "salesforce", "crm", "sales", "marketing", "digital marketing", "seo", "sem", "content marketing", "market research", "customer relationship management", "operations management", "supply chain", "procurement", "logistics", "inventory management", "human resources", "recruitment", "talent acquisition", "payroll", "performance management", "employee relations"],
    "Design & Creative": ["figma", "adobe photoshop", "photoshop", "illustrator", "adobe illustrator", "indesign", "premiere pro", "after effects", "blender", "autocad", "solidworks", "revit", "ui design", "ux design", "user research", "wireframing", "prototyping", "graphic design", "video editing", "content writing", "copywriting", "technical writing", "storytelling", "typography"],
    "Healthcare & Life Sciences": ["patient care", "clinical research", "clinical documentation", "medical coding", "medical terminology", "nursing", "phlebotomy", "first aid", "cpr", "hipaa", "electronic health records", "ehr", "laboratory techniques", "quality control", "pharmacology", "public health", "infection control", "patient assessment", "healthcare administration"],
    "Education & Research": ["curriculum development", "lesson planning", "classroom management", "instructional design", "assessment design", "student counseling", "academic writing", "research methods", "qualitative research", "quantitative research", "literature review", "data collection", "teaching", "tutoring"],
    "Engineering & Manufacturing": ["autocad", "solidworks", "catia", "ansys", "matlab", "six sigma", "lean manufacturing", "quality assurance", "quality control", "root cause analysis", "plc", "scada", "electrical design", "mechanical design", "civil engineering", "structural analysis", "project planning", "safety compliance", "process improvement", "preventive maintenance", "hvac", "reliability engineering"],
    "Professional": ["communication", "leadership", "teamwork", "problem solving", "critical thinking", "time management", "public speaking", "presentation skills", "research", "adaptability", "attention to detail", "customer service", "negotiation", "conflict resolution", "interpersonal skills", "decision making", "written communication", "verbal communication"],
}

# Common variants normalize to one canonical label. Keys are lower-case.
SKILL_ALIASES = {
    "js": "javascript", "javascript": "javascript", "ts": "typescript",
    "sklearn": "scikit-learn", "scikit learn": "scikit-learn", "scikit-learn": "scikit-learn",
    "powerbi": "power bi", "power-bi": "power bi", "postgres": "postgresql", "postgre sql": "postgresql",
    "nodejs": "node.js", "node js": "node.js", "reactjs": "react", "react.js": "react",
    "nextjs": "next.js", "next js": "next.js", "ci cd": "ci/cd", "ci-cd": "ci/cd",
    "nlp": "natural language processing", "ml": "machine learning", "ai": "artificial intelligence",
    "aws cloud": "aws", "amazon web services": "aws", "microsoft azure": "azure",
    "google cloud platform": "gcp", "google cloud": "gcp", "ms excel": "excel",
    "ms sql server": "microsoft sql server", "sql server": "microsoft sql server",
    "customer relationship management": "crm", "talent acquisition": "recruitment",
    "user experience design": "ux design", "user interface design": "ui design",
    "power point": "powerpoint", "ms powerpoint": "powerpoint", "project coordination": "project management",
    "natural language processing (nlp)": "natural language processing", "large language model": "large language models",
}

# Do not include extremely ambiguous single-letter skills as normal matches.
AMBIGUOUS_SKILLS = {"c", "r"}
ALL_SKILLS = sorted({s for group in SKILL_GROUPS.values() for s in group} | set(SKILL_ALIASES.values()), key=len, reverse=True)

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = text.replace("–", "-").replace("—", "-").replace("‑", "-").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()

def canonical_skill(skill: str) -> str:
    key = normalize(skill).strip(" .,:;|()[]{}")
    return SKILL_ALIASES.get(key, key)

def _contains_phrase(text: str, phrase: str) -> bool:
    # Flexible whitespace; boundaries prevent 'java' matching 'javascript'.
    p = re.escape(phrase).replace(r"\ ", r"\s+")
    return bool(re.search(r"(?<![\w+#])" + p + r"(?![\w+#])", text, flags=re.I))

def vocabulary_skills(text: str) -> list[str]:
    t = normalize(text)
    if not t:
        return []
    found = set()
    for skill in ALL_SKILLS:
        # Avoid false positives for abbreviations that are ordinary words.
        variants = {skill}
        variants.update(alias for alias, canonical in SKILL_ALIASES.items() if canonical == skill)
        if skill in AMBIGUOUS_SKILLS:
            continue
        for variant in variants:
            if _contains_phrase(t, variant):
                found.add(skill)
                break
    # Single-letter language detection only when context is explicit.
    if re.search(r"\b(?:programming in|written in|language)\s+c\b|\bc\s+(?:programming|language)\b", t):
        found.add("c")
    if re.search(r"\b(?:programming in|written in|language)\s+r\b|\br\s+(?:programming|language)\b", t):
        found.add("r")
    return sorted(found)

def find_skills(text: str) -> list[str]:
    return vocabulary_skills(text)

def find_target_skills(text: str) -> list[str]:
    return vocabulary_skills(text)

def category_for(skill: str) -> str:
    key = canonical_skill(skill)
    for cat, skills in SKILL_GROUPS.items():
        if key in skills:
            return cat
    return "Other"

def score_resume(resume_skills: list[str], target_skills: list[str]) -> int | None:
    target = {canonical_skill(s) for s in target_skills}
    if not target:
        return None
    resume = {canonical_skill(s) for s in resume_skills}
    return round(100 * len(resume & target) / len(target))

def predict_resume_category(text: str, model=None):
    """Predict one of the dataset's nine categories; returns None if model unavailable."""
    if model is None or not str(text).strip():
        return None
    # Training uses the first 6,000 characters, so inference mirrors training.
    return str(model.predict([str(text)[:6000]])[0])
