import streamlit as st
from pathlib import Path
import re, io, math
from collections import Counter

# AI/NLP model: resume-specific BERT NER
try:
    from transformers import pipeline
except Exception:
    pipeline = None

# Optional document readers
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    from docx import Document
except Exception:
    Document = None

st.set_page_config(
    page_title="TalentLens | Resume Skill Analyzer",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- DATA -----------------------------
SKILL_GROUPS = {
    "Programming": [
        "python","java","c","c++","c#","javascript","typescript","go","rust","kotlin","swift","php","ruby","r","matlab","scala"
    ],
    "Web & App": [
        "html","css","react","next.js","node.js","express","angular","vue","django","flask","fastapi","spring","android","flutter"
    ],
    "Data & AI": [
        "machine learning","deep learning","artificial intelligence","natural language processing","nlp",
        "computer vision","pandas","numpy","scikit-learn","tensorflow","pytorch","keras","opencv",
        "data analysis","data visualization","statistics","sql","power bi","tableau","excel"
    ],
    "Cloud & DevOps": [
        "aws","azure","gcp","docker","kubernetes","git","github","gitlab","ci/cd","linux","terraform","jenkins"
    ],
    "Databases": [
        "mysql","postgresql","mongodb","sqlite","redis","oracle","firebase","sql"
    ],
    "Business": [
        "project management","product management","marketing","sales","market research",
        "financial analysis","business analysis","operations","strategy"
    ],
    "Professional": [
        "communication","leadership","teamwork","problem solving","critical thinking",
        "time management","public speaking","presentation","research","adaptability"
    ],
}

ALL_SKILLS = sorted(set(s for group in SKILL_GROUPS.values() for s in group), key=len, reverse=True)

def normalize(text):
    text = text.lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text

def extract_text(uploaded):
    if uploaded is None:
        return ""
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        if not pdfplumber:
            raise RuntimeError("Install pdfplumber to read PDF files.")
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    if name.endswith(".docx"):
        if not Document:
            raise RuntimeError("Install python-docx to read DOCX files.")
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("Supported formats: PDF, DOCX, TXT")

@st.cache_resource(show_spinner=False)
def load_resume_ner():
    """Load the resume-specific BERT NER model once per app session."""
    if pipeline is None:
        return None
    try:
        return pipeline(
            "token-classification",
            model="yashpwr/resume-ner-bert",
            aggregation_strategy="simple",
        )
    except Exception:
        return None

def ner_extract(text):
    """Extract resume entities with the fine-tuned BERT NER model."""
    ner = load_resume_ner()
    if ner is None or not text.strip():
        return []

    # BERT tokenizers have practical input limits. Process text in chunks
    # while keeping paragraphs reasonably intact.
    chunks = []
    current = []
    current_len = 0
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if current_len + len(paragraph) > 3000 and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph)

    if current:
        chunks.append("\n".join(current))

    entities = []
    for chunk in chunks:
        try:
            entities.extend(ner(chunk))
        except Exception:
            continue
    return entities

def extract_skills_with_ner(text):
    """Return normalized SKILLS entities detected by the resume NER model."""
    entities = ner_extract(text)
    skills = []
    for entity in entities:
        label = entity.get("entity_group", entity.get("label", ""))
        if label.upper() == "SKILLS":
            value = entity.get("word", "").strip()
            value = re.sub(r"\s+", " ", value)
            value = value.strip(" ,;|")
            if value:
                skills.append(value)
    return sorted(set(skills), key=str.lower)

def find_skills(text):
    """NER-first skill extraction with vocabulary fallback.

    The fallback makes the app usable if the Hugging Face model cannot be
    downloaded, while the NER results remain the primary AI extraction path.
    """
    ner_skills = extract_skills_with_ner(text)
    if ner_skills:
        return ner_skills
    return sorted(set(s for s in ALL_SKILLS if re.search(
        r"(?<!\w)" + re.escape(s.lower()) + r"(?!\w)", normalize(text)
    )))

def find_target_skills(text):
    """Job-description skills: NER first, then vocabulary matching."""
    return find_skills(text)

def category_for(skill):
    for cat, skills in SKILL_GROUPS.items():
        if skill in skills:
            return cat
    return "Other"

def score_resume(resume_skills, target_skills):
    if not target_skills:
        return 0
    matched = set(resume_skills) & set(target_skills)
    return round(100 * len(matched) / len(set(target_skills)))

def estimate_ats(text):
    t = normalize(text)
    checks = {
        "Contact information": bool(re.search(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", t)),
        "Phone number": bool(re.search(r"\b(?:\+?\d[\d ()-]{8,}\d)\b", t)),
        "Education section": bool(re.search(r"\b(education|academic|degree|bachelor|master|b\.?tech|b\.?sc|mba)\b", t)),
        "Experience section": bool(re.search(r"\b(experience|employment|internship|work history)\b", t)),
        "Projects section": bool(re.search(r"\b(projects?|portfolio)\b", t)),
        "Skills section": bool(re.search(r"\b(skills?|technical skills|competencies)\b", t)),
        "Action verbs": len(re.findall(r"\b(led|built|developed|created|designed|improved|analyzed|managed|implemented|optimized|automated|achieved)\b", t)) >= 3,
    }
    return checks

# ----------------------------- UI -----------------------------
st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
    .hero {
        padding: 2rem 2.2rem; border-radius: 24px;
        background: linear-gradient(135deg, #111827 0%, #1f2937 50%, #312e81 100%);
        color: white; margin-bottom: 1.2rem;
        border: 1px solid rgba(255,255,255,.12);
    }
    .hero h1 {font-size: 3rem; margin: 0; letter-spacing: -1.5px;}
    .hero p {font-size: 1.08rem; color: #d1d5db; margin: .5rem 0 0;}
    .chip {
        display:inline-block; padding: .35rem .7rem; margin:.2rem;
        border-radius: 999px; background:#eef2ff; color:#3730a3;
        border:1px solid #c7d2fe; font-size:.86rem;
    }
    .missing {background:#fff1f2; color:#be123c; border-color:#fecdd3;}
    .matched {background:#ecfdf5; color:#047857; border-color:#a7f3d0;}
    .section-card {
        padding: 1rem 1.2rem; border:1px solid #e5e7eb;
        border-radius:18px; background:white; height:100%;
    }
    .small {color:#6b7280; font-size:.9rem;}
    .big-score {font-size:3.2rem; font-weight:800; line-height:1;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>✦ TalentLens</h1>
<p>AI-powered resume skill intelligence. Extract skills, compare them to a target role, and turn gaps into an action plan.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Analysis controls")
    uploaded = st.file_uploader("Upload your resume", type=["pdf","docx","txt"])
    jd = st.text_area(
        "Paste a target job description",
        height=220,
        placeholder="Example: We are looking for a Python developer with SQL, Git, Docker and machine learning experience..."
    )
    demo = st.button("Load polished demo data", use_container_width=True)
    st.divider()
    st.caption("Privacy-first demo: this version processes files locally and does not send resumes to an external API.")

if demo:
    resume_text = """
    ALEX SHARMA
    alex@example.com | +91 98765 43210

    SUMMARY
    Computer science student passionate about software engineering and data science.

    EDUCATION
    B.Tech Computer Science, 2026

    EXPERIENCE
    Software Engineering Intern
    Built Python automation tools and REST APIs. Worked with Git and Linux.
    Improved a data processing pipeline and collaborated with a team.

    PROJECTS
    Customer Churn Predictor using Python, Pandas, NumPy and Scikit-learn.
    Dashboard built with SQL and Power BI.

    SKILLS
    Python, JavaScript, SQL, Pandas, NumPy, Scikit-learn, Git, Linux, Docker,
    Machine Learning, Data Analysis, Communication, Teamwork, Problem Solving
    """
    jd = """
    We are hiring a Junior Machine Learning Engineer.
    Requirements: Python, SQL, machine learning, scikit-learn, pandas, numpy,
    statistics, Git, Docker, data analysis, communication and problem solving.
    Experience with TensorFlow or PyTorch is a plus.
    """
elif uploaded:
    try:
        resume_text = extract_text(uploaded)
    except Exception as e:
        st.error(str(e))
        resume_text = ""
else:
    resume_text = ""

if not resume_text:
    st.info("Upload a resume or use the demo data from the sidebar to start the analysis.")
    st.markdown("### What TalentLens does")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("1", "Extract")
    c2.metric("2", "Compare")
    c3.metric("3", "Score")
    c4.metric("4", "Recommend")
    st.stop()

resume_skills = find_skills(resume_text)
jd_skills = find_target_skills(jd)
matched = sorted(set(resume_skills) & set(jd_skills))
missing = sorted(set(jd_skills) - set(resume_skills))
score = score_resume(resume_skills, jd_skills)
ats = estimate_ats(resume_text)
ner_available = load_resume_ner() is not None

# Header metrics
if ner_available:
    st.success("AI model active: resume-specific BERT NER is extracting entities.")
else:
    st.warning("BERT NER model unavailable. Using the built-in skill vocabulary fallback.")

st.subheader("Executive summary")
m1,m2,m3,m4 = st.columns(4)
m1.metric("Skill match", f"{score}%")
m2.metric("Skills detected", len(resume_skills))
m3.metric("Target skills", len(set(jd_skills)))
m4.metric("Missing skills", len(missing))

st.progress(score/100)

# Skill chips
left,right = st.columns(2)
with left:
    st.markdown("### ✓ Matched skills")
    if matched:
        st.markdown("".join(f'<span class="chip matched">{s}</span>' for s in matched), unsafe_allow_html=True)
    else:
        st.caption("No direct skill matches detected.")
with right:
    st.markdown("### + Priority gaps")
    if missing:
        st.markdown("".join(f'<span class="chip missing">{s}</span>' for s in missing), unsafe_allow_html=True)
    else:
        st.success("No target skill gaps detected.")

st.divider()

# Category breakdown
st.subheader("Skill landscape")
categories = sorted(set(category_for(s) for s in resume_skills) | set(category_for(s) for s in jd_skills))
for cat in categories:
    have = sorted(set(s for s in resume_skills if category_for(s)==cat))
    need = sorted(set(s for s in jd_skills if category_for(s)==cat))
    if not need:
        continue
    cat_score = round(100 * len(set(have)&set(need))/len(set(need)))
    c1,c2,c3 = st.columns([2,5,1])
    c1.write(f"**{cat}**")
    c2.progress(cat_score/100)
    c3.write(f"**{cat_score}%**")
    st.caption(f"Matched: {', '.join(sorted(set(have)&set(need))) or 'None'}  •  Target: {', '.join(need)}")

st.divider()

# ATS + recommendations
a,b = st.columns(2)
with a:
    st.subheader("ATS readiness")
    passed = sum(ats.values())
    ats_score = round(100*passed/len(ats))
    st.metric("Structure score", f"{ats_score}%")
    for label, ok in ats.items():
        st.write(("✅ " if ok else "⚠️ ") + label)
with b:
    st.subheader("Action plan")
    if missing:
        st.write("**1. Close the highest-impact gaps**")
        st.write(", ".join(missing[:5]) + ("..." if len(missing)>5 else ""))
    st.write("**2. Strengthen evidence**")
    st.write("For each important skill, show a project, metric, result, or concrete technology used.")
    st.write("**3. Improve ATS structure**")
    failed = [k for k,v in ats.items() if not v]
    st.write("Address: " + (", ".join(failed) if failed else "No major structural gaps detected."))

st.divider()

with st.expander("🤖 AI model details"):
    st.markdown("""
    **Model:** `yashpwr/resume-ner-bert`

    This is a BERT-based token-classification model fine-tuned for resume/CV information extraction. It can identify entities including **SKILLS, EDUCATION, EXPERIENCE, ORG, PERSON, LOCATION, EMAIL and PHONE**.

    The model is downloaded from Hugging Face the first time the app runs and is cached locally for subsequent runs.

    **Important:** NER identifies entities; it does not prove that a candidate is proficient in a skill. TalentLens therefore uses NER as the extraction layer and keeps the final matching score transparent.
    """)

with st.expander("🔬 Methodology — useful for the CBSE viva"):
    st.markdown("""
    **1. Text extraction:** PDF, DOCX or TXT resume content is converted into plain text.

    **2. Preprocessing:** Text is lowercased and normalized so variations in formatting do not affect matching.

    **3. Skill extraction:** A curated skill vocabulary is scanned using word-boundary matching.

    **4. Job matching:** Resume skills are compared with skills detected in the target job description.

    **5. Skill-match score:**

    `Score = (Number of matched target skills / Number of target skills) × 100`

    **6. ATS readiness:** Simple NLP/rule-based checks look for contact details, education, experience, projects, skills and action-oriented language.

    This project deliberately separates **AI/NLP analysis** from ordinary interface code, making the pipeline easier to explain, test and improve.
    """)

with st.expander("📄 Resume text preview"):
    st.text_area("Extracted text", resume_text[:12000], height=300, label_visibility="collapsed")
