import streamlit as st
from pathlib import Path
import re, io

try:
    import joblib
except Exception:
    joblib = None

from talentlens_core import (SKILL_GROUPS, ALL_SKILLS, SKILL_ALIASES, normalize, canonical_skill,
    vocabulary_skills, find_skills, find_target_skills, category_for, score_resume, predict_resume_category)

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from docx import Document
except Exception:
    Document = None

st.set_page_config(
    page_title="TalentLens | Resume Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- TEXT EXTRACTION -----------------------------

def extract_text(uploaded):
    """Extract text from TXT, DOCX, or PDF, including common template sections."""
    if uploaded is None:
        return ""
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".txt"):
        return data.decode("utf-8-sig", errors="ignore")
    if name.endswith(".pdf"):
        chunks = []
        if pdfplumber:
            try:
                with pdfplumber.open(io.BytesIO(data)) as pdf:
                    chunks = [page.extract_text(layout=False) or "" for page in pdf.pages]
            except Exception:
                chunks = []
        text = "\n".join(chunks).strip()
        if len(text) < 30:
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(data))
                text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            except Exception:
                pass
        return text
    if name.endswith(".docx"):
        if not Document:
            raise RuntimeError("Install python-docx to read DOCX files.")
        doc = Document(io.BytesIO(data))
        chunks = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    chunks.append(" | ".join(cells))
        for section in doc.sections:
            for part in (section.header, section.footer):
                chunks.extend(p.text for p in part.paragraphs if p.text.strip())
        return "\n".join(chunks)
    raise ValueError("Supported formats: PDF, DOCX, TXT")


def estimate_ats(text):
    t = normalize(text)
    return {
        "Contact information": bool(re.search(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", t)),
        "Phone number": bool(re.search(r"\b(?:\+?\d[\d ()-]{8,}\d)\b", t)),
        "Education section": bool(re.search(r"\b(education|academic|degree|bachelor|master|b\.?tech|b\.?sc|mba)\b", t)),
        "Experience section": bool(re.search(r"\b(experience|employment|internship|work history)\b", t)),
        "Projects section": bool(re.search(r"\b(projects?|portfolio)\b", t)),
        "Skills section": bool(re.search(r"\b(skills?|technical skills|competencies)\b", t)),
        "Action verbs": len(re.findall(r"\b(led|built|developed|created|designed|improved|analyzed|managed|implemented|optimized|automated|achieved)\b", t)) >= 3,
    }


@st.cache_resource(show_spinner=False)
def load_category_model():
    model_path = Path(__file__).with_name("resume_category_model.joblib")
    if joblib is None or not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None

# ----------------------------- POLISHED UI -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --ink: #111827;
    --muted: #667085;
    --line: #e7eaf0;
    --panel: #ffffff;
    --soft: #f7f8fc;
    --accent: #6d5dfc;
    --accent2: #8b5cf6;
    --green: #059669;
    --red: #e11d48;
}

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(109,93,252,.08), transparent 28rem),
        radial-gradient(circle at 100% 15%, rgba(139,92,246,.06), transparent 26rem),
        #fafbfc;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3,h4 { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
.block-container { max-width: 1440px; padding: 2.2rem 2.5rem 4rem; }

.hero {
    position: relative;
    overflow: hidden;
    padding: 2.4rem 2.6rem;
    border-radius: 28px;
    color: white;
    margin-bottom: 1.6rem;
    background: linear-gradient(120deg, #111827 0%, #24204d 48%, #5146a5 100%);
    box-shadow: 0 18px 45px rgba(31, 28, 72, .18);
}
.hero:after {
    content: "";
    position: absolute;
    width: 260px; height: 260px;
    right: -70px; top: -100px;
    border-radius: 50%;
    background: rgba(255,255,255,.08);
}
.eyebrow {
    font-size: .76rem; font-weight: 700; letter-spacing: .14em;
    text-transform: uppercase; color: #c4b5fd; margin-bottom: .65rem;
}
.hero h1 { color: white; font-size: 3.1rem; line-height: 1; margin: 0; letter-spacing: -.05em; }
.hero p { color: #d8d8e7; max-width: 760px; font-size: 1.02rem; margin: .75rem 0 0; line-height: 1.65; }
.badge {
    display:inline-flex; align-items:center; gap:.4rem;
    margin-top:1.2rem; padding:.42rem .72rem; border-radius:999px;
    background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.16);
    color:#eeeaff; font-size:.8rem; font-weight:600;
}

div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1rem 1.15rem;
    box-shadow: 0 5px 18px rgba(16,24,40,.035);
}
div[data-testid="stMetricLabel"] { color: var(--muted); font-size: .82rem; }
div[data-testid="stMetricValue"] { font-family:'Space Grotesk'; color:var(--ink); }

.panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 1.2rem 1.35rem;
    box-shadow: 0 6px 22px rgba(16,24,40,.035);
}
.panel-title { font-family:'Space Grotesk'; font-size:1.05rem; font-weight:700; color:var(--ink); }
.panel-sub { color:var(--muted); font-size:.86rem; margin-top:.2rem; }

.score-ring {
    width: 150px; height: 150px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    margin:.6rem auto 1rem;
    background: conic-gradient(var(--accent) calc(var(--score) * 1%), #ececf3 0);
    position:relative;
}
.score-ring:before {
    content:""; position:absolute; width:116px; height:116px;
    border-radius:50%; background:white;
}
.score-value { position:relative; z-index:1; font:700 2.2rem 'Space Grotesk'; color:var(--ink); }
.score-caption { text-align:center; color:var(--muted); font-size:.82rem; }

.chip {
    display:inline-block; padding:.38rem .68rem; margin:.22rem .18rem;
    border-radius:999px; font-size:.8rem; font-weight:600;
    border:1px solid;
}
.matched { background:#ecfdf5; color:#047857; border-color:#bbf7d0; }
.missing { background:#fff1f2; color:#be123c; border-color:#fecdd3; }
.neutral { background:#f5f3ff; color:#5b21b6; border-color:#ddd6fe; }

.step {
    display:flex; align-items:center; gap:.7rem; margin:.45rem 0;
    color:#475467; font-size:.88rem;
}
.step-num {
    width:25px; height:25px; display:inline-flex; align-items:center; justify-content:center;
    border-radius:8px; background:#f0edff; color:#5b4cf0; font-weight:700; font-size:.76rem;
}

div[data-testid="stFileUploader"] {
    background:#fff; border:1px dashed #c9cdd8; border-radius:16px; padding:.3rem;
}
[data-testid="stSidebar"] { background:#f6f7fb; border-right:1px solid var(--line); }
[data-testid="stSidebar"] h2 { font-family:'Space Grotesk'; }
.stButton > button {
    border-radius:12px; font-weight:700; border:1px solid #dedcf5;
}
.stProgress > div > div > div > div { border-radius:99px; }
div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:16px; background:white; }

.footer {
    text-align:center; color:#98a2b3; font-size:.78rem;
    padding:1.5rem 0 .5rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">Resume intelligence platform</div>
  <h1>✦ TalentLens</h1>
  <p>Turn a resume and a target job description into a clear, explainable view of skill fit, missing capabilities, and next steps.</p>
  <div class="badge">● NLP-assisted &nbsp;•&nbsp; Explainable scoring &nbsp;•&nbsp; Resume analysis</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Analysis workspace")
    st.caption("Feed TalentLens a resume and the role it should be evaluated against.")
    uploaded = st.file_uploader("Resume", type=["pdf","docx","txt"], help="PDF, DOCX and TXT are supported.")
    jd = st.text_area(
        "Target job description",
        height=230,
        placeholder="Paste the job description here…",
    )
    demo = st.button("✦ Load demo analysis", use_container_width=True)
    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("""
    <div class="step"><span class="step-num">1</span> Extract resume text</div>
    <div class="step"><span class="step-num">2</span> Detect skills and aliases</div>
    <div class="step"><span class="step-num">3</span> Compare against role</div>
    <div class="step"><span class="step-num">4</span> Generate recommendations</div>
    """, unsafe_allow_html=True)
    st.divider()
    st.caption("Privacy-first demo. Resume files are processed by this app rather than uploaded to a custom external API.")

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

if uploaded and not resume_text.strip():
    st.error("No selectable text could be extracted from this file. If this is a scanned/image-only PDF, it needs OCR before TalentLens can analyze it. Try a text-based PDF, DOCX, or TXT file.")
    st.stop()

if not resume_text:
    st.markdown('<div class="panel"><div class="panel-title">Start your first analysis</div><div class="panel-sub">Upload a resume from the sidebar or load the demo to explore the full dashboard.</div></div>', unsafe_allow_html=True)
    st.markdown("### How it works")
    c1,c2,c3,c4 = st.columns(4)
    cards = [
        ("01", "Extract", "Read resume content from PDF, DOCX or TXT."),
        ("02", "Understand", "Detect known skills and normalize common aliases."),
        ("03", "Compare", "Measure overlap with the target role."),
        ("04", "Improve", "Turn gaps and structure issues into an action plan."),
    ]
    for col, (num, title, desc) in zip((c1,c2,c3,c4), cards):
        with col:
            st.markdown(f'<div class="panel"><div style="color:#6d5dfc;font-weight:800">{num}</div><h3 style="margin:.35rem 0">{title}</h3><div class="panel-sub">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">TalentLens • CBSE Class XII AI Capstone</div>', unsafe_allow_html=True)
    st.stop()

resume_skills = find_skills(resume_text)
jd_skills = find_target_skills(jd)
matched = sorted(set(resume_skills) & set(jd_skills))
missing = sorted(set(jd_skills) - set(resume_skills))
score = score_resume(resume_skills, jd_skills)
ats = estimate_ats(resume_text)
category_model = load_category_model()
predicted_category = predict_resume_category(resume_text, category_model)

st.markdown(
    f'<div class="panel" style="margin-bottom:1rem"><div class="panel-title">Analysis ready</div>'
    f'<div class="panel-sub">{len(resume_text.split()):,} words scanned • {len(resume_skills)} skills detected • {len(jd_skills)} target skills identified</div></div>',
    unsafe_allow_html=True
)

st.info("Local skill extraction active • common aliases are normalized automatically. No external AI service or GPU download is required.")

st.markdown("### Executive overview")
m1,m2,m3,m4 = st.columns(4)
m1.metric("Skill match", f"{score}%" if score is not None else "N/A")
m2.metric("Skills detected", len(resume_skills))
m3.metric("Target skills", len(set(jd_skills)))
m4.metric("Priority gaps", len(missing))
if score is not None:
    st.progress(score / 100)
else:
    st.warning("No recognized skills were found in the job description. Paste a fuller description or enter a known skill such as Python, SQL, Excel, or nursing.")

if predicted_category:
    with st.expander("Resume category classifier", expanded=True):
        st.markdown(f"**Predicted category:** {predicted_category}")
        st.caption("This separate text classifier was trained on the supplied 9-category dataset. It is limited to those categories and is not a job-fit score.")
        if len(resume_text.split()) < 100:
            st.warning("This resume has very little extracted text, so the category prediction may be unreliable. Check the extracted-text panel below.")
else:
    st.caption("Resume category classifier unavailable: model file or scikit-learn dependencies are missing.")

left, center, right = st.columns([1.15, 1.5, 1.5])
with left:
    st.markdown(f'<div class="panel"><div class="panel-title">Overall fit</div><div class="score-ring" style="--score:{score or 0}"><span class="score-value">{str(score) + "%" if score is not None else "N/A"}</span></div><div class="score-caption">Direct skill overlap with the target role</div></div>', unsafe_allow_html=True)
with center:
    st.markdown('<div class="panel"><div class="panel-title">✓ Matched skills</div><div class="panel-sub">Skills detected in both the resume and target role</div><div style="margin-top:.7rem">', unsafe_allow_html=True)
    if matched:
        st.markdown("".join(f'<span class="chip matched">{s}</span>' for s in matched), unsafe_allow_html=True)
    else:
        st.caption("No direct matches detected.")
    st.markdown('</div></div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="panel"><div class="panel-title">+ Priority gaps</div><div class="panel-sub">Target skills not currently detected in the resume</div><div style="margin-top:.7rem">', unsafe_allow_html=True)
    if missing:
        st.markdown("".join(f'<span class="chip missing">{s}</span>' for s in missing), unsafe_allow_html=True)
    else:
        st.success("No target skill gaps detected.")
    st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown("### Skill landscape")
categories = sorted(set(category_for(s) for s in resume_skills) | set(category_for(s) for s in jd_skills))
for cat in categories:
    have = sorted(set(s for s in resume_skills if category_for(s) == cat))
    need = sorted(set(s for s in jd_skills if category_for(s) == cat))
    if not need:
        continue
    cat_score = round(100 * len(set(have) & set(need)) / len(set(need)))
    c1,c2,c3 = st.columns([1.5, 5.5, .8])
    c1.write(f"**{cat}**")
    c2.progress(cat_score / 100)
    c3.write(f"**{cat_score}%**")
    overlap = sorted(set(have) & set(need))
    st.caption(f"Matched: {', '.join(overlap) or 'None'}  •  Target: {', '.join(need)}")

st.markdown("### Resume readiness")
a,b = st.columns(2)
with a:
    passed = sum(ats.values())
    ats_score = round(100 * passed / len(ats))
    st.markdown(f'<div class="panel"><div class="panel-title">ATS structure</div><div class="panel-sub">Basic structural signals that help machine-readable resumes.</div><div style="margin-top:1rem">', unsafe_allow_html=True)
    st.metric("Structure score", f"{ats_score}%")
    for label, ok in ats.items():
        st.write(("✅ " if ok else "⚠️ ") + label)
    st.markdown('</div></div>', unsafe_allow_html=True)
with b:
    st.markdown('<div class="panel"><div class="panel-title">Action plan</div><div class="panel-sub">The highest-value changes suggested by this analysis.</div><div style="margin-top:1rem"></div>', unsafe_allow_html=True)
    if missing:
        st.write("**1. Close the highest-impact gaps**")
        st.write(", ".join(missing[:5]) + ("…" if len(missing) > 5 else ""))
    st.write("**2. Strengthen evidence**")
    st.write("For important skills, show a project, metric, result, or concrete technology used.")
    st.write("**3. Improve ATS structure**")
    failed = [k for k,v in ats.items() if not v]
    st.write("Address: " + (", ".join(failed) if failed else "No major structural gaps detected."))
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("🤖 AI model & scoring details"):
    st.markdown("""
    **Extraction:** Local vocabulary-based skill detection with common aliases. A separate TF-IDF + Linear SVM classifier predicts one of nine categories from the supplied dataset.

    Skill extraction uses boundary-aware phrase matching and canonical aliases. The category classifier is separate from the skill match score and predicts only among the nine labels present in its training dataset.

    **Matching score:** `(matched target skills / target skills) × 100`

    The score measures direct detected overlap. It does not claim that a candidate is proficient in a skill simply because the skill appears in text.
    """)

with st.expander("🔬 Methodology — CBSE viva ready"):
    st.markdown("""
    **1. Text extraction:** Resume content is extracted from PDF, DOCX or TXT.

    **2. Skill extraction:** A boundary-aware vocabulary matcher detects known skills and common aliases across several professional areas.

    **3. Normalization:** Common variants such as `sklearn` and `scikit learn` are mapped to a canonical skill name.

    **4. Job comparison:** Skills detected in the resume are compared with skills detected in the target job description.

    **5. Transparent scoring:** The app reports the percentage of target skills that were directly detected in the resume.

    **6. ATS checks:** Rule-based checks look for contact information, education, experience, projects, skills and action-oriented language.

    The system combines a supervised text classifier with transparent rule-based skill matching, making its decisions easier to explain and evaluate.
    """)

with st.expander("📄 Extracted resume text"):
    st.text_area("Extracted text", resume_text[:12000], height=320, label_visibility="collapsed")

st.markdown('<div class="footer">TalentLens • Resume Skill Intelligence • Built for the CBSE Class XII AI Capstone</div>', unsafe_allow_html=True)
