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
  color-scheme: dark;
  --bg: #080b12;
  --bg-soft: #0d1220;
  --panel: rgba(17, 24, 39, .88);
  --panel-strong: #111827;
  --panel-hover: #172033;
  --line: rgba(148, 163, 184, .18);
  --text: #edf2ff;
  --muted: #98a6c2;
  --accent: #a78bfa;
  --accent-2: #67e8f9;
  --green: #34d399;
  --red: #fb7185;
  --amber: #fbbf24;
}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
  font-family: 'DM Sans', Inter, ui-sans-serif, system-ui, sans-serif;
}
.stApp {
  color: var(--text);
  background:
    radial-gradient(ellipse at 8% -5%, rgba(124, 58, 237, .20), transparent 34rem),
    radial-gradient(ellipse at 92% 8%, rgba(6, 182, 212, .12), transparent 32rem),
    linear-gradient(180deg, #080b12 0%, #0a0f1b 50%, #080b12 100%);
}
[data-testid="stHeader"] {
  background: rgba(8, 11, 18, .78);
}
[data-testid="stToolbar"] { right: 1rem; }
.block-container { max-width: 1500px; padding: 1.6rem 2.2rem 3.5rem; }
h1, h2, h3, h4, p, label, span, li, div { scroll-margin-top: 2rem; }
h1, h2, h3, h4 {
  font-family: 'Space Grotesk', 'DM Sans', sans-serif;
  color: var(--text);
  letter-spacing: -.025em;
}
h2 { margin-top: 1.2rem; }
p, li, [data-testid="stMarkdownContainer"] { color: inherit; }
small, .muted, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

.hero {
  position: relative;
  isolation: isolate;
  overflow: hidden;
  padding: 2.5rem 2.7rem;
  border: 1px solid rgba(167, 139, 250, .25);
  border-radius: 28px;
  color: var(--text);
  margin: .35rem 0 1.7rem;
  background:
    radial-gradient(circle at 85% 20%, rgba(103, 232, 249, .16), transparent 16rem),
    radial-gradient(circle at 70% 110%, rgba(167, 139, 250, .20), transparent 22rem),
    linear-gradient(125deg, #111827 0%, #11152a 52%, #171b36 100%);
  box-shadow: 0 22px 65px rgba(0, 0, 0, .24), inset 0 1px 0 rgba(255,255,255,.04);
}
.hero:after {
  content: "";
  position: absolute;
  z-index: -1;
  width: 330px; height: 330px;
  right: -100px; top: -150px;
  border-radius: 50%;
  border: 1px solid rgba(103,232,249,.17);
  box-shadow: 0 0 0 35px rgba(103,232,249,.025), 0 0 0 75px rgba(167,139,250,.025);
}
.eyebrow {
  font-size: .72rem; font-weight: 800; letter-spacing: .17em;
  text-transform: uppercase; color: var(--accent-2); margin-bottom: .8rem;
}
.hero h1 {
  color: #f8fafc; font-size: clamp(2.5rem, 5vw, 4rem);
  line-height: .98; margin: 0; letter-spacing: -.065em;
}
.hero p { color: #c2cbe0; max-width: 760px; font-size: 1.02rem; margin: .95rem 0 0; line-height: 1.75; }
.badge {
  display:inline-flex; align-items:center; gap:.4rem;
  margin-top:1.25rem; padding:.45rem .8rem; border-radius:999px;
  background:rgba(167,139,250,.10); border:1px solid rgba(167,139,250,.23);
  color:#e9e3ff; font-size:.78rem; font-weight:700;
}

div[data-testid="stMetric"] {
  background: linear-gradient(145deg, rgba(20, 29, 47, .95), rgba(13, 18, 32, .95));
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1.05rem 1.15rem;
  box-shadow: 0 10px 28px rgba(0,0,0,.13);
  transition: transform .18s ease, border-color .18s ease;
}
div[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: rgba(167,139,250,.42); }
div[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .82rem; }
div[data-testid="stMetricValue"] { font-family:'Space Grotesk'; color:#f8fafc !important; }
div[data-testid="stMetricDelta"] { color: var(--accent-2) !important; }

.panel {
  background: linear-gradient(145deg, rgba(18, 26, 43, .92), rgba(12, 17, 30, .94));
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 1.25rem 1.35rem;
  box-shadow: 0 12px 34px rgba(0,0,0,.13);
  color: var(--text);
}
.panel-title { font-family:'Space Grotesk'; font-size:1.04rem; font-weight:700; color:#f3f6ff; }
.panel-sub { color:var(--muted); font-size:.85rem; margin-top:.25rem; line-height:1.55; }

.score-ring {
  width: 158px; height: 158px; border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  margin:.9rem auto 1rem;
  background: conic-gradient(var(--accent) calc(var(--score) * 1%), #273147 0);
  position:relative;
  box-shadow: 0 0 35px rgba(167,139,250,.13);
}
.score-ring:before {
  content:""; position:absolute; width:122px; height:122px;
  border-radius:50%; background:#111827; border:1px solid rgba(255,255,255,.06);
}
.score-value { position:relative; z-index:1; font:700 2.2rem 'Space Grotesk'; color:#f8fafc; }
.score-caption { text-align:center; color:var(--muted); font-size:.8rem; }

.chip {
  display:inline-block; padding:.38rem .68rem; margin:.22rem .18rem;
  border-radius:999px; font-size:.79rem; font-weight:650;
  border:1px solid;
}
.matched { background:rgba(16,185,129,.12); color:#6ee7b7; border-color:rgba(52,211,153,.28); }
.missing { background:rgba(244,63,94,.11); color:#fda4af; border-color:rgba(251,113,133,.28); }
.neutral { background:rgba(167,139,250,.12); color:#ddd6fe; border-color:rgba(167,139,250,.25); }

.step { display:flex; align-items:center; gap:.7rem; margin:.55rem 0; color:#bdc8dd; font-size:.87rem; }
.step-num {
  width:27px; height:27px; display:inline-flex; align-items:center; justify-content:center;
  border-radius:9px; background:rgba(167,139,250,.14); border:1px solid rgba(167,139,250,.18);
  color:#d8ccff; font-weight:800; font-size:.75rem;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0e1422 0%, #0a0f1a 100%);
  border-right:1px solid var(--line);
}
[data-testid="stSidebar"] > div { background: transparent; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color:#f3f6ff; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color:#cbd5e1; }
[data-testid="stFileUploader"] {
  background: rgba(12, 17, 30, .75);
  border: 1px dashed rgba(148,163,184,.34);
  border-radius: 16px; padding:.35rem;
}
[data-testid="stFileUploader"] section { background: transparent; }
[data-testid="stFileUploader"] button {
  background: #1a2440 !important; color:#f1f5f9 !important;
  border:1px solid rgba(167,139,250,.3) !important;
}
[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input, [data-testid="stSelectbox"] > div > div {
  background:#0b1120 !important; color:#edf2ff !important;
  border-color:rgba(148,163,184,.28) !important; border-radius:12px !important;
}
textarea::placeholder, input::placeholder { color:#64748b !important; }
[data-testid="stTextArea"] label, [data-testid="stTextInput"] label,
[data-testid="stFileUploader"] label { color:#d4dbea !important; }
.stButton > button, [data-testid="stDownloadButton"] button {
  border-radius:12px; font-weight:750; min-height:2.7rem;
  border:1px solid rgba(167,139,250,.32);
  color:#f5f3ff; background:linear-gradient(135deg, #6d4aff, #5140bd);
  box-shadow:0 7px 20px rgba(109,74,255,.18);
  transition:transform .16s ease, filter .16s ease;
}
.stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
  border-color:rgba(103,232,249,.6); filter:brightness(1.1); transform:translateY(-1px);
}
.stButton > button:focus { box-shadow:0 0 0 2px rgba(103,232,249,.25); }
.stProgress > div > div { background:#202a40; border-radius:99px; }
.stProgress > div > div > div > div {
  background:linear-gradient(90deg, #8b5cf6, #67e8f9); border-radius:99px;
}
div[data-testid="stExpander"] {
  border:1px solid var(--line); border-radius:16px;
  background:rgba(13,18,32,.72); overflow:hidden;
}
div[data-testid="stExpander"] summary { color:#edf2ff; }
[data-testid="stAlert"] { border-radius:14px; border:1px solid var(--line); }
hr { border-color:var(--line) !important; }
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border:1px solid var(--line); border-radius:12px; overflow:hidden;
}
code { color:#c4b5fd; background:rgba(167,139,250,.09); }
a { color:#a5f3fc !important; }
.footer { text-align:center; color:#65738e; font-size:.78rem; padding:1.7rem 0 .5rem; }
@media (max-width: 800px) {
  .block-container { padding:1rem 1rem 2.5rem; }
  .hero { padding:1.7rem 1.4rem; border-radius:22px; }
  .panel { padding:1rem; }
}
</style>

""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">Career insights · Resume intelligence</div>
  <h1>✦ TalentLens</h1>
  <p>Turn a resume and a target job description into a clear, explainable view of skill fit, missing capabilities, and next steps.</p>
  <div class="badge">✦ NLP-assisted &nbsp;·&nbsp; Explainable scoring &nbsp;·&nbsp; Private by design</div>
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
