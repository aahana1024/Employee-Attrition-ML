import streamlit as st
import pandas as pd
import PyPDF2
import io
from catboost import CatBoostClassifier, Pool

st.set_page_config(page_title="AttritionIQ", layout="wide")
st.title("🚀 AttritionIQ – Employee Attrition Predictor")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("cat_model.cbm")
    return model

model = load_model()

CAT_COLS = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime"
]

# ─────────────────────────────────────────────────────────────────────────────
# JOB ROLES — loaded once, used by both sections
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_job_roles():
    return pd.read_csv("job_roles.csv")

try:
    job_roles_df = load_job_roles()
except FileNotFoundError:
    st.error("job_roles.csv not found. Place it in the same directory as app.py.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY ROLE STORE
# Simulates a vector DB: maps each Role to list of required skills.
# At prediction time this is filtered to only high-risk roles.
# Works for both static CSV and live data inputs.
# ─────────────────────────────────────────────────────────────────────────────

def build_role_store(job_roles_df):
    store = {}
    for _, row in job_roles_df.iterrows():
        skills = [s.strip().lower() for s in str(row["Required_Skills"]).split(",") if s.strip()]
        store[row["Role"]] = skills
    return store

def filter_store_to_high_risk(role_store, high_risk_roles):
    filtered = {role: skills for role, skills in role_store.items() if role in high_risk_roles}
    return filtered if filtered else role_store

full_role_store = build_role_store(job_roles_df)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: ATTRITION PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader("Upload Employee CSV", type=["csv"])

high_risk_roles_detected = []

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    df_proc = df.copy()

    for col in ["Attrition", "EmployeeCount", "EmployeeNumber", "StandardHours", "Over18"]:
        if col in df_proc.columns:
            df_proc = df_proc.drop(col, axis=1)

    model_feature_names = model.feature_names_
    df_proc = df_proc[model_feature_names]

    for col in CAT_COLS:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].astype(str)

    cat_feature_indices = [df_proc.columns.tolist().index(col) for col in CAT_COLS if col in df_proc.columns]
    pool = Pool(df_proc, cat_features=cat_feature_indices)
    prob = model.predict_proba(pool)[:, 1]

    df["Attrition Risk %"] = (prob * 100).round(2)
    df["Risk Level"] = pd.cut(
        prob,
        bins=[0, 0.4, 0.7, 1.0],
        labels=["Low", "Medium", "High"]
    )

    st.subheader("Predictions")
    st.dataframe(df)

    st.subheader("High Risk Employees")
    high = df[df["Attrition Risk %"] > 70]
    st.dataframe(high)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", len(df))
    col2.metric("High Risk", len(high))
    col3.metric("Avg Attrition Risk", f"{df['Attrition Risk %'].mean():.1f}%")

    if "JobRole" in high.columns and not high.empty:
        high_risk_roles_detected = high["JobRole"].unique().tolist()

else:
    st.info("Upload a CSV file to start.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: RESUME SCREENING & HIRING
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.header("Resume Screening & Hiring")

active_role_store = filter_store_to_high_risk(full_role_store, high_risk_roles_detected)

if high_risk_roles_detected:
    st.info(
        f"Automated mode: Screening filtered to {len(active_role_store)} high-risk role(s) "
        f"from your employee data: {', '.join(sorted(active_role_store.keys()))}"
    )
else:
    st.info(
        "All-roles mode: No employee CSV uploaded yet or no high-risk employees found. "
        "Screening against all available job roles."
    )

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def clean_text(raw_text):
    return raw_text.lower().strip()

def match_skills(resume_text, required_skills):
    matched = [s for s in required_skills if s in resume_text]
    missing = [s for s in required_skills if s not in resume_text]
    score = len(matched) / len(required_skills) if required_skills else 0.0
    return {"matched": matched, "missing": missing, "score": score}

role_options = list(active_role_store.keys())
selected_role = st.selectbox("Select Job Role to Screen For", role_options)

required_skills_display = ", ".join(s.title() for s in active_role_store[selected_role])
role_dept = job_roles_df.loc[job_roles_df["Role"] == selected_role, "Department"]
dept_display = role_dept.values[0] if not role_dept.empty else "N/A"
st.caption(f"Department: {dept_display}  |  Required Skills: {required_skills_display}")

resume_file = st.file_uploader("Upload Candidate Resume (PDF)", type=["pdf"], key="resume_uploader")

if resume_file is not None:
    with st.spinner("Analysing resume..."):
        raw_text = extract_text_from_pdf(resume_file)

        if not raw_text.strip():
            st.warning("Could not extract text from this PDF. It may be scanned or image-based.")
        else:
            cleaned_resume = clean_text(raw_text)
            required_skills = active_role_store[selected_role]
            result = match_skills(cleaned_resume, required_skills)
            score = result["score"]
            score_pct = round(score * 100, 1)

            st.subheader("Screening Results")
            st.metric(
                label="Match Score",
                value=f"{score_pct}%",
                delta=f"{len(result['matched'])} of {len(required_skills)} skills found"
            )

            col_found, col_missing = st.columns(2)
            with col_found:
                st.success(f"Skills Found ({len(result['matched'])})")
                for skill in result["matched"]:
                    st.write(f"- {skill.title()}")
                if not result["matched"]:
                    st.write("None detected")

            with col_missing:
                st.error(f"Missing Skills ({len(result['missing'])})")
                for skill in result["missing"]:
                    st.write(f"- {skill.title()}")
                if not result["missing"]:
                    st.write("None — full match!")

            st.subheader("Recommendation")
            if score > 0.7:
                st.success(f"Recommended Candidate — Strong skill alignment ({score_pct}% match).")
            else:
                st.warning(f"Not a strong match — Skill coverage is {score_pct}%. Consider additional screening.")
else:
    st.info("Upload a PDF resume above to begin screening.")
