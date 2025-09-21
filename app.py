# app.py
import streamlit as st
from parser import extract_text_from_uploaded_file
from scoring import compute_scores, generate_suggestions, parse_jd
from db import init_db, save_evaluation, fetch_evaluations
import os
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Resume Relevance Checker", layout="wide")

st.title("Automated Resume Relevance Check — MVP")
st.markdown(
    "Upload a Job Description (JD) and student resumes; "
    "the app returns relevance scores, missing skills, and suggestions."
)

# initialize DB
conn = init_db()

# sidebar controls
st.sidebar.header("Options")
hard_weight = st.sidebar.slider("Hard-match weight", 0.0, 1.0, 0.6, step=0.1)
soft_weight = 1.0 - hard_weight
st.sidebar.write(f"Soft-match weight = {soft_weight:.1f}")
fuzz_threshold = st.sidebar.slider(
    "Skill fuzzy-match threshold (higher = stricter)", 50, 100, 80, step=1
)

mode = st.sidebar.radio("Mode", ["Evaluate resumes", "View saved evaluations"])

# =======================
# MODE 1: EVALUATE RESUMES
# =======================
if mode == "Evaluate resumes":
    st.header("1) Provide Job Description (JD)")
    jd_file = st.file_uploader(
        "Upload JD file (optional, PDF/DOCX/TXT) or paste text below",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False,
    )
    jd_text = ""
    if jd_file is not None:
        jd_text = extract_text_from_uploaded_file(jd_file)
        st.success("JD loaded from file.")
    jd_text_manual = st.text_area(
        "Or paste JD text here (overrides uploaded file)", height=200
    )
    if jd_text_manual.strip():
        jd_text = jd_text_manual

    if not jd_text.strip():
        st.info("Please upload or paste a JD to proceed. A sample JD is below for quick testing.")
        if st.button("Load sample JD"):
            import pathlib
            sample_path = pathlib.Path("sample_data/sample_jd.txt")
            if sample_path.exists():
                jd_text = sample_path.read_text()
                st.experimental_rerun()
            else:
                st.warning("No sample JD found. Continue and paste a JD.")
    else:
        parsed = parse_jd(jd_text)
        st.markdown(f"**Job title detected:** {parsed['title']}")
        st.markdown(
            "**Auto-extracted required skills:** "
            + (", ".join(parsed["required_skills"]) if parsed["required_skills"] else "None detected")
        )
        st.markdown(
            "**Auto-extracted optional skills:** "
            + (", ".join(parsed["optional_skills"]) if parsed["optional_skills"] else "None detected")
        )

    st.markdown("---")
    st.header("2) Upload one or more resumes to evaluate")
    uploaded = st.file_uploader(
        "Upload resumes (PDF/DOCX/TXT). You can select multiple.",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt"],
    )

    if uploaded:
        if st.button("Evaluate uploaded resumes"):
            results = []
            for file in uploaded:
                with st.spinner(f"Parsing {file.name} ..."):
                    txt = extract_text_from_uploaded_file(file)
                with st.spinner(f"Scoring {file.name} ..."):
                    score_res = compute_scores(
                        txt,
                        jd_text or "",
                        hard_weight=hard_weight,
                        soft_weight=soft_weight,
                        fuzz_threshold=fuzz_threshold,
                    )
                # store
                save_evaluation(
                    conn,
                    file.name,
                    score_res["title"],
                    score_res["final_score"],
                    score_res["verdict"],
                    score_res["missing_skills"],
                    score_res["matched_skills"],
                    txt[:1000],
                    jd_text,
                )
                # suggestions (without OpenAI by default)
                suggestion = generate_suggestions(
                    txt,
                    score_res["missing_skills"],
                    score_res["title"],
                    openai_api_key=os.environ.get("OPENAI_API_KEY"),
                )
                # show
                st.subheader(f"{file.name} — Score: {score_res['final_score']} ({score_res['verdict']})")
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(
                        f"**Hard score:** {score_res['hard_score']} | **Soft score:** {score_res['soft_score']}"
                    )
                    st.markdown(
                        "**Matched skills:** "
                        + (", ".join(score_res["matched_skills"]) if score_res["matched_skills"] else "None")
                    )
                    st.markdown(
                        "**Missing required skills:** "
                        + (", ".join(score_res["missing_skills"]) if score_res["missing_skills"] else "None")
                    )
                    st.markdown("**Suggestions:**")
                    st.write(suggestion)
                    with st.expander("Show resume snippet"):
                        st.code(txt[:2000])
                with col2:
                    if st.button(f"Mark {file.name} as shortlisted (High)", key=f"short_{file.name}"):
                        st.success(f"Marked {file.name}")
                results.append(score_res)
            st.success("Evaluation completed for all uploaded resumes.")

# =======================
# MODE 2: DASHBOARD (PLOTLY)
# =======================
elif mode == "View saved evaluations":
    st.header("Saved evaluations")
    min_score = st.slider("Minimum score filter", 0, 100, 0)
    job_filter = st.text_input("Job title contains (optional)")
    rows = fetch_evaluations(
        conn, min_score=min_score, job_title_substr=job_filter or None
    )

    st.subheader("Evaluation Summary")

    if not rows:
        st.info("No saved evaluations yet.")
    else:
        df = pd.DataFrame(rows)
        df["missing_skills"] = df["missing_skills"].apply(
            lambda x: ", ".join(eval(x)) if x else ""
        )

        col1, col2 = st.columns(2)

        # Score distribution
        with col1:
            fig = px.histogram(df, x="score", nbins=10, title="Score Distribution")
            fig.update_layout(title={'x':0.5})  # center title
            st.plotly_chart(fig, use_container_width=True)

        # Verdict distribution
        with col2:
            fig2 = px.pie(df, names="verdict", title="Verdict Distribution")
            fig2.update_layout(title={'x':0.5})
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)

        # Avg score by job
        with col3:
            avg_scores = df.groupby("job_title")["score"].mean().reset_index()
            fig3 = px.bar(avg_scores, x="score", y="job_title",
                          orientation="h", title="Avg Score by Job")
            fig3.update_layout(title={'x':0.5})
            st.plotly_chart(fig3, use_container_width=True)

        # Scores over time
        with col4:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            daily = df.groupby(df["timestamp"].dt.date)["score"].mean().reset_index()
            fig4 = px.line(daily, x="timestamp", y="score",
                           markers=True, title="Scores Over Time")
            fig4.update_layout(title={'x':0.5})
            st.plotly_chart(fig4, use_container_width=True)

        # 🔍 Filter by verdict
        verdict_filter = st.multiselect(
            "Filter by verdict",
            df["verdict"].unique(),
            default=df["verdict"].unique(),
        )
        filtered = df[df["verdict"].isin(verdict_filter)]

        # 📋 Show table
        st.dataframe(
            filtered[
                ["id", "timestamp", "resume_filename", "job_title", "score", "verdict", "missing_skills"]
            ].sort_values(by="score", ascending=False)
        )

        # 💾 Download button
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export CSV of evaluations",
            csv,
            file_name="evaluations.csv",
            mime="text/csv",
        )
