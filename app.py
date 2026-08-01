from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from matching import calculate_final_match_score, load_matching_model
from preprocessing import preprocess_text
from skill_extractor import compare_skills, extract_skills
from utils import extract_text_from_pdf, load_classifier, predict_category
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "resume_classifier"
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)
@st.cache_resource
def get_classifier():
    return load_classifier(MODEL_PATH)
@st.cache_resource
def get_matching_model():
    return load_matching_model()

def score_chart(score):
    return go.Figure(
        go.Pie(
            labels=["Matched", "Missing"],
            values=[score, 100 - score],
            hole=0.65,
            textinfo="label+percent"
        )
    ).update_layout(
        title="ATS Resume Match Score",
        height=330,
        showlegend=False
    )

def recommendations(missing_skills):
    groups = {
        "Cloud Deployment": {
            "AWS",
            "Azure",
            "GCP",
            "Docker",
            "Kubernetes"
        },

        "Data Science": {
            "Python",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Machine Learning"
        },

        "Deep Learning": {
            "TensorFlow",
            "PyTorch",
            "Deep Learning",
            "NLP"
        }
    }
    tips = [
        f"Add projects or experience related to {skill}"
        for skill in missing_skills[:3]
    ]
    for topic, skills in groups.items():
        if skills.intersection(missing_skills):
            tips.append(
                f"Improve your {topic} skills for better job compatibility."
            )
    return list(dict.fromkeys(tips))[:5]
st.title("AI Resume Analyzer & Job Matcher")
st.caption(
    "DistilBERT role classification + Skill Extraction + Sentence Transformer ATS Matching"
)
uploaded_file = st.file_uploader(
    "Upload PDF Resume",
    type=["pdf"]
)
manual_text = st.text_area(
    "Or paste resume text"
)
job_description = st.text_area(
    "Paste Job Description"
)

if st.button("Analyze Resume", type="primary"):
    try:
        if uploaded_file:
            resume_text = extract_text_from_pdf(
                uploaded_file.getvalue()
            )
        else:
            resume_text = manual_text
        if not resume_text.strip():
            st.warning(
                "Please upload resume or paste text"
            )
            st.stop()
        cleaned_resume = preprocess_text(
            resume_text
        )
        skills = extract_skills(
            resume_text
        )
        classifier = get_classifier()
        if classifier:
            prediction, confidence = predict_category(
                cleaned_resume,
                classifier
            )

        else:

            prediction = "Model not trained"
            confidence = 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Predicted Role",
            prediction
        )
        c2.metric(
            "Confidence",
            f"{confidence:.1f}%"
        )
        c3.metric(
            "Words",
            len(resume_text.split())
        )
        c4.metric(
            "Skills",
            len(skills)
        )
        st.subheader(
            "Detected Skills"
        )
        st.write(
            " · ".join(
                f"✓ {skill}"
                for skill in skills
            )
        )
        if job_description.strip():
            match_model = get_matching_model()
            matched, missing = compare_skills(
                resume_text,
                job_description
            )
            score = calculate_final_match_score(
                resume_text,
                job_description,
                matched,
                missing,
                match_model
            )
            left, right = st.columns(2)
            left.plotly_chart(
                score_chart(score),
                use_container_width=True
            )
            right.subheader(
                f"ATS Match Score: {score}%"
            )
            right.caption(
                "Score based on 70% skill matching and 30% semantic similarity"
            )
            right.write(
                "**Matching Skills:**"
            )
            right.write(
                ", ".join(
                    f"✓ {x}"
                    for x in matched
                )
                if matched
                else "None"
            )
            right.write(
                "**Missing Skills:**"
            )
            right.write(
                ", ".join(
                    f"✗ {x}"
                    for x in missing
                )
                if missing
                else "No missing skills"
            )
            tips = recommendations(
                missing
            )
            if tips:
                st.subheader(
                    "Recommendations"
                )
                for tip in tips:
                    st.write(
                        "• " + tip
                    )
    except Exception as e:
        st.error(
            f"Analysis failed: {e}"
        )