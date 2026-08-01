
from __future__ import annotations

import re
SKILL_DATABASE = [
    # Programming Languages
    "Python",
    "Java",
    "C++",
    "C",
    "JavaScript",
    "R",
    "Scala",

    # Web Development
    "React",
    "React.js",
    "Node.js",
    "Express",
    "Express.js",
    "Spring Boot",
    "HTML",
    "CSS",
    "Bootstrap",
    "Material UI",
    "REST API",
    "API Development",

    # Databases
    "MongoDB",
    "MySQL",
    "SQL",
    "PostgreSQL",
    "Oracle",
    "SQLite",
    "NoSQL",
    "Database Management System",
    "DBMS",
    "Database Design",

    # Computer Science Fundamentals
    "Data Structures",
    "Data Structures and Algorithms",
    "Algorithms",
    "Object Oriented Programming",
    "OOP",
    "Operating Systems",
    "Computer Networks",
    "Software Engineering",
    "System Design",

    # Data Analysis Skills
    "Excel",
    "Advanced Excel",
    "Power BI",
    "Tableau",
    "Data Visualization",
    "Data Analysis",
    "Business Intelligence",
    "Statistics",
    "Exploratory Data Analysis",
    "EDA",
    "Reporting",
    "Dashboard",

    # Python Data Science Libraries
    "Pandas",
    "NumPy",
    "Matplotlib",
    "Seaborn",
    "Plotly",
    "SciPy",

    # Machine Learning
    "Machine Learning",
    "Supervised Learning",
    "Unsupervised Learning",
    "Regression",
    "Classification",
    "Clustering",
    "Feature Engineering",
    "Model Evaluation",
    "Scikit-learn",
    "XGBoost",
    "Random Forest",
    "Decision Tree",
    "Logistic Regression",
    "SVM",
    "KNN",

    # Deep Learning
    "Deep Learning",
    "Neural Networks",
    "ANN",
    "CNN",
    "RNN",
    "LSTM",
    "TensorFlow",
    "Keras",
    "PyTorch",

    # NLP
    "NLP",
    "Natural Language Processing",
    "Text Processing",
    "Text Classification",
    "Sentiment Analysis",
    "Transformers",
    "BERT",
    "GPT",
    "Hugging Face",

    # Generative AI
    "Generative AI",
    "LLM",
    "Large Language Model",
    "RAG",
    "LangChain",
    "Prompt Engineering",
    "Vector Database",
    "Embeddings",

    # Cloud & Deployment
    "AWS",
    "Azure",
    "Google Cloud",
    "GCP",
    "Docker",
    "Kubernetes",
    "CI/CD",
    "Git",
    "GitHub",
    "Linux",

    # Data Engineering
    "ETL",
    "Data Pipeline",
    "Apache Spark",
    "Hadoop",
    "Kafka",
    "Airflow",
    "Data Warehouse",
    "Big Data"
]
def extract_skills(text, skill_database=SKILL_DATABASE):
    found_skills = []
    text = text.lower()
    for skill in skill_database:
        pattern = (
            r"(?<!\w)"
            + re.escape(skill.lower())
            + r"(?!\w)"
        )
        if re.search(pattern, text):
            found_skills.append(skill)
    return found_skills

def compare_skills(resume_text, job_description):
    resume_skills = set(
        extract_skills(resume_text)
    )
    job_skills = set(
        extract_skills(job_description)
    )
    matched_skills = resume_skills & job_skills
    missing_skills = job_skills - resume_skills
    return (
        sorted(matched_skills),
        sorted(missing_skills)
    )