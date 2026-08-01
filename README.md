# 🚀 AI Resume Analyzer & Job Matching System

An intelligent **NLP-based Resume Analysis Platform** built using **DistilBERT, Sentence Transformers, and Explainable Skill Extraction**.

This application analyzes resumes, predicts candidate professional categories, extracts technical skills, matches resumes with job descriptions, identifies missing skills, and provides actionable recommendations.

The project is designed as an end-to-end AI portfolio application using **Natural Language Processing, Transformer Models, Semantic Similarity, and Streamlit Deployment**.

---

# 📌 Project Overview

Recruiters often need to analyze hundreds of resumes for a single job opening. Manual resume screening is slow and inefficient.

This project automates resume screening by understanding resume content using modern NLP techniques.

The system provides:

✅ Resume Category Prediction  
✅ Confidence Score  
✅ Technical Skill Extraction  
✅ Resume–Job Description Matching  
✅ Missing Skill Identification  
✅ Match Score Analysis  

---

# ✨ Features

## 1. Resume Classification using DistilBERT

The system uses a fine-tuned **DistilBERT transformer model** to classify resumes into professional categories.

Example:

```
Resume:

Python
TensorFlow
Machine Learning
Deep Learning
Pandas
SQL


Prediction:

Category:
Information Technology


Confidence:
94.5%
```

---

# 2. Skill Extraction

The application automatically extracts important technical skills from resume text.

Example:

```
Input:

Experienced in Python, AWS, Docker, TensorFlow and SQL


Extracted Skills:

✓ Python
✓ AWS
✓ Docker
✓ TensorFlow
✓ SQL
```

The skill extraction module is transparent and extendable through a custom skill database.

---

# 3. Resume & Job Description Matching

The system compares resume content with job requirements using:

- Sentence Transformer embeddings
- Semantic similarity
- Cosine similarity


Workflow:

```
Resume Text

      +

Job Description

      ↓

Sentence Transformer

      ↓

Text Embeddings

      ↓

Similarity Calculation

      ↓

Match Score
```

Example:

```
Resume Match Score:

87%


Matched Skills:

✓ Python
✓ Machine Learning
✓ NLP
✓ SQL


Missing Skills:

✗ Kubernetes
✗ System Design
```

---

# 4. AI-Based Recommendations

The system identifies missing skills and provides improvement suggestions.

Example:

```
Recommendations:

• Learn cloud deployment technologies
• Add machine learning projects
• Improve system design knowledge
```

---

# 🧠 System Architecture


```
                 Resume PDF
                     |
                     |
             PDF Text Extraction
                     |
                     |
              Text Preprocessing
                     |
                     |
          -------------------------
          |                       |
          |                       |
   DistilBERT Model        Skill Extraction
          |                       |
          |                       |
 Resume Category           Technical Skills
          |
          |
 Confidence Score


Job Description
        |
        |
Sentence Transformer
(all-MiniLM-L6-v2)
        |
        |
Semantic Embedding
        |
        |
Cosine Similarity
        |
        |
Match Score

```
## 🌐 Live Demo

🔗 **Try the application here:**  
https://airesumeanalyser-at1.streamlit.app/
---

# 🛠️ Tech Stack


## Programming Language

- Python


## NLP & Deep Learning

- Natural Language Processing
- Transformers
- DistilBERT
- Sentence Transformers
- Text Classification
- Semantic Similarity


## Machine Learning Libraries

- PyTorch
- Hugging Face Transformers
- Scikit-learn
- NumPy
- Pandas


## Deployment

- Streamlit


---

# 📂 Project Structure


```
AI-Resume-Analyzer/

│
├── app.py
│       # Streamlit application

│
├── train_classifier.py
│       # DistilBERT training pipeline

│
├── preprocessing.py
│       # Text cleaning and preprocessing

│
├── matching.py
│       # Resume-job semantic matching

│
├── skill_extractor.py
│       # Rule-based skill extraction

│
├── utils.py
│       # Helper functions

│
├── requirements.txt

│
├── data/
│       └── Resume.csv

│
├── models/
│       └── resume_classifier/

│
└── README.md

```

---

# 📊 Dataset

The project uses a resume dataset containing:

- Resume text
- Professional categories


Example categories:

```
Information Technology
Data Science
Software Engineering
HR
Finance
Healthcare
Marketing
Engineering
```

The dataset is processed only when the dashboard or training pipeline is executed.

---

# 🤖 Machine Learning Pipeline


## Resume Classification Pipeline


```
Resume Text

      ↓

Text Cleaning

      ↓

DistilBERT Tokenizer

      ↓

Input IDs + Attention Mask

      ↓

DistilBERT Encoder

      ↓

Classification Layer

      ↓

Predicted Category

```

---

# 🔥 Model Details


## Resume Classification Model

Model:

```
distilbert-base-uncased
```

Task:

```
Multi-class Resume Classification
```


Why DistilBERT?

- Faster inference
- Smaller model size
- Efficient deployment
- Maintains strong language understanding


---

## Resume Matching Model


Model:

```
all-MiniLM-L6-v2
```


Method:

```
Sentence Embeddings

+

Cosine Similarity

=

Resume Match Score
```

Download NLTK resources:

```bash
python -m nltk.downloader wordnet stopwords
```

---

# ▶️ Run Application


Start Streamlit:

```bash
streamlit run app.py
```


Application will open:

```
http://localhost:8501
```

---

# 🏋️ Train Resume Classifier


Train DistilBERT classifier:

```bash
python train_classifier.py --epochs 2
```


For quick testing:

```bash
python train_classifier.py --sample-size 1000 --epochs 1
```


After training, model files are saved:

```
models/resume_classifier/
```

Restart Streamlit to enable resume category prediction.

---

---

# 🚀 Future Improvements


- ATS Resume Score Prediction
- LLM-based Resume Feedback
- Resume Ranking System
- AI Interview Question Generator
- RAG Career Assistant
- Multi-language Resume Support
- Automatic Resume Improvement Suggestions

---

# 🎯 Skills Demonstrated


This project demonstrates:

✅ NLP Pipeline Development  
✅ Transformer Fine-Tuning  
✅ DistilBERT Classification  
✅ Sentence Embeddings  
✅ Semantic Search  
✅ Text Preprocessing  
✅ Explainable AI Components  
✅ Streamlit Deployment  


---

# 👨‍💻 Author


## Anjani Tiwari

AI / Machine Learning Engineer


GitHub:

https://github.com/anjanitiwari123


---

⭐ If you found this project useful, consider giving it a star!
