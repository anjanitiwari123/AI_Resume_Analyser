
from __future__ import annotations
import json
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_resume_data(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"Resume", "Category"}
    if not required.issubset(data.columns):
        raise ValueError(
            "Dataset must contain Resume and Category columns."
        )
    return (
        data
        .dropna(subset=["Resume", "Category"])
        .drop_duplicates(
            subset=["Resume", "Category"]
        )
        .reset_index(drop=True)
    )

def extract_text_from_pdf(pdf_bytes):
    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )
    text = ""
    for page in document:
        text += page.get_text()
    return text

def load_classifier(model_dir):
    model_path = Path(model_dir)
    label_file = model_path / "labels.json"
    if not (
        (model_path / "config.json").exists()
        and label_file.exists()
    ):
        return None
    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )
    with open(label_file) as f:
        labels = json.load(f)
    model.eval()
    return tokenizer, model, labels

def predict_category(text, classifier):
    tokenizer, model, labels = classifier

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        output = model(**inputs)
    probabilities = torch.softmax(
        output.logits,
        dim=1
    )[0]
    index = torch.argmax(
        probabilities
    ).item()

    return (
        labels[str(index)],
        probabilities[index].item()*100
    )
