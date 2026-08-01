import json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from preprocessing import preprocess_text
from utils import load_resume_data
MODEL_NAME = "distilbert-base-uncased"
class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.tokens = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=256
        )
        self.labels = list(labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        item = {
            key: torch.tensor(value[index])
            for key, value in self.tokens.items()
        }
        item["labels"] = torch.tensor(
            self.labels[index]
        )
        return item
    
def calculate_accuracy(output):
    predictions = np.argmax(
        output.predictions,
        axis=1
    )
    accuracy = (
        predictions == output.label_ids
    ).mean()
    return {
        "accuracy": accuracy
    }

def train():
    data = load_resume_data(
        "data/Resume.csv"
    )
    data["clean_resume"] = data["Resume"].apply(
        preprocess_text
    )
    encoder = LabelEncoder()
    labels = encoder.fit_transform(
        data["Category"]
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data["clean_resume"],
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(
            encoder.classes_
        )
    )
    args = TrainingArguments(
        output_dir="models/checkpoints",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        report_to="none"

    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ResumeDataset(
            train_texts,
            train_labels,
            tokenizer
        ),
        eval_dataset=ResumeDataset(
            val_texts,
            val_labels,
            tokenizer
        ),
        compute_metrics=calculate_accuracy

    )
    trainer.train()
    save_path = Path(
        "models/resume_classifier"
    )
    save_path.mkdir(
        exist_ok=True
    )
    trainer.save_model(
        save_path
    )
    tokenizer.save_pretrained(
        save_path
    )

    labels_dict = {
        str(i):label
        for i,label in enumerate(
            encoder.classes_
        )
    }
    with open(
        save_path/"labels.json",
        "w"
    ) as f:

        json.dump(
            labels_dict,
            f,
            indent=2
        )
    print(
        "Model Saved Successfully"
    )
if __name__ == "__main__":

    train()