# src/predict.py
"""Prediction module: single text and batch prediction."""

import torch
import pandas as pd
from typing import Union, List, Dict
from transformers import RobertaTokenizerFast

from src.model import RoBERTaBiLSTM
from src.utils import NewsDataset


class Predictor:
    def __init__(
        self,
        model_dir: str,
        tokenizer_name: str = "roberta-base",
        max_length: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.max_length = max_length

        self.model = RoBERTaBiLSTM.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)

    def predict_single(self, text: str) -> Dict[str, Union[bool, float, int]]:
        """Predict a single text."""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding["input_ids"].to(self.device),
                attention_mask=encoding["attention_mask"].to(self.device),
            )
            probs = torch.softmax(outputs["logits"], dim=-1).cpu().numpy()[0]
            pred = int(probs.argmax())

        return {
            "is_fake": bool(pred),
            "confidence": float(probs[pred]),
            "prob_real": float(probs[0]),
            "prob_fake": float(probs[1]),
        }

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
    ) -> List[Dict[str, Union[bool, float, int]]]:
        """Predict a batch of texts."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            dataset = NewsDataset(
                pd.DataFrame({"content": batch_texts, "label": [0] * len(batch_texts)}),
                self.tokenizer,
                self.max_length,
            )
            dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

            for batch in dl:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    probs_batch = torch.softmax(outputs["logits"], dim=-1).cpu().numpy()

                for probs in probs_batch:
                    pred = int(probs.argmax())
                    results.append({
                        "is_fake": bool(pred),
                        "confidence": float(probs[pred]),
                        "prob_real": float(probs[0]),
                        "prob_fake": float(probs[1]),
                    })
        return results