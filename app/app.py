# app/app.py
"""Streamlit demo interface for the RoBERTa-BiLSTM fake news classifier."""

import os

import streamlit as st
import torch
from transformers import RobertaTokenizer

from src.model import RoBERTaBiLSTM

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.markdown(
    """
    <style>
    .main-title { font-size: 40px; font-weight: 800; text-align: center; color: #1E3A8A; margin-bottom: 10px; }
    .sub-title { font-size: 18px; text-align: center; color: #4B5563; margin-bottom: 30px; }
    .prediction-fake { padding: 20px; background-color: #FEE2E2; border-left: 5px solid #EF4444; border-radius: 5px; color: #991B1B; font-size: 22px; font-weight: bold; text-align: center; }
    .prediction-real { padding: 20px; background-color: #D1FAE5; border-left: 5px solid #10B981; border-radius: 5px; color: #065F46; font-size: 22px; font-weight: bold; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='main-title'>📰 Fake News Detection System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Hybrid RoBERTa + BiLSTM + Attention architecture</div>",
    unsafe_allow_html=True,
)

# Label convention is fixed at training time in src/data_prep.py
# (load_isot_dataset: df_fake["label"] = 1, df_true["label"] = 0) — it is
# not something to guess or swap at inference time. A previous version of
# this file had a comment suggesting the labels could be swapped here if
# predictions "look reversed"; if predictions ever look inverted, the bug
# is in training-data labeling, not in this display code, and should be
# fixed at the source rather than papered over here.
LABEL_REAL = 0
LABEL_FAKE = 1


@st.cache_resource
def load_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = "models/final_hybrid_roberta_bilstm"

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RoBERTaBiLSTM(model_name="roberta-base")

    model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained checkpoint found at {model_path}. Run `python -m src.train` first."
        )
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()
    return model, tokenizer, device


try:
    model, tokenizer, device = load_pipeline()
    st.success("Model loaded.")
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()

user_input = st.text_area(
    "Paste the news article text below to verify:",
    height=200,
    placeholder="Type or paste the full news text content here...",
)

if st.button("Verify Authenticity", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(
                user_input, truncation=True, padding=True, max_length=128, return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                prediction = torch.argmax(logits, dim=1).item()
                probabilities = torch.softmax(logits, dim=1)[0]

        if prediction == LABEL_FAKE:
            st.markdown(
                f"<div class='prediction-fake'>Predicted FAKE "
                f"({probabilities[LABEL_FAKE] * 100:.2f}% confidence)</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='prediction-real'>Predicted REAL "
                f"({probabilities[LABEL_REAL] * 100:.2f}% confidence)</div>",
                unsafe_allow_html=True,
            )

        st.caption(
            "This prediction reflects patterns learned from the training data "
            "(ISOT + LIAR + Kaggle) and is not a fact-check. See the "
            "repository README's Known Limitations section for dataset caveats."
        )
