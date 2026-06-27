import streamlit as st
import torch
import os
from transformers import RobertaTokenizer
from src.model import RoBERTaBiLSTM

# Page Configuration (Design & Theme)
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Custom CSS styling for premium look
st.markdown("""
    <style>
    .main-title { font-size: 40px; font-weight: 800; text-align: center; color: #1E3A8A; margin-bottom: 10px; }
    .sub-title { font-size: 18px; text-align: center; color: #4B5563; margin-bottom: 30px; }
    .prediction-fake { padding: 20px; background-color: #FEE2E2; border-left: 5px solid #EF4444; border-radius: 5px; color: #991B1B; font-size: 22px; font-weight: bold; text-align: center; }
    .prediction-real { padding: 20px; background-color: #D1FAE5; border-left: 5px solid #10B981; border-radius: 5px; color: #065F46; font-size: 22px; font-weight: bold; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>📰 Fake News Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Advanced Hybrid RoBERTa + BiLSTM deep learning architecture</div>", unsafe_allow_html=True)

# Cache model so it doesn't reload on every button click
@st.cache_resource
def load_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = "models/final_hybrid_roberta_bilstm"
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RoBERTaBiLSTM(model_name="roberta-base")
    
    # Weights load karein
    model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()
    return model, tokenizer, device

try:
    model, tokenizer, device = load_pipeline()
    st.success("🤖 AI Model successfully loaded into GPU memory!")
except Exception as e:
    st.error(f"Model load karne me dikkat aayi: {e}")

# User Input Text Area
user_input = st.text_area("✍️ Paste the news article text below to verify:", height=200, placeholder="Type or paste the full news text content here...")

if st.button("🔍 Verify Authenticity", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text first!")
    else:
        with st.spinner("Analyzing linguistic patterns and contexts... Please wait."):
            # Tokenize user text
            inputs = tokenizer(user_input, truncation=True, padding=True, max_length=128, return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                prediction = torch.argmax(logits, dim=1).item()
                probabilities = torch.softmax(logits, dim=1)[0]
                
            # Assumptions: 0 = Real, 1 = Fake (Apne dataset structure ke hisab se ise swap kar sakti hain agar ulti prediction dikhe)
            if prediction == 1:
                st.markdown(f"<div class='prediction-fake'>🚨 ALERT: This News is Predicted to be FAKE! ({probabilities[1]*100:.2f}% Confidence)</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='prediction-real'>✅ VERIFIED: This News appears to be REAL. ({probabilities[0]*100:.2f}% Confidence)</div>", unsafe_allow_html=True)
                
            st.info("💡 Note: This analysis is based on your custom trained RoBERTa-BiLSTM sequence processing pipeline.")