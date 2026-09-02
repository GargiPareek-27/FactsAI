# FactsAI: Hybrid RoBERTa-BiLSTM Fake News Detection System
## Streamlit Application

<p align="center">
<img src="assets/streamlit_demo.png" width="900">
</p>


**FactsAI** is a modular hybrid deep learning system for fake news detection built with PyTorch and Hugging Face Transformers. By synergizing the contextual intelligence of **RoBERTa** with the sequential modeling of **Bidirectional LSTMs (BiLSTM)** and an integrated **Attention Mechanism**, FactsAI provides a robust framework for identifying deceptive linguistic patterns in news media.

-----

### 📖 Project Motivation

Misinformation spreads rapidly across digital platforms, making automated fact verification increasingly important. Classical machine learning models often struggle to capture contextual semantics, while transformer models alone may not fully exploit sequential dependencies. **FactsAI** addresses these challenges by combining RoBERTa's deep contextual embeddings with a BiLSTM's ability to model narrative flow, ensuring a high-resolution analysis of news content.

-----

### 🏗️ Architecture & How It Works

#### **Conceptual Workflow**

`News Article` → `Cleaning` → `Deduplication` → `Tokenizer` → `RoBERTa Encoder` → `BiLSTM` → `Attention` → `Classifier` → `Prediction`

#### **Technical Specification**

The system implements a sophisticated hybrid architecture defined in `src/model.py`:

1.  **RoBERTa-Base Encoder**: Processes raw tokens into 768-dimensional contextual embeddings.
2.  **Bidirectional LSTM (BiLSTM)**: Processes these embeddings from both directions to capture long-range sequential dependencies.
3.  **Additive Attention Layer**: Learns to assign importance weights to specific words or phrases most indicative of "fake" or "real" news, with padded positions masked out of the softmax so they contribute zero weight.
4.  **Classification Head**: A dense layer with dropout regularization for final binary classification.

#### **Mathematical Formulation**

  - **Embedding Extraction**: $H = \\text{RoBERTa}(X)$
  - **Sequential Modeling**: $H' = \\text{BiLSTM}(H)$
  - **Attention Score**: $\\alpha = \\text{softmax}(W H' + b)$
  - **Weighted Context**: $C = \\sum \\alpha H'$
  - **Prediction**: $\\hat{y} = \\text{softmax}(W\_c C + b\_c)$

-----

### 📂 Repository Structure

``` text
FactsAI/
├── app/
│   └── app.py                    # Streamlit Web Interface
├── assets/                       # Architecture diagrams and screenshots
├── configs/
│   └── config.yaml               # Training & Model Hyperparameters
├── data/
│   ├── raw/                      # Source datasets (ISOT, LIAR, Kaggle)
│   └── processed/                # Cleaned, deduplicated CSV splits + integrity logs
├── models/
│   └── final_hybrid_roberta_bilstm/       # Saved PyTorch model checkpoints
├── notebooks/
│   └── EDA.ipynb                 # Exploratory Data Analysis
├── results/                      # Generated metrics, reports, and plots (not committed)
├── src/
│   ├── model.py                  # Hybrid Architecture Definition
│   ├── train.py                  # GPU-optimized Training Pipeline
│   ├── evaluate.py               # Performance Metrics & Visualization
│   ├── predict.py                # Inference Module
│   ├── data_prep.py              # Preprocessing, Deduplication & Splitting
│   ├── audit_data_integrity.py   # Automated split/leakage audit
│   └── test_data_integrity.py    # Fast tests (no training required)
├── requirements.txt               # Dependency Management
└── .gitignore                     # Repository safety
```

-----

### ⚙️ Installation & Usage

1.  **Clone & Setup Environment**

    ``` bash
    git clone https://github.com/GargiPareek-27/FactsAI.git
    cd FactsAI
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **Data Preparation**
    Place your raw data in `data/raw/` and run the pipeline. This step deduplicates and resolves label conflicts before creating the train/val/test split — see [Data Integrity & Evaluation](#-data-integrity--evaluation) below.

    ``` bash
    python -m src.data_prep
    python -m src.audit_data_integrity --json-out data/processed/logs/audit_report.json
    ```

3.  **Training**

    ``` bash
    python -m src.train
    ```

4.  **Evaluation**

    ``` bash
    python -m src.evaluate \
        --model_dir models/final_hybrid_roberta_bilstm \
        --test_csv data/processed/test.csv \
        --checkpoint_trained_post_dedup_fix
    ```

5.  **Inference (Streamlit)**

    ``` bash
    streamlit run app/app.py
    ```

6.  **Tests** (fast, synthetic fixtures — no model training)

    ``` bash
    pytest src/test_data_integrity.py -q
    ```

-----

### 🛡️ GitHub Upload Strategy

| ✅ Commit These (Code & Config)  | ❌ Do NOT Commit (Data & Binaries) |
| :------------------------------ | :-------------------------------- |
| `src/`, `app/`, `notebooks/`    | `data/` (Raw/Processed CSVs)      |
| `configs/config.yaml`           | `models/*.bin` (Large Weights)    |
| `README.md`, `requirements.txt` | `results/` (Generated Plots)      |
| `.gitignore`, `LICENSE`         | `venv/`, `__pycache__/`           |

-----

### 🔍 Data Integrity & Evaluation

- Datasets (ISOT + Kaggle + LIAR) are merged, cleaned, and **deduplicated with label-conflict resolution before the train/val/test split** (`src/data_prep.py`), so an identical article cannot appear in both training and evaluation data.
- `src/audit_data_integrity.py` independently verifies this: it checks exact cross-split overlap, within-split duplicates, and label conflicts, and reports a plain **PASS / WARNING / FAIL** status — it does not print PASS unconditionally.
- An optional MinHash-based near-duplicate check is available via `--check-near-duplicates` as a secondary diagnostic (not a guarantee of zero leakage).
- `src/evaluate.py` computes accuracy, precision, recall, binary/macro/weighted F1, and ROC-AUC directly from the loaded checkpoint's predictions, and writes `results/metrics.json`, `results/classification_report.json`, `results/confusion_matrix.png`, `results/roc_curve.png`, and `results/evaluation_summary.md` — all generated at run time, none hardcoded.
- `src/test_data_integrity.py` covers the above with fast synthetic fixtures (no model training required).

-----

### 📝 Notes & Scope

- **Metrics require a fresh evaluation run.** No checkpoint or dataset is committed to this repo (see the upload strategy above), so `results/` numbers you see are only ever produced by actually running `src/evaluate.py` against a trained checkpoint.
- **ISOT's real/fake subsets carry known formatting differences** (e.g. wire-service dateline conventions) independent of content truthfulness. Deduplication removes leakage from repeated articles but doesn't remove this dataset-level stylistic signal — treat reported accuracy as a measure of pattern classification, not verified fact-checking.
- English-language input only.

-----

### 🚀 Future Improvements

- [ ] **Multilingual Support**: Integration of mBERT or XLM-RoBERTa.
- [ ] **Explainable AI (XAI)**: Integration of SHAP/LIME for better attention interpretability.
- [ ] **Model Quantization**: Exporting to ONNX for faster edge deployment.
- [ ] **API Access**: Deployment as a FastAPI REST service.

-----

## 👩‍💻 Author

**Gargi Pareek**

B.Tech Computer Science & Engineering, IIIT Pune

Aspiring AI/ML Engineer | Deep Learning | NLP | LLMs | Open Source

📧 Mail ID: gargipareek2007@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/gargi-pareek-004895364/

💻 GitHub: https://github.com/GargiPareek-27
