[README.md](https://github.com/user-attachments/files/26951175/README.md)
# GlucoSense AI — Diabetes Prediction System

A full-stack Machine Learning web application that predicts diabetes risk using the PIMA Indians Diabetes Dataset.

---

## Project Structure

```
glucosense/
├── app.py                    ← FastAPI server (all API routes)
├── train_model.py            ← Run once to train & save model
├── pyproject.toml            ← All Python dependencies
├── data/
│   └── diabetes.csv          ← ← ← PUT KAGGLE DATASET HERE
├── models/
│   ├── diabetes_model.pkl    ← Auto-generated after training
│   └── scaler.pkl            ← Auto-generated after training
├── templates/
│   └── index.html            ← Full frontend UI
└── utils/
    ├── predictor.py          ← ML model, training, SHAP-like explanations
    ├── report_generator.py   ← PDF report generation (ReportLab)
    └── report_parser.py      ← Parse uploaded PDF/JSON/CSV reports
```

---

## Step-by-Step Assembly Guide

### STEP 1 — Download the Kaggle Dataset

1. Go to: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
2. Click **Download** (you need a free Kaggle account)
3. Extract the ZIP — you'll find `diabetes.csv`
4. Place it inside the `data/` folder:
   ```
   glucosense/data/diabetes.csv
   ```

> **Without the dataset:** The app still works using synthetic training data,
> but accuracy may differ. For your college project, always use the real dataset.

---

### STEP 2 — Set Up Python Environment

Open a terminal, navigate to the project folder, then:

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install all dependencies
uv sync

# Set Gemini API key for AI chat
export GEMINI_API_KEY="your_api_key_here"
# Optional: override model
# export GEMINI_MODEL="gemini-2.5-flash"
```

---

### STEP 3 — Train the Model

```bash
python train_model.py
```

You should see output like:
```
Training Complete!
  Accuracy  : 79.22%
  ROC-AUC   : 0.8541
  CV Mean   : 77.86% ± 2.3%
  Algorithm : Random Forest (200 trees)
  Model saved to models/
```

This creates `models/diabetes_model.pkl` and `models/scaler.pkl`.

---

### STEP 4 — Run the App

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 5000
```

Open your browser at: **http://localhost:5000**

---

## Features

| Feature | Description |
|---|---|
| **Risk Gauge** | Animated 0–100% risk score meter |
| **8 Biomarker Sliders** | PIMA dataset features with color-coded status bars |
| **Feature Analysis Tab** | Shows which input contributed most to the prediction |
| **What-If Simulator** | Shows how improving each metric reduces risk |
| **Upload Report** | Upload PDF / JSON / CSV to auto-fill sliders |
| **Download PDF Report** | Full medical-style PDF with gauge, tips, table |
| **Export JSON / CSV** | Export results in both formats |
| **Batch Analysis** | Upload CSV with many patients, get results CSV |
| **Prediction History** | Tracks last 20 predictions (browser localStorage) |
| **AI Chat Tab** | Gemini-powered assistant with streamed responses |
| **Model Info Tab** | Shows accuracy, AUC, CV score, feature importances |

---

## API Endpoints

| Method | URL | Description |
|---|---|---|
| POST | `/api/predict` | Single patient prediction |
| POST | `/api/download-report` | Generate and download PDF |
| POST | `/api/upload-report` | Parse uploaded PDF/JSON/CSV |
| POST | `/api/batch-predict` | Batch CSV prediction |
| POST | `/api/chat` | Chat with GlucoBot assistant (non-stream) |
| POST | `/api/chat/stream` | Stream chat response chunks from Gemini |
| GET  | `/api/model-info` | Model metrics and feature importances |

---

## ML Model Details

- **Algorithm:** Random Forest Classifier (200 trees)
- **Dataset:** PIMA Indians Diabetes Database (768 samples, 8 features)
- **Preprocessing:** StandardScaler, median imputation for zero values
- **Evaluation:** Train/test split 80/20 + 5-fold cross-validation
- **Explainability:** Permutation-based feature contribution scores

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `uv sync` |
| AI chat says API key missing | Set `GEMINI_API_KEY` before running server |
| PDF download fails | Make sure `reportlab` is installed |
| Model not found error | Run `python train_model.py` first |
| Port 5000 in use | Run with `--port 5001` in the uvicorn command |
| Kaggle CSV not loading | Check file is named `diabetes.csv` in `data/` folder |
