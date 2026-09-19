# VisionLens Detector

VisionLens Detector is the FYP implementation of the **AI-Based Eye Disease Detection System** proposal. It provides a professional screening workflow for three cataract stages: Normal, Immature, and Mature.

## Features

- Streamlit clinical dashboard with patient details and drag-and-drop upload
- FastAPI prediction service with CLAHE + Gaussian preprocessing
- ResNet-50 model loading after training
- Grad-CAM heatmap generation for model explainability
- SQLite patient and screening history
- Searchable patient screening history and doctor-review notes
- Analytics dashboard for screening volume, classifications, and pending reviews
- Local model-evaluation workspace with per-class precision, recall, F1, and a confusion matrix
- Downloadable PDF screening report
- Dataset preparation and model-training script

## Run locally

Open two terminals from this folder after activating `.venv`.

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload --port 8000
```

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

The dashboard opens at `http://localhost:8501`; the API documentation is at `http://localhost:8000/docs`.

## Train the model

First unpack `archive.zip` to `data/raw/` so the folders are `data/raw/archive/train/Normal`, `immature`, and `mature`. Then install TensorFlow and run:

```powershell
pip install tensorflow
python training/train.py --data-dir data/raw/archive/train --epochs 20
```

This creates `models/cataract_resnet50.keras`. The API will use it automatically. Until a trained model exists, the system intentionally does **not** return a diagnostic prediction.

## Optional Groq LLM guidance

The AI guidance button creates patient-friendly next-step explanations after a screening. It uses only the local model's stage and confidence; the uploaded eye image is **not** transmitted to the LLM.

1. Install the Groq Python package: `\.venv\Scripts\python.exe -m pip install groq`
2. Create an API key in the [Groq Console](https://console.groq.com/keys).
3. Copy `.env.example` to `.env`, then add your newly-created key after `GROQ_API_KEY=`. The backend loads this file automatically whenever it starts.
4. Start or restart the FastAPI backend.

The `.env` file is ignored by Git, so the real key is not committed with your project. Never place a real key in source code or share it in screenshots or chat.

The application now defaults to `openai/gpt-oss-20b` and automatically selects a compatible Groq model when an older configured model is unavailable.

## Clinical notice

This is an academic screening support tool, not a replacement for an ophthalmologist, slit-lamp examination, or medical advice. Validate the trained model on an independent test set before any clinical use.
