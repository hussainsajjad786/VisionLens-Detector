# Streamlit Community Cloud deployment

## 1. Test the single-app version locally

Open PowerShell in this folder (the folder containing this document):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

No Uvicorn process is needed. The original frontend/app.py plus FastAPI workflow still works separately. ML preprocessing, class order, eye validation and Grad-CAM are unchanged. The new entry point runs the existing API routes in-process.

## 2. Upload source code safely

The repository root must be this cataract-ai folder, NOT its parent workspace.
Do not upload .env, .streamlit/secrets.toml, .venv, databases, logs, reports, patient images or the training dataset.

For a new local repository:

```powershell
git init
git branch -M main
git remote add origin https://github.com/Maaskhan/VisionLens-Detector.git
git add .gitignore README.md DEPLOYMENT.md requirements.txt streamlit_app.py backend frontend .streamlit/config.toml models/eye_reference_previous.npz models/training_history.json models/evaluation_result.json
git diff --cached --stat
```

Inspect the staged files for credentials and patient information before committing. Only publish the reference embeddings and model if your dataset/model license permits redistribution. If origin already exists, inspect `git remote -v` instead of adding it again. If the remote already contains commits, reconcile its history; do not force-push.

```powershell
git commit -m "Prepare VisionLens Detector for Streamlit Cloud"
git push -u origin main
```

## 3. Publish the correct model as a release asset

The restored model is approximately 166 MB, so it is excluded from Git commits.
On your GitHub repository, open Releases → Create a new release, choose a tag such as v1.0-model, and attach:

`models/cataract_resnet50_previous.keras`

Do NOT upload cataract_resnet50.keras instead: that is a different model.
Publish the release and copy the asset's direct download link. For a public repository and the example tag, the URL is:

`https://github.com/Maaskhan/VisionLens-Detector/releases/download/v1.0-model/cataract_resnet50_previous.keras`

This URL only works AFTER you publish that exact tag and asset. Private release URLs need a different authenticated hosting arrangement; the current downloader expects an accessible HTTPS URL.

## 4. Create the Streamlit app

Open https://share.streamlit.io/ and select Create app:

- Repository: Maaskhan/VisionLens-Detector
- Branch: main
- Main file: streamlit_app.py
- Advanced settings → Python: 3.12

Paste into Advanced settings → Secrets:

```toml
VISIONLENS_MODEL_URL = "https://github.com/Maaskhan/VisionLens-Detector/releases/download/v1.0-model/cataract_resnet50_previous.keras"
VISIONLENS_MODEL_SHA256 = "d157911f5888f3d24ff461e486a6826cf468fa5ffd35c0cb10350f799f5cb98f"
GROQ_API_KEY = "replace-with-your-new-valid-key"
GROQ_MODEL = "openai/gpt-oss-20b"
```

Use your actual accessible Groq model if different. Never commit these credentials. Revoke any key previously shared in chat. Groq guidance is optional; missing credentials do not disable local image classification.

Click Deploy. First startup downloads and verifies the model. Resource limits may prevent TensorFlow inference on the free tier; successful local startup is not a guarantee of cloud capacity. Inspect Cloud logs if the process restarts or runs out of memory.

## Storage and evaluation limitations

- This public deployment is an academic demo, not a clinical service. Use fictional patient details.
- Each Streamlit browser session receives a separate temporary SQLite database. History, feedback and analytics work within that session. They do not expose the existing local patient database.
- Refreshing/reconnecting can start a new session. Download PDF reports before leaving. Permanent multi-user records require authenticated access and a durable external database, not the free app filesystem.
- Saved training history and evaluation results can be displayed. Running a fresh evaluation requires the original labelled dataset, which is intentionally not uploaded. Saved evaluation results are not new cloud test results.
- Eye-image rejection is a heuristic, not a guarantee that every non-eye picture will be rejected. This deployment does not change its behavior.

## Verify after deployment

1. Open the app and confirm the service is online without port 8000.
2. Upload a known eye image; check class, confidence and Grad-CAM against the local app.
3. Download its PDF, save feedback and check analytics/history.
4. Open an incognito browser: it must not show the first session's records.
5. Test a non-eye image and optional Groq guidance.
6. View saved training/evaluation data. Do not interpret confidence as clinical accuracy.

Official documentation:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
