"""Single-process entry point for Streamlit Community Cloud."""
import hashlib
import os
from pathlib import Path
import runpy
import tempfile

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
os.environ["VISIONLENS_SINGLE_APP"] = "1"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
st.set_page_config(page_title="VisionLens Detector", page_icon=":material/visibility:", layout="wide")

try:
    for key in ("GROQ_API_KEY", "GROQ_MODEL", "VISIONLENS_MODEL_URL", "VISIONLENS_MODEL_SHA256"):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except FileNotFoundError:
    pass


@st.cache_resource
def prepare_model():
    path = ROOT / "models" / "cataract_resnet50_previous.keras"
    if path.exists():
        return
    url = os.getenv("VISIONLENS_MODEL_URL", "")
    expected = os.getenv("VISIONLENS_MODEL_SHA256", "").lower()
    if not url.startswith("https://") or len(expected) != 64:
        raise ValueError("Add VISIONLENS_MODEL_URL (HTTPS) and VISIONLENS_MODEL_SHA256 in Streamlit Secrets. See DEPLOYMENT.md.")
    path.parent.mkdir(exist_ok=True)
    temporary = None
    try:
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".download", delete=False) as output:
            temporary = Path(output.name)
            with requests.get(url, stream=True, timeout=(15, 180)) as response:
                response.raise_for_status()
                size = 0
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > 400 * 1024 * 1024:
                        raise ValueError("Model download exceeds 400 MB.")
                    digest.update(chunk)
                    output.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("Model checksum mismatch. Check the release asset and SHA256 value.")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


try:
    with st.spinner("Preparing screening model…"):
        prepare_model()
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except requests.RequestException:
    st.error("Model download failed. Check the release URL and network, then retry.")
    st.stop()

if "_database_directory" not in st.session_state:
    st.session_state["_database_directory"] = tempfile.TemporaryDirectory(prefix="visionlens-session-")
st.info("Academic demo: use fictional patient details only. History and feedback are private to this browser session and are not permanent. Download reports before leaving.")
runpy.run_path(str(ROOT / "frontend" / "app.py"), run_name="__main__")
