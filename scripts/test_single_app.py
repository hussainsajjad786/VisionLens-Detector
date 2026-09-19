"""Offline smoke checks: python scripts/test_single_app.py."""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["VISIONLENS_SINGLE_APP"] = "1"
from backend.local_client import LocalClient

with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
    first, second = LocalClient(one), LocalClient(two)
    assert first.get("http://local/health").ok
    fake = {"stage": "Normal", "confidence": .9, "probabilities": {"Normal": .9, "Immature": .05, "Mature": .05}, "heatmap": b"test"}
    with patch("backend.main.predict", return_value=fake):
        response = first.post("http://local/predict", data={"patient_name": "Demo", "age": 20, "eye": "Right"}, files={"image": ("eye.png", b"test", "image/png")})
    assert response.ok, response.text
    record = response.json()["id"]
    assert len(first.get("http://local/screenings").json()) == 1
    assert second.get("http://local/screenings").json() == []
    assert first.post(f"http://local/screenings/{record}/feedback", json={"review_status": "Agree", "doctor_name": "Demo doctor", "notes": "Test"}).ok
    assert first.get("http://local/analytics").json()["reviewed"] == 1
    assert first.get(f"http://local/report/{record}").content.startswith(b"%PDF")
    assert first.delete(f"http://local/screenings/{record}").ok
    assert first.get("http://local/screenings").json() == []
    assert first.get("http://local/training-history").ok
    assert first.get("http://local/evaluation").ok
print("PASS: isolated sessions, prediction route (mock ML), feedback, analytics, PDF, deletion, saved metrics")
