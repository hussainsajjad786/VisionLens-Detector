from pathlib import Path
import base64
import io
import json
from datetime import datetime
import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from fastapi.responses import Response
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from .database import analytics_summary, delete_screening, get_screening, init_db, patient_history, recent_screenings, save_feedback, save_screening
from .evaluation_service import evaluate_model, last_evaluation
from .model_service import predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# The project .env is the single source of truth for this local deployment.
# This intentionally replaces any stale GROQ_API_KEY inherited from Windows or
# an older PowerShell session.
load_dotenv(PROJECT_ROOT / ".env", override=os.getenv("VISIONLENS_SINGLE_APP") != "1")

app = FastAPI(title="VisionLens Detector API", version="1.0.0")
if os.getenv("VISIONLENS_SINGLE_APP") != "1":
    init_db()


class GuidanceRequest(BaseModel):
    stage: str = Field(pattern="^(Normal|Immature|Mature)$")
    confidence: float = Field(ge=0, le=1)
    language: str = Field(default="English", pattern="^(English|Urdu)$")


class FeedbackRequest(BaseModel):
    review_status: str = Field(pattern="^(Agree|Disagree|Needs review)$")
    doctor_name: str = Field(min_length=2, max_length=100)
    notes: str = Field(default="", max_length=1000)


class EvaluationRequest(BaseModel):
    samples_per_class: int = Field(default=25, ge=10, le=60)


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "online", "service": "VisionLens Detector"}


@app.get("/screenings")
def screenings():
    return recent_screenings()


@app.get("/screenings/{screening_id}")
def screening_details(screening_id: int):
    record = get_screening(screening_id)
    if not record:
        raise HTTPException(404, "Screening not found.")
    return record


@app.get("/patients/history")
def history(patient_id: str = "", patient_name: str = ""):
    return patient_history(patient_id=patient_id, patient_name=patient_name)


@app.get("/analytics")
def analytics():
    return analytics_summary()


@app.post("/screenings/{screening_id}/feedback")
def doctor_feedback(screening_id: int, feedback: FeedbackRequest):
    if not save_feedback(screening_id, feedback.review_status, feedback.doctor_name, feedback.notes):
        raise HTTPException(404, "Screening not found.")
    return get_screening(screening_id)


@app.delete("/screenings/{screening_id}")
def remove_screening(screening_id: int):
    if not delete_screening(screening_id):
        raise HTTPException(404, "Screening record not found.")
    return {"message": "Screening record removed successfully.", "id": screening_id}


@app.get("/evaluation")
def evaluation_result():
    return {"result": last_evaluation()}


@app.get("/training-history")
def training_history():
    history_path = Path(__file__).resolve().parent.parent / "models" / "training_history.json"
    if not history_path.exists():
        return {"records": []}
    try:
        records = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(500, "The saved training history could not be read.")
    return {"records": records}


@app.post("/evaluation")
def run_evaluation(request: EvaluationRequest):
    try:
        return evaluate_model(request.samples_per_class)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(503, str(error))


@app.post("/ai-guidance")
def ai_guidance(request: GuidanceRequest):
    """Generate general post-screening guidance without transmitting an eye image."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(503, "AI guidance is not configured. Add GROQ_API_KEY to the project's .env file, then restart the backend.")
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        prompt = f"""You are the patient-education assistant inside an academic cataract screening website.
The local image model produced this screening output: stage={request.stage}, confidence={request.confidence:.1%}.

Write concise, calm guidance in {request.language}. Use exactly these headings:
1. What this screening result means
2. Suggested next step
3. Image-quality advice for a future screening photo
4. Important safety note

Rules: This is NOT a diagnosis. Do not claim the person has cataract, prescribe medicines, recommend specific surgery, estimate urgency beyond general referral language, or provide emergency instructions beyond advising urgent in-person care for sudden vision loss, severe pain, serious injury, or sudden flashes/floaters. Clearly state that an ophthalmologist must confirm the result. Do not mention that you saw an image; you only received the local model's output."""
        # Llama 3.3 70B was retired for free/developer plans in August 2026.
        # Select a currently accessible model automatically if a stale custom
        # GROQ_MODEL value is still set in the user's terminal.
        available = {item.id for item in client.models.list().data}
        preferred = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        if preferred not in available:
            for candidate in ("openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.6-27b", "groq/compound-mini"):
                if candidate in available:
                    preferred = candidate
                    break
            else:
                raise RuntimeError("No supported text-generation model is enabled for this Groq API key.")
        response = client.chat.completions.create(
            model=preferred,
            messages=[
                {"role": "system", "content": "You provide safe, patient-friendly educational guidance for an academic cataract screening application."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=550,
        )
        return {"guidance": response.choices[0].message.content}
    except Exception as error:
        # Return a short, non-secret error to make local setup debuggable.
        # Groq exceptions do not contain the API key itself.
        message = str(error).replace("\n", " ")[:260]
        raise HTTPException(
            502,
            f"Groq request failed ({type(error).__name__}): {message or 'No error details returned.'}",
        ) from error


@app.post("/predict")
async def prediction(
    image: UploadFile = File(...), patient_name: str = Form(...), patient_id: str = Form(""),
    age: int | None = Form(None), eye: str = Form("Right"),
):
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(400, "Upload a JPG, PNG, or WEBP eye image.")
    raw_image = await image.read()
    try:
        result = predict(raw_image)
    except FileNotFoundError as error:
        raise HTTPException(503, str(error))
    except ValueError as error:
        raise HTTPException(422, str(error))
    record_id = save_screening(patient_name, patient_id, age, eye, result["stage"], result["confidence"])
    result["id"] = record_id
    result["heatmap_base64"] = base64.b64encode(result.pop("heatmap")).decode()
    return result


@app.get("/report/{screening_id}")
def report(screening_id: int):
    r = get_screening(screening_id)
    if not r:
        raise HTTPException(404, "Screening not found")
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    pdf.setFillColor(colors.HexColor("#0F766E")); pdf.rect(0, height - 105, width, 105, fill=1, stroke=0)
    pdf.setFillColor(colors.white); pdf.setFont("Helvetica-Bold", 24); pdf.drawString(46, height - 58, "VisionLens Detector")
    pdf.setFont("Helvetica", 11); pdf.drawString(46, height - 80, "Academic Cataract Screening Report")
    pdf.setFillColor(colors.HexColor("#1E293B")); pdf.setFont("Helvetica-Bold", 15); pdf.drawString(46, height - 145, "Screening summary")
    lines = [("Patient", r["patient_name"]), ("Patient ID", r["patient_id"] or "Not provided"), ("Age", str(r["age"] or "Not provided")), ("Eye", r["eye"]), ("Predicted stage", r["stage"]), ("Model confidence", f"{r['confidence'] * 100:.1f}%"), ("Date", r["created_at"])]
    y = height - 180
    for label, value in lines:
        pdf.setFillColor(colors.HexColor("#1E293B"))
        pdf.setFont("Helvetica-Bold", 11); pdf.drawString(55, y, f"{label}:")
        if label == "Predicted stage":
            stage_color = {"Normal": "#16A34A", "Immature": "#EAB308", "Mature": "#DC2626"}.get(value, "#1E293B")
            pdf.setFillColor(colors.HexColor(stage_color))
            pdf.circle(178, y + 4, 4, fill=1, stroke=0)
            pdf.setFillColor(colors.HexColor("#715500" if value == "Immature" else stage_color))
        pdf.setFont("Helvetica", 11); pdf.drawString(185, y, value); y -= 28
    pdf.setFillColor(colors.HexColor("#B45309")); pdf.setFont("Helvetica-Bold", 11); pdf.drawString(46, 155, "Important clinical notice")
    pdf.setFillColor(colors.HexColor("#334155")); pdf.setFont("Helvetica", 10)
    pdf.drawString(46, 135, "This academic AI output supports screening only and is not a medical diagnosis.")
    pdf.drawString(46, 120, "A qualified ophthalmologist must review all results and determine appropriate treatment.")
    pdf.showPage(); pdf.save(); buffer.seek(0)
    return Response(buffer.read(), media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="visionlens-report-{screening_id}.pdf"'})
