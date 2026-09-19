import base64
import os
import requests
import streamlit as st
import pandas as pd
import altair as alt
from PIL import Image

API_URL = os.getenv("VISIONLENS_API_URL", "http://localhost:8000")
if os.getenv("VISIONLENS_SINGLE_APP") == "1":
    from backend.local_client import LocalClient
    requests = LocalClient(st.session_state["_database_directory"].name)
CLASS_COLORS = {"Normal": "#16A34A", "Immature": "#EAB308", "Mature": "#DC2626"}


def class_color():
    return alt.Color("stage:N", title="Class", scale=alt.Scale(
        domain=list(CLASS_COLORS), range=list(CLASS_COLORS.values())))


def class_bars(data, value, horizontal=False):
    category = alt.Y("stage:N", title="Class") if horizontal else alt.X("stage:N", title="Class")
    quantity = alt.X(f"{value}:Q") if horizontal else alt.Y(f"{value}:Q")
    chart = alt.Chart(data).mark_bar().encode(color=class_color(), tooltip=["stage:N", f"{value}:Q"])
    chart = chart.encode(y=category, x=quantity) if horizontal else chart.encode(x=category, y=quantity)
    st.altair_chart(chart, width="stretch")


def stage_table(data):
    frame = pd.DataFrame(data)
    label_column = "stage" if "stage" in frame.columns else "class"
    return frame.style.map(
        lambda value: f"background-color: {CLASS_COLORS[value]}; color: {'#17202A' if value == 'Immature' else '#FFFFFF'}; font-weight: bold" if value in CLASS_COLORS else "",
        subset=[label_column],
    ) if label_column in frame.columns else frame

st.set_page_config(page_title="VisionLens Detector", page_icon=":material/visibility:", layout="wide")
st.html("""
<style>
 :root { --ink: #183747; --muted: #627b88; --line: #d9e7e9; --navy: #17465a; --teal: #168b89; --mist: #f5faf9; --surface: #ffffff; }
 .stApp { background: radial-gradient(ellipse 70% 30% at 96% 0%, #deefed 0%, transparent 74%), radial-gradient(ellipse 45% 18% at 5% 14%, #eaf4f4 0%, transparent 75%), linear-gradient(180deg, #f7fbfa 0%, #fff 38%, #f8fbfb 100%); }
 .block-container { max-width: 1240px; padding-top: 1.65rem; padding-bottom: 4.5rem; }
 .medical-hero { position: relative; isolation: isolate; overflow: hidden; background: linear-gradient(119deg, #123e51 0%, #19596a 53%, #218e8a 100%); padding: 3rem 3.1rem; border-radius: 28px; color: white; box-shadow: 0 24px 60px rgba(20, 70, 82, .19); margin-bottom: 1.05rem; }
 .medical-hero::before { content: ""; position: absolute; z-index: -1; inset: 0; background: linear-gradient(90deg, rgba(255,255,255,.05), transparent 45%); }
 .medical-hero::after { content: ""; position: absolute; right: -5rem; top: -10rem; width: 27rem; height: 27rem; border: 1px solid rgba(255,255,255,.22); border-radius: 50%; box-shadow: 0 0 0 34px rgba(255,255,255,.055), 0 0 0 70px rgba(255,255,255,.035), 0 0 0 110px rgba(255,255,255,.018); }
 .medical-hero h1 { position: relative; z-index: 1; font-size: clamp(2.35rem, 4vw, 3.25rem); line-height: 1; margin: 0; letter-spacing: -.052em; font-weight: 720; }
 .medical-hero p { position: relative; z-index: 1; margin: .85rem 0 0; max-width: 38rem; font-size: 1.08rem; line-height: 1.65; opacity: .92; }
 .medical-hero span { position: relative; z-index: 1; display: inline-flex; align-items: center; gap: .4rem; margin-top: 1.4rem; border: 1px solid rgba(255,255,255,.26); background: rgba(255,255,255,.1); backdrop-filter: blur(10px); padding: .48rem .85rem; border-radius: 999px; font-size: .78rem; letter-spacing: .045em; text-transform: uppercase; }
 .medical-hero span::before { content: ""; width: .45rem; height: .45rem; background: #a7f0dc; border-radius: 50%; box-shadow: 0 0 0 4px rgba(167,240,220,.16); }
 .journey { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin: 0 0 1.8rem; }
 .journey-item { display: flex; align-items: center; gap: .85rem; padding: 1rem 1.1rem; background: rgba(255,255,255,.86); border: 1px solid rgba(205,225,226,.95); border-radius: 16px; color: #627b88; font-size: .86rem; box-shadow: 0 8px 20px rgba(27,78,84,.035); transition: transform .2s ease, box-shadow .2s ease; }
 .journey-item:hover { transform: translateY(-2px); box-shadow: 0 14px 25px rgba(27,78,84,.08); }
 .journey-number { display: grid; place-items: center; flex: 0 0 auto; width: 2rem; height: 2rem; border-radius: 50%; background: linear-gradient(135deg, #e2f5ef, #d8eeee); color: #087675; font-weight: 720; font-size: .8rem; }
 .journey-item strong { display: block; margin-bottom: .1rem; color: #183747; font-size: .92rem; }
 [data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255,255,255,.9); border: 1px solid var(--line) !important; border-radius: 20px !important; box-shadow: 0 10px 28px rgba(28, 74, 81, .055); }
 [data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 20px; }
 [data-testid="stAlert"] { border-radius: 15px; border: 1px solid rgba(19, 105, 118, .14); padding: .85rem 1rem; }
 [data-testid="stMetric"] { background: linear-gradient(135deg, #f5fbfa, #edf7f7); border: 1px solid #d4e7e6 !important; border-radius: 15px; padding: .9rem 1rem; }
 [data-testid="stMetricLabel"] { color: var(--muted); font-size: .83rem; letter-spacing: .015em; }
 [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 15px; overflow: hidden; }
 .stButton > button, .stDownloadButton > button { min-height: 2.9rem; border-radius: 12px; font-weight: 650; letter-spacing: .005em; transition: transform .18s ease, box-shadow .18s ease; }
 .stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); }
 .stButton > button[kind="primary"] { border: 0; background: linear-gradient(105deg, #167d7c, #209a93); box-shadow: 0 9px 20px rgba(18,132,126,.2); }
 .stButton > button[kind="primary"]:hover { background: linear-gradient(105deg, #116e70, #168a87); box-shadow: 0 12px 24px rgba(18,132,126,.25); }
 .stDownloadButton > button { border-color: #bcdad9; color: #176b6b; background: #f7fcfc; }
 [data-testid="stFileUploaderDropzone"] { min-height: 11.5rem; background: linear-gradient(135deg, #f6fbfb, #eef8f7); border: 1.5px dashed #96c8c6; border-radius: 16px; }
 [data-testid="stFileUploaderDropzone"]:hover { background: #e8f7f5; border-color: #168b89; }
 [data-baseweb="input"] > div, [data-baseweb="select"] > div { border-radius: 11px !important; border-color: #cbdfe0 !important; background: #fcfefe !important; }
 [data-baseweb="input"] > div:focus-within { border-color: #168b89 !important; box-shadow: 0 0 0 3px rgba(22,139,137,.1) !important; }
 [data-testid="stSegmentedControl"] { background: #eef6f5; border-radius: 12px; padding: .18rem; }
 h2, h3 { color: var(--ink); letter-spacing: -.028em; }
 [data-testid="stCaptionContainer"] { color: var(--muted); line-height: 1.55; }
 hr { border-color: var(--line); }
 @media (max-width: 700px) { .block-container { padding: .9rem 1rem 2.5rem; } .medical-hero { padding: 2.15rem 1.5rem; border-radius: 21px; } .medical-hero h1 { font-size: 2.35rem; } .journey { grid-template-columns: 1fr; margin-bottom: 1.4rem; } }
</style>
<section class="medical-hero"><h1>VisionLens Detector</h1><p>Calm, thoughtful cataract screening support for clearer decisions and compassionate eye care.</p><span>Academic clinical-support platform</span></section>
<section class="journey"><div class="journey-item"><div class="journey-number">1</div><div><strong>Upload</strong>One clear eye photograph</div></div><div class="journey-item"><div class="journey-number">2</div><div><strong>Screen</strong>Local AI analysis</div></div><div class="journey-item"><div class="journey-number">3</div><div><strong>Review</strong>Guidance and report</div></div></section>
""")

try:
    online = requests.get(f"{API_URL}/health", timeout=2).ok
except requests.RequestException:
    online = False

with st.container(horizontal=True, horizontal_alignment="distribute"):
    st.caption(":material/health_and_safety: Secure academic screening workspace")
    if online:
        st.badge("Screening service online", icon=":material/check_circle:", color="green")
    else:
        st.badge("Screening service offline", icon=":material/error:", color="red")

left, right = st.columns([1.05, 1])
with left:
    with st.container(border=True):
     st.subheader("New screening", icon=":material/add_a_photo:")
    st.caption("Enter patient details, then upload one clear close-up photograph of a single eye. Other images are rejected automatically.")
    with st.form("screening", clear_on_submit=True):
        patient_name = st.text_input("Patient name *", placeholder="e.g., Ayesha Khan")
        a, b = st.columns(2)
        with a: patient_id = st.text_input("Patient ID", placeholder="e.g., VL-001")
        with b: age = st.number_input("Age", min_value=0, max_value=120, value=45)
        eye = st.segmented_control("Eye examined", ["Right", "Left"], default="Right")
        uploaded = st.file_uploader("Eye image *", type=["jpg", "jpeg", "png", "webp"], help="Only close-up eye photographs are accepted. Photos of people, objects, documents, or screenshots are rejected.")
        submitted = st.form_submit_button("Run AI screening", type="primary", width="stretch", icon=":material/biotech:")
    if submitted:
        if not patient_name.strip() or uploaded is None:
            st.error("Patient name and an eye image are required.")
        elif not online:
            st.error("The API is offline. Start FastAPI on port 8000, then try again.")
        else:
            st.image(Image.open(uploaded), caption="Uploaded image", width="stretch")
            uploaded.seek(0)
            with st.spinner("Preprocessing image and analysing the lens..."):
                response = requests.post(f"{API_URL}/predict", data={"patient_name": patient_name, "patient_id": patient_id, "age": age, "eye": eye}, files={"image": (uploaded.name, uploaded.getvalue(), uploaded.type)}, timeout=60)
            if response.ok:
                st.session_state["result"] = response.json()
            else:
                try:
                    detail = response.json().get("detail", "Screening could not be completed.")
                except requests.JSONDecodeError:
                    detail = response.text.strip() or "The backend returned an empty error response. Check the backend terminal for details."
                st.error(f"Screening could not be completed: {detail}")

with right:
    with st.container(border=True):
     st.subheader("Screening result", icon=":material/analytics:")
    result = st.session_state.get("result")
    if result:
        stage = result["stage"]
        color = {"Normal": "green", "Immature": "yellow", "Mature": "red"}[stage]
        st.badge(f"{stage} cataract classification", icon=":material/visibility:", color=color)
        st.metric("Model confidence", f"{result['confidence'] * 100:.1f}%", border=True)
        st.caption("Class probabilities")
        class_bars(pd.DataFrame([{"stage": name, "probability": probability} for name, probability in result["probabilities"].items()]), "probability", horizontal=True)
        heatmap = base64.b64decode(result["heatmap_base64"])
        st.image(heatmap, caption="Grad-CAM: areas that contributed most to the classification", width="stretch")
        guidance = {"Normal": "Routine eye follow-up is advised.", "Immature": "Recommend ophthalmology review and periodic monitoring.", "Mature": "Prioritize ophthalmology referral for clinical assessment and treatment planning."}[stage]
        st.info(guidance)
        pdf = requests.get(f"{API_URL}/report/{result['id']}", timeout=15)
        if pdf.ok:
            st.download_button("Download PDF report", pdf.content, f"visionlens-report-{result['id']}.pdf", "application/pdf", width="stretch", icon=":material/download:")
        st.subheader("AI guidance", icon=":material/psychology:")
        st.caption("Generates patient-friendly next-step guidance from the local screening result. The uploaded eye image is not sent to the LLM.")
        language = st.segmented_control("Guidance language", ["English", "Urdu"], default="English", key="guidance_language")
        if st.button("Generate AI guidance", icon=":material/auto_awesome:", width="stretch"):
            with st.spinner("Preparing guidance from the screening output..."):
                guidance_response = requests.post(
                    f"{API_URL}/ai-guidance",
                    json={"stage": stage, "confidence": result["confidence"], "language": language},
                    timeout=45,
                )
            if guidance_response.ok:
                st.session_state["ai_guidance"] = guidance_response.json()["guidance"]
            else:
                try:
                    message = guidance_response.json().get("detail", "AI guidance could not be generated.")
                except requests.JSONDecodeError:
                    message = "AI guidance could not be generated. Check the backend terminal."
                st.error(message)
        if guidance := st.session_state.get("ai_guidance"):
            st.info(guidance, icon=":material/lightbulb:")
    else:
        st.info("Your result, confidence score, Grad-CAM explanation, and PDF report will appear here.")

st.warning("Clinical safety notice: VisionLens Detector is an academic screening-support system. It does not replace examination or diagnosis by a qualified ophthalmologist.", icon=":material/medical_services:")
st.subheader("Recent screening activity", icon=":material/history:")
if online:
    try:
        rows = requests.get(f"{API_URL}/screenings", timeout=5).json()
        if rows:
            st.dataframe(stage_table(rows), width="stretch", hide_index=True)
            with st.expander("Remove a screening record", icon=":material/delete:"):
                record_options = {
                    f"#{row['id']} — {row['patient_name']} ({row.get('patient_id') or 'No patient ID'}) — {row['created_at']}": row
                    for row in rows
                }
                selected_record_label = st.selectbox(
                    "Select the record to remove",
                    options=list(record_options),
                    key="delete_screening_record",
                )
                selected_record = record_options[selected_record_label]
                st.warning(
                    "This permanently removes the selected screening record and its doctor feedback. "
                    "It cannot be undone."
                )
                confirmed_deletion = st.checkbox(
                    "I understand that this action cannot be undone.",
                    key="confirm_delete_screening",
                )
                if st.button(
                    "Remove selected record",
                    icon=":material/delete:",
                    type="primary",
                    disabled=not confirmed_deletion,
                    key="delete_screening_button",
                ):
                    try:
                        response = requests.delete(
                            f"{API_URL}/screenings/{selected_record['id']}", timeout=15
                        )
                        response.raise_for_status()
                        st.session_state.pop("patient_history_results", None)
                        st.success("The screening record was removed.")
                        st.rerun()
                    except requests.RequestException as exc:
                        st.error(f"Could not remove the screening record: {friendly_api_error(exc)}")
        else: st.caption("No screening records yet.")
    except requests.RequestException: st.caption("Activity will appear when the API is available.")
else:
    st.caption("API offline - start the backend to view saved screening activity.")

st.subheader("Clinical workspace", icon=":material/clinical_notes:")
st.caption("Patient records, doctor review, screening trends, and reproducible model evaluation.")
history_tab, feedback_tab, analytics_tab, evaluation_tab = st.tabs([
    ":material/person_search: Patient history",
    ":material/rate_review: Doctor feedback",
    ":material/monitoring: Analytics",
    ":material/fact_check: Model evaluation",
])

with history_tab:
    with st.container(border=True):
        st.subheader("Find a patient's screening history")
        search_id, search_name = st.columns(2)
        with search_id:
            history_id = st.text_input("Patient ID", placeholder="e.g., VL-001", key="history_patient_id")
        with search_name:
            history_name = st.text_input("Patient name", placeholder="e.g., Ayesha Khan", key="history_patient_name")
        if st.button("Search patient history", icon=":material/search:", key="history_search"):
            if not history_id.strip() and not history_name.strip():
                st.warning("Enter a patient ID or patient name to search.")
            elif not online:
                st.error("The API is offline. Start the backend before searching.")
            else:
                try:
                    response = requests.get(
                        f"{API_URL}/patients/history",
                        params={"patient_id": history_id, "patient_name": history_name}, timeout=15,
                    )
                    response.raise_for_status()
                    st.session_state["patient_history_results"] = response.json()
                except requests.RequestException:
                    st.error("Patient history could not be loaded. Check the backend terminal.")
        records = st.session_state.get("patient_history_results", [])
        if records:
            st.success(f"Found {len(records)} screening record(s).")
            table = pd.DataFrame(records)[["id", "patient_name", "patient_id", "age", "eye", "stage", "confidence", "review_status", "created_at"]]
            st.dataframe(
                stage_table(table),
                width="stretch",
                hide_index=True,
                column_config={"confidence": st.column_config.NumberColumn("Confidence", format="percent")},
            )
            trend = table.copy()
            trend["created_at"] = pd.to_datetime(trend["created_at"])
            st.caption("Recorded screening confidence over time")
            st.altair_chart(alt.Chart(trend).mark_line(point=True).encode(x="created_at:T", y="confidence:Q", color=class_color(), tooltip=["created_at:T", "stage:N", "confidence:Q"]), width="stretch")
        elif "patient_history_results" in st.session_state:
            st.info("No screening history was found for this search.")

with feedback_tab:
    with st.container(border=True):
        st.subheader("Doctor review and feedback")
        st.caption("Doctor feedback is stored separately and never changes the original AI classification.")
        if not online:
            st.error("The API is offline. Start the backend before recording feedback.")
        else:
            try:
                review_rows = requests.get(f"{API_URL}/screenings", timeout=15).json()
            except requests.RequestException:
                review_rows = []
                st.error("Screening records could not be loaded.")
            if review_rows:
                choices = {
                    f"#{row['id']} - {row['patient_name']} | {row['stage']} | {row['created_at']}": row
                    for row in review_rows
                }
                selected_label = st.selectbox("Select a screening", list(choices), key="feedback_record")
                selected = choices[selected_label]
                status_colour = {"Agree": "green", "Disagree": "red", "Needs review": "orange", "Pending": "blue"}
                st.badge(f"Current review: {selected['review_status']}", color=status_colour.get(selected["review_status"], "blue"))
                with st.form("doctor_feedback_form"):
                    doctor_name = st.text_input("Doctor / reviewer name *", value=selected.get("doctor_name", ""), placeholder="e.g., Dr. Ahmed")
                    review_status = st.segmented_control("Review decision *", ["Agree", "Disagree", "Needs review"], default="Needs review")
                    doctor_notes = st.text_area("Clinical notes", value=selected.get("doctor_notes", ""), placeholder="Add review comments, limitations, or follow-up notes.", max_chars=1000)
                    save_review = st.form_submit_button("Save doctor feedback", type="primary", icon=":material/save:", width="stretch")
                if save_review:
                    if len(doctor_name.strip()) < 2:
                        st.error("Enter the doctor or reviewer name.")
                    else:
                        try:
                            response = requests.post(
                                f"{API_URL}/screenings/{selected['id']}/feedback",
                                json={"review_status": review_status, "doctor_name": doctor_name, "notes": doctor_notes}, timeout=15,
                            )
                            response.raise_for_status()
                            st.success("Doctor feedback saved. Refresh this tab to see the updated record.")
                        except requests.RequestException:
                            st.error("Feedback could not be saved. Check the backend terminal.")
            else:
                st.info("No screening records are available for review yet.")

with analytics_tab:
    with st.container(border=True):
        st.subheader("Screening analytics")
        if not online:
            st.error("The API is offline. Start the backend to view analytics.")
        else:
            try:
                summary = requests.get(f"{API_URL}/analytics", timeout=15).json()
                metric_a, metric_b, metric_c = st.columns(3)
                metric_a.metric("Total screenings", summary["total"], border=True)
                metric_b.metric("Doctor reviewed", summary["reviewed"], border=True)
                metric_c.metric("Pending review", summary["pending"], border=True)
                chart_left, chart_right = st.columns(2)
                with chart_left:
                    st.markdown("**Classification distribution**")
                    stage_data = pd.DataFrame([
                        {"stage": stage, "screenings": summary["stages"].get(stage, 0)}
                        for stage in ["Normal", "Immature", "Mature"]
                    ])
                    class_bars(stage_data, "screenings")
                with chart_right:
                    st.markdown("**Screenings over time**")
                    timeline = pd.DataFrame(summary["timeline"])
                    if timeline.empty:
                        st.caption("Screening volume will appear after records are created.")
                    else:
                        st.line_chart(timeline, x="date", y="screenings", color="#4F86A8")
            except requests.RequestException:
                st.error("Analytics could not be loaded. Check the backend terminal.")

with evaluation_tab:
    with st.container(border=True):
        st.subheader("Model evaluation")
        st.caption("Run a reproducible internal evaluation on a balanced sample from your local labelled dataset. This is for academic demonstration, not external clinical validation.")
        st.markdown("#### Training accuracy and loss history")
        if online:
            try:
                training_records = requests.get(f"{API_URL}/training-history", timeout=15).json().get("records", [])
                if training_records:
                    training_df = pd.DataFrame(training_records)
                    run_options = list(reversed(training_df["run_id"].drop_duplicates().tolist()))
                    selected_run = st.selectbox("Training run", run_options, key="training_history_run")
                    run_history = training_df[training_df["run_id"] == selected_run].copy().reset_index(drop=True)
                    run_history["training_step"] = run_history.index + 1
                    history_left, history_right = st.columns(2)
                    with history_left:
                        st.markdown("**Accuracy per epoch**")
                        accuracy_chart = run_history.rename(columns={"accuracy": "Training accuracy", "val_accuracy": "Validation accuracy"})
                        st.line_chart(
                            accuracy_chart,
                            x="training_step",
                            y=["Training accuracy", "Validation accuracy"],
                            color=["#168B89", "#4F86A8"],
                        )
                    with history_right:
                        st.markdown("**Loss per epoch**")
                        loss_chart = run_history.rename(columns={"loss": "Training loss", "val_loss": "Validation loss"})
                        st.line_chart(
                            loss_chart,
                            x="training_step",
                            y=["Training loss", "Validation loss"],
                            color=["#168B89", "#C87958"],
                        )
                    st.dataframe(
                        run_history[["phase", "epoch", "loss", "accuracy", "val_loss", "val_accuracy", "recorded_at"]],
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "loss": st.column_config.NumberColumn("Training loss", format="%.4f"),
                            "accuracy": st.column_config.NumberColumn("Training accuracy", format="percent"),
                            "val_loss": st.column_config.NumberColumn("Validation loss", format="%.4f"),
                            "val_accuracy": st.column_config.NumberColumn("Validation accuracy", format="percent"),
                            "recorded_at": "Recorded at (UTC)",
                        },
                    )
                else:
                    st.info("No saved training history yet. Retrain once to create the accuracy and validation-loss record.")
            except requests.RequestException:
                st.caption("Training history will appear when the backend is available.")
        sample_count = st.slider("Images per class", min_value=10, max_value=60, value=25, step=5, key="evaluation_samples")
        if st.button("Run model evaluation", type="primary", icon=":material/play_circle:", key="run_evaluation"):
            if not online:
                st.error("The API is offline. Start the backend before evaluation.")
            else:
                with st.spinner("Running local model evaluation. This may take a little time on CPU..."):
                    try:
                        response = requests.post(f"{API_URL}/evaluation", json={"samples_per_class": sample_count}, timeout=240)
                        response.raise_for_status()
                        st.session_state["evaluation_result"] = response.json()
                    except requests.RequestException:
                        st.error("Model evaluation could not be completed. Confirm the trained model and dataset are available, then check the backend terminal.")
        if "evaluation_result" not in st.session_state and online:
            try:
                saved = requests.get(f"{API_URL}/evaluation", timeout=15).json().get("result")
                if saved:
                    st.session_state["evaluation_result"] = saved
            except requests.RequestException:
                pass
        evaluation = st.session_state.get("evaluation_result")
        if evaluation:
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{evaluation['accuracy']:.1%}", border=True)
            m2.metric("Macro F1 score", f"{evaluation['macro_f1']:.1%}", border=True)
            m3.metric("Mean confidence", f"{evaluation['mean_confidence']:.1%}", border=True)
            st.caption(f"Evaluation sample: {evaluation['sample_size']} images | Last run: {evaluation['evaluated_at']}")
            evaluation_left, evaluation_right = st.columns(2)
            with evaluation_left:
                st.markdown("**Per-class metrics**")
                per_class = pd.DataFrame(evaluation["per_class"])
                st.dataframe(
                    stage_table(per_class),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "precision": st.column_config.NumberColumn("Precision", format="percent"),
                        "recall": st.column_config.NumberColumn("Recall", format="percent"),
                        "f1": st.column_config.NumberColumn("F1 score", format="percent"),
                    },
                )
            with evaluation_right:
                st.markdown("**Confusion matrix**")
                matrix = pd.DataFrame(evaluation["confusion_matrix"], index=evaluation["classes"], columns=evaluation["classes"])
                matrix.index.name, matrix.columns.name = "Actual", "Predicted"
                st.dataframe(matrix, width="stretch")
            st.warning(evaluation["notice"], icon=":material/info:")
        else:
            st.info("No model evaluation has been run yet. Select a sample size and use the button above.")

st.subheader("About cataracts", icon=":material/menu_book:")
st.caption("Educational information for patients, families, and screening teams.")

with st.container(border=True):
    st.markdown("### What is a cataract?")
    st.write(
        "A cataract is clouding of the eye's natural lens. The lens normally focuses light on the retina; "
        "when it becomes cloudy, vision can gradually become blurred, hazy, less colourful, or difficult in bright light. "
        "Cataracts are common with ageing, but they are treatable."
    )

info_left, info_right = st.columns(2)
with info_left:
    with st.container(border=True):
        st.markdown("### Common symptoms")
        st.markdown("""
- Blurry, cloudy, or hazy vision
- Colours appearing faded or yellowed
- Glare or halos around lights
- Difficulty seeing at night, especially while driving
- Double vision in one eye
- Frequent changes to glasses prescription
""")
    with st.container(border=True):
        st.markdown("### Causes and risk factors")
        st.write("Most cataracts develop with age as proteins in the lens change. Risk can also increase with diabetes, smoking, heavy alcohol use, long-term steroid medication, previous eye injury or surgery, family history, radiation exposure, and prolonged ultraviolet sunlight exposure.")

with info_right:
    with st.container(border=True):
        st.markdown("### Cataract stages in VisionLens Detector")
        st.markdown("""
- :green[**Normal:**] no cataract pattern detected by this academic model; maintain routine eye examinations.
- :yellow[**Immature:**] early or partial lens clouding pattern; arrange an ophthalmology assessment and monitoring.
- :red[**Mature:**] advanced lens clouding pattern; prioritize a qualified ophthalmology review for treatment planning.
""")
        st.caption("These labels are screening outputs, not a final medical diagnosis.")
    with st.container(border=True):
        st.markdown("### Diagnosis and treatment")
        st.write("An ophthalmologist diagnoses cataract through a complete eye examination, commonly including a dilated eye exam. Early symptoms may sometimes be managed with better lighting, anti-glare glasses, or an updated prescription. Surgery is the only treatment that removes a cataract: the clouded lens is replaced with an artificial intraocular lens when vision affects daily life or another eye condition needs treatment.")

with st.container(border=True):
    st.markdown("### Eye-health and prevention steps")
    prevention_a, prevention_b, prevention_c = st.columns(3)
    with prevention_a:
        st.markdown(":material/wb_sunny: **Protect from UV**")
        st.caption("Wear UV-blocking sunglasses and a brimmed hat outdoors.")
    with prevention_b:
        st.markdown(":material/no_smoking: **Avoid smoking**")
        st.caption("Stopping smoking supports eye and general health.")
    with prevention_c:
        st.markdown(":material/restaurant: **Support overall health**")
        st.caption("Manage diabetes, protect eyes from injury, and eat a balanced diet rich in fruits and vegetables.")

st.warning(
    "Seek urgent eye care for sudden vision loss, severe eye pain, a new curtain/shadow in vision, a serious eye injury, or sudden flashes and many new floaters. These may be signs of conditions other than cataract.",
    icon=":material/emergency:",
)
st.caption("Sources: [National Eye Institute - Cataracts](https://www.nei.nih.gov/eye-health-information/eye-conditions-and-diseases/cataracts) and [World Health Organization - Blindness and vision impairment](https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment). This page is educational and does not replace care from a qualified eye specialist.")
