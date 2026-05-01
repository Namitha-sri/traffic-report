import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time

from detection_service import DetectionService
from analysis_service import AnalysisService
from database_service import DatabaseService
from use_case_service import UseCaseService
from config import COCO_CLASSES, COCO_NAME_TO_ID, BUILT_IN_USE_CASES, DEFAULT_USE_CASE

# ------------------------------------------------------------------
# Settings storage (same pattern as v1)
# ------------------------------------------------------------------
settings = {
    "yolo_endpoint": os.getenv("YOLO_ENDPOINT", "https://your-yolo-endpoint.com/predict"),
    "yolo_api_key": os.getenv("YOLO_API_KEY", "your-yolo-api-key-here"),
    "qwen_endpoint": os.getenv("QWEN_ENDPOINT", "https://your-qwen-endpoint.com/v1/chat/completions"),
    "qwen_api_key": os.getenv("QWEN_API_KEY", "your-qwen-api-key-here"),
    "db_url": os.getenv("DB_URL", "localhost"),
    "db_user": os.getenv("DB_USER", "postgres"),
    "db_password": os.getenv("DB_PASSWORD", ""),
    "db_name": os.getenv("DB_NAME", "traffic_analysis"),
}


def update_settings(yolo_endpoint, yolo_api_key, qwen_endpoint, qwen_api_key,
                    db_url=None, db_user=None, db_password=None, db_name=None):
    settings["yolo_endpoint"] = yolo_endpoint
    settings["yolo_api_key"] = yolo_api_key
    settings["qwen_endpoint"] = qwen_endpoint
    settings["qwen_api_key"] = qwen_api_key

    if db_url is not None:
        settings["db_url"] = db_url
    if db_user is not None:
        settings["db_user"] = db_user
    if db_password is not None:
        settings["db_password"] = db_password
    if db_name is not None:
        settings["db_name"] = db_name

    os.environ["YOLO_ENDPOINT"] = yolo_endpoint.replace("/predict", "") if yolo_endpoint else ""
    os.environ["YOLO_API_KEY"] = yolo_api_key if yolo_api_key else ""
    os.environ["QWEN_ENDPOINT"] = qwen_endpoint if qwen_endpoint else ""
    os.environ["QWEN_API_KEY"] = qwen_api_key if qwen_api_key else ""
    os.environ["DB_URL"] = settings["db_url"]
    os.environ["DB_USER"] = settings["db_user"]
    os.environ["DB_PASSWORD"] = settings["db_password"]
    os.environ["DB_NAME"] = settings["db_name"]

    for s in ("detector", "analyzer", "db_service", "use_case_service"):
        if s in st.session_state:
            del st.session_state[s]

    return "Settings updated successfully!"


# ------------------------------------------------------------------
# Lazy service getters
# ------------------------------------------------------------------
def get_detector():
    if 'detector' not in st.session_state:
        st.session_state.detector = DetectionService()
    return st.session_state.detector


def get_analyzer():
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = AnalysisService()
    return st.session_state.analyzer


def get_db_service():
    if 'db_service' not in st.session_state:
        st.session_state.db_service = DatabaseService()
    return st.session_state.db_service


def get_use_case_service():
    if 'use_case_service' not in st.session_state:
        st.session_state.use_case_service = UseCaseService()
    return st.session_state.use_case_service


def get_active_use_case():
    """Return the currently selected use case dict."""
    ucs = get_use_case_service()
    key = st.session_state.get("active_use_case_key", DEFAULT_USE_CASE)
    uc = ucs.get_use_case(key)
    if uc is None:
        # Fall back to default if the chosen one was deleted
        uc = ucs.get_use_case(DEFAULT_USE_CASE)
        st.session_state.active_use_case_key = DEFAULT_USE_CASE
    return uc


def _prompt_with_extras(use_case):
    """
    If the use case has extra_class_names (objects not in COCO like 'robot'),
    prepend a note to the Qwen prompt so the VLM knows to look for them.
    YOLO can't box them, but Qwen can describe them.
    """
    prompt = use_case["prompt"]
    extras = use_case.get("extra_class_names", []) or []
    if not extras:
        return prompt
    note = (
        "Note: Please also look for and describe these specific objects in the scene "
        "(even if the object detector did not report them): "
        f"{', '.join(extras)}.\n\n"
    )
    return note + prompt


# ------------------------------------------------------------------
# Image + video processing (use-case aware)
# ------------------------------------------------------------------
def process_image(image, use_case):
    try:
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        detector = get_detector()
        analyzer = get_analyzer()

        yolo_valid, yolo_msg = detector.validate_endpoint()
        detections = detector.detect_objects(pil_image, classes_filter=use_case["classes"])
        annotated_image = detector.draw_detections(pil_image, detections) if detections else pil_image
        analysis = analyzer.analyze_scene(pil_image, detections, _prompt_with_extras(use_case))
        results_text = format_analysis_results(detections, analysis, use_case, yolo_valid, yolo_msg)
        return annotated_image, results_text
    except Exception as e:
        return image, f"## Error Processing Image\n\n**Error:** {str(e)}"


def process_video(video_path, use_case, interval_seconds=3):
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * interval_seconds)

        results = []
        frame_count = 0
        detector = get_detector()
        analyzer = get_analyzer()

        # Build the prompt once (includes the "also look for X" preamble if needed)
        enriched_prompt = _prompt_with_extras(use_case)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb)
                detections = detector.detect_objects(pil_frame, classes_filter=use_case["classes"])
                analysis = analyzer.analyze_scene(pil_frame, detections, enriched_prompt)
                annotated = detector.draw_detections(pil_frame, detections) if detections else pil_frame
                results.append({
                    "frame": frame_count,
                    "timestamp": frame_count / fps,
                    "detections": detections,
                    "analysis": analysis,
                    "annotated_frame": annotated
                })
                if len(results) >= 10:
                    break
            frame_count += 1

        cap.release()

        annotated_frames = [
            {"frame": r["annotated_frame"],
             "timestamp": r["timestamp"],
             "frame_number": r["frame"]}
            for r in results
        ]
        return annotated_frames, f"**Analysis Summary:** Processed {len(results)} frames at {interval_seconds}-second intervals using use case: {use_case['name']}\n\n", results
    except Exception as e:
        return None, f"Error processing video: {str(e)}", []


def format_analysis_results(detections, analysis, use_case, yolo_valid=True, yolo_msg=""):
    """Format results for display. Detection labels adapt to the use case."""
    text = f"### {use_case['icon']} {use_case['name']}\n\n"

    if not yolo_valid:
        text += f"⚠️ **YOLO Detection API:** {yolo_msg}\n\n"

    text += "### Detected Objects\n"
    if detections:
        counts = {}
        for d in detections:
            counts[d['class'].title()] = counts.get(d['class'].title(), 0) + 1
        for obj, c in counts.items():
            text += f"- **{obj}:** {c}\n"
    else:
        text += "- No relevant objects detected for this use case\n"

    text += "\n### AI Report\n"
    content = analysis.get('content', 'Analysis not available')
    text += content if isinstance(content, str) else str(content)
    return text


def format_single_frame_results(frame_result, frame_index, use_case):
    if not frame_result:
        return "No analysis results available"
    ts = frame_result['timestamp']
    text = f"## {use_case['icon']} Frame {frame_index} - {ts:.1f}s\n\n### Detected Objects\n"
    detections = frame_result['detections']
    if detections:
        counts = {}
        for d in detections:
            counts[d['class']] = counts.get(d['class'], 0) + 1
        for c, n in counts.items():
            text += f"- **{c.title()}:** {n}\n"
    else:
        text += "- No objects detected\n"
    analysis = frame_result['analysis']
    content = analysis.get('content', '')
    if isinstance(content, str):
        text += "\n### AI Analysis Report\n" + content
    return text


def export_to_database(video_name, use_case_key, analysis_results):
    try:
        db = get_db_service()
        return db.save_analysis_results(video_name, use_case_key, analysis_results)
    except Exception as e:
        return False, f"Error exporting to database: {str(e)}"


# ==================================================================
# UI
# ==================================================================
def create_interface():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True

    st.set_page_config(
        page_title="Smart Video Analytics",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎥 Smart Video Analytics")
    st.markdown("*Configurable video analysis for any use case — traffic, hospital, workshop, or your own.*")

    # --------------------------------------------------------------
    # Active use case selector (persistent across tabs)
    # --------------------------------------------------------------
    ucs = get_use_case_service()

    # Make sure the use_cases table exists (DB might not be configured yet)
    try:
        ucs.create_table_if_not_exists()
    except Exception:
        pass

    all_use_cases = ucs.get_all_use_cases()
    uc_labels = [f"{uc['icon']} {uc['name']}" + (" (built-in)" if uc['is_builtin'] else " (custom)") for uc in all_use_cases]
    uc_keys = [uc['key'] for uc in all_use_cases]

    if "active_use_case_key" not in st.session_state:
        st.session_state.active_use_case_key = DEFAULT_USE_CASE

    try:
        current_index = uc_keys.index(st.session_state.active_use_case_key)
    except ValueError:
        current_index = 0
        st.session_state.active_use_case_key = uc_keys[0]

    col_uc1, col_uc2 = st.columns([3, 1])
    with col_uc1:
        selected_label = st.selectbox("**Active Use Case**", uc_labels, index=current_index,
                                      help="Pick the scenario you're analyzing — this controls what YOLO filters for and what Qwen is asked.")
        st.session_state.active_use_case_key = uc_keys[uc_labels.index(selected_label)]
    with col_uc2:
        if st.button("🔍 Check API & DB Status"):
            with st.spinner("Checking..."):
                detector = get_detector()
                yolo_valid, yolo_msg = detector.validate_endpoint()
                if yolo_valid:
                    st.success(f"✅ YOLO: {yolo_msg}")
                else:
                    st.error(f"❌ YOLO: {yolo_msg}")
                db = get_db_service()
                db_valid, db_msg = db.validate_connection()
                if db_valid:
                    st.success(f"✅ Database: {db_msg}")
                else:
                    st.error(f"❌ Database: {db_msg}")

    active_uc = get_active_use_case()
    st.info(f"{active_uc['icon']} **Current:** {active_uc['name']} — {active_uc['description']}")
    st.markdown("---")

    # --------------------------------------------------------------
    # Tabs
    # --------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "🎬 Video Analysis", "⚙️ Settings"])

    # --------------- Tab 1: Image Analysis ---------------
    with tab1:
        st.header("Image Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Upload Image")
            image_input = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], key="image_upload")
            if image_input and st.button("Analyze Image", type="primary"):
                with st.spinner(f"Analyzing as {active_uc['name']}..."):
                    try:
                        pil_image = Image.open(image_input)
                        annotated, results_text = process_image(pil_image, active_uc)
                        st.session_state.image_results = (annotated, results_text)
                    except Exception as e:
                        st.error(f"Error processing image: {e}")
        with col2:
            st.subheader("Results")
            if hasattr(st.session_state, 'image_results') and st.session_state.image_results:
                annotated, results_text = st.session_state.image_results
                if annotated:
                    st.image(annotated, caption="Detection Results", use_column_width=True)
                if results_text:
                    st.markdown(results_text)

    # --------------- Tab 2: Video Analysis ---------------
    with tab2:
        st.header("Video Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Upload Video")
            video_input = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov', 'mkv'], key="video_upload")

            video_duration = None
            if video_input:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(video_input.getvalue())
                    tmp_path = tmp.name
                try:
                    cap = cv2.VideoCapture(tmp_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_duration = int(total / fps) if fps > 0 else 30
                    cap.release()
                finally:
                    os.unlink(tmp_path)

            if video_duration:
                interval_seconds = st.number_input(
                    "Analysis Interval (seconds)",
                    min_value=1, max_value=video_duration, value=3, step=1,
                    help=f"Extract frames every N seconds. Video duration: {video_duration}s"
                )
            else:
                interval_seconds = st.number_input("Analysis Interval (seconds)",
                                                   min_value=1, max_value=300, value=3, step=1)

            if video_input and st.button("Analyze Video", type="primary"):
                with st.spinner(f"Analyzing as {active_uc['name']}..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                        tmp.write(video_input.read())
                        tmp_path = tmp.name
                    try:
                        annotated_frames, video_results, frame_results = process_video(
                            tmp_path, active_uc, interval_seconds
                        )
                        st.session_state.video_results = (annotated_frames, video_results)
                        st.session_state.video_analysis_results = frame_results
                        st.session_state.video_name = video_input.name
                        st.session_state.video_use_case_key = active_uc['key']
                    finally:
                        os.unlink(tmp_path)

            # Export to DB
            if hasattr(st.session_state, 'video_analysis_results') and st.session_state.video_analysis_results:
                st.markdown("---")
                st.subheader("💾 Database Export")
                custom_name = st.text_input("Video Name (for database)",
                                            value=getattr(st.session_state, 'video_name', "video"))
                if st.button("Export to Database", type="primary"):
                    with st.spinner("Exporting..."):
                        db = get_db_service()
                        ok, msg = db.validate_connection()
                        if ok:
                            success, message = export_to_database(
                                custom_name,
                                st.session_state.video_use_case_key,
                                st.session_state.video_analysis_results
                            )
                            if success:
                                st.success(f"✅ {message}")
                            else:
                                st.error(f"❌ {message}")
                        else:
                            st.error(f"❌ Database connection failed: {msg}")

        with col2:
            st.subheader("Results")
            if hasattr(st.session_state, 'video_results') and st.session_state.video_results:
                annotated_frames, video_results = st.session_state.video_results
                if annotated_frames:
                    # Live-stream simulation controls
                    col_sa, col_sb = st.columns(2)
                    with col_sa:
                        if st.button("▶️ Start Live Stream Simulation", key="start_stream"):
                            st.session_state.streaming = True
                    with col_sb:
                        if st.button("⏹️ Stop Stream", key="stop_stream"):
                            st.session_state.streaming = False

                    if 'streaming' not in st.session_state:
                        st.session_state.streaming = False

                    if st.session_state.streaming:
                        st.markdown(f"### 🔴 Live — {active_uc['icon']} {active_uc['name']}")
                        stream_placeholder = st.empty()
                        report_placeholder = st.empty()
                        for i, frame_data in enumerate(annotated_frames):
                            if not st.session_state.streaming:
                                break
                            with stream_placeholder.container():
                                st.image(frame_data['frame'],
                                         caption=f"🔴 LIVE — Frame at {frame_data['timestamp']:.1f}s",
                                         use_column_width=True)
                            with report_placeholder.container():
                                if hasattr(st.session_state, 'video_analysis_results'):
                                    fr = st.session_state.video_analysis_results[i]
                                    st.markdown(format_single_frame_results(fr, i + 1, active_uc))
                            time.sleep(2)
                        st.session_state.streaming = False
                        st.success("Stream completed!")
                    else:
                        st.markdown("### Select Frame for Analysis")
                        frame_options = [f"Frame {i+1} ({f['timestamp']:.1f}s)" for i, f in enumerate(annotated_frames)]
                        idx = st.selectbox("Choose a frame to analyze:", range(len(frame_options)),
                                           format_func=lambda x: frame_options[x], key="frame_selector")
                        if idx is not None and idx < len(annotated_frames):
                            fd = annotated_frames[idx]
                            st.image(fd['frame'], caption=f"Frame at {fd['timestamp']:.1f}s",
                                     use_column_width=True)
                            if hasattr(st.session_state, 'video_analysis_results'):
                                fr = st.session_state.video_analysis_results[idx]
                                st.markdown("### Frame Analysis Report")
                                st.markdown(format_single_frame_results(fr, idx + 1, active_uc))

    # --------------- Tab 3: Settings ---------------
    with tab3:
        st.header("Configuration")
        s_tab1, s_tab2, s_tab3 = st.tabs(["🔌 Model APIs", "🗄️ Database", "🎯 Use Cases"])

        # ----- Model APIs -----
        with s_tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("YOLO Detection Settings")
                yolo_endpoint_input = st.text_input("YOLO Endpoint", value=settings["yolo_endpoint"])
                yolo_key_input = st.text_input("YOLO API Key", value=settings["yolo_api_key"], type="password")
            with col2:
                st.subheader("Qwen2.5-VL Analysis Settings")
                qwen_endpoint_input = st.text_input("Qwen Endpoint", value=settings["qwen_endpoint"])
                qwen_key_input = st.text_input("Qwen API Key", value=settings["qwen_api_key"], type="password")
            if st.button("Save API Settings", type="primary"):
                update_settings(yolo_endpoint_input, yolo_key_input, qwen_endpoint_input, qwen_key_input)
                st.success("API settings saved!")

        # ----- Database -----
        with s_tab2:
            col1, col2 = st.columns(2)
            with col1:
                db_url_input = st.text_input("Database Host", value=settings["db_url"])
                db_name_input = st.text_input("Database Name", value=settings["db_name"])
            with col2:
                db_user_input = st.text_input("Database User", value=settings["db_user"])
                db_password_input = st.text_input("Database Password", value=settings["db_password"], type="password")
            if st.button("Save Database Settings", type="primary"):
                update_settings(settings["yolo_endpoint"], settings["yolo_api_key"],
                                settings["qwen_endpoint"], settings["qwen_api_key"],
                                db_url_input, db_user_input, db_password_input, db_name_input)
                with st.spinner("Testing connection..."):
                    db = get_db_service()
                    ok, msg = db.validate_connection()
                    if ok:
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ {msg}")
                    if ok:
                        ok2, msg2 = db.create_table_if_not_exists()
                        if ok2:
                            st.success(f"✅ {msg2}")
                        else:
                            st.error(f"❌ {msg2}")
                        ok3, msg3 = get_use_case_service().create_table_if_not_exists()
                        if ok3:
                            st.success(f"✅ {msg3}")
                        else:
                            st.error(f"❌ {msg3}")

        # ----- Use Cases -----
        with s_tab3:
            st.subheader("🎯 Use Cases")
            st.markdown("Use cases define *what* YOLO filters for and *what question* Qwen answers. Built-ins ship with the app; you can also create your own below.")

            st.markdown("### Available Use Cases")
            for uc in ucs.get_all_use_cases():
                with st.expander(f"{uc['icon']} {uc['name']} {'(built-in)' if uc['is_builtin'] else '(custom)'}", expanded=False):
                    st.markdown(f"**Description:** {uc['description']}")
                    class_names = [COCO_CLASSES[c] for c in uc['classes'] if c in COCO_CLASSES]
                    st.markdown(f"**YOLO detects:** {', '.join(class_names) if class_names else 'none'}")
                    extras = uc.get('extra_class_names', []) or []
                    if extras:
                        st.markdown(f"**Qwen also looks for:** {', '.join(extras)}  _(not in YOLO's class list — described by VLM only)_")
                    st.markdown("**Prompt:**")
                    st.code(uc['prompt'], language="markdown")
                    if not uc['is_builtin']:
                        if st.button(f"🗑️ Delete", key=f"del_{uc['key']}"):
                            ok, msg = ucs.delete_custom_use_case(uc['key'])
                            if ok:
                                st.success(msg)
                            else:
                                st.error(msg)
                            st.rerun()

            st.markdown("---")
            st.markdown("### ➕ Create a Custom Use Case")
            with st.form("new_use_case_form"):
                col_a, col_b = st.columns(2)
                with col_a:
                    new_name = st.text_input("Name*", placeholder="e.g., School Drop-off Zone")
                    new_key = st.text_input("Key* (lowercase, underscores)", placeholder="e.g., school_dropoff",
                                            help="Short identifier. No spaces. Used in the DB and dropdown.")
                with col_b:
                    new_icon = st.text_input("Icon (emoji)", value="⚙️", max_chars=4)
                    new_desc = st.text_input("Short Description", placeholder="What does this scenario monitor?")

                st.markdown("**Which objects should YOLOv8 detect?***")
                st.caption("Type a comma-separated list of class names (YOLOv8 knows 80 object types).")
                classes_text = st.text_input(
                    "Classes to detect",
                    placeholder="e.g., person, car, truck, backpack",
                    help="Separate names with commas. Names must match YOLO's COCO classes — use the expander below to see all 80.",
                    label_visibility="collapsed"
                )
                with st.expander("💡 Show all 80 detectable objects"):
                    # Display names in a clean comma-separated list grouped by common/uncommon
                    all_names = sorted([n for n in COCO_CLASSES.values()])
                    st.markdown("**Available class names (copy the ones you need):**")
                    st.code(", ".join(all_names), language=None)

                st.markdown("**Qwen Analysis Prompt***")
                st.caption("Write what you want Qwen to analyze. Must include `{detections}` where the detected objects list should appear.")
                new_prompt = st.text_area(
                    "Prompt",
                    height=250,
                    value="""Analyze this scene with the following detected objects: {detections}

Please provide a concise analysis in this format:

**Traffic Status:** [status here]
**Traffic Flow:** [flow level here]
**Incident Analysis:** [what's happening]
**Safety Assessment:** [safety notes]
**Recommendations:** [suggestions]

Keep each section to 1-2 sentences maximum.
"""
                )

                submitted = st.form_submit_button("💾 Save Use Case", type="primary")
                if submitted:
                    # Parse the comma-separated class names.
                    # We accept BOTH COCO-known names (YOLO will draw boxes) AND
                    # unknown names like "robot" or "forklift" (YOLO can't box them,
                    # but Qwen can still describe them in the scene analysis).
                    raw_names = [n.strip().lower() for n in classes_text.split(",") if n.strip()]
                    selected_classes = []          # COCO IDs → YOLO will detect + box
                    extra_class_names = []         # non-COCO names → passed to Qwen only
                    for name in raw_names:
                        if name in COCO_NAME_TO_ID:
                            selected_classes.append(COCO_NAME_TO_ID[name])
                        else:
                            extra_class_names.append(name)

                    if not raw_names:
                        st.error("❌ You must enter at least one object class.")
                    else:
                        ok, msg = ucs.save_custom_use_case(
                            key=new_key.strip(),
                            name=new_name.strip(),
                            icon=new_icon.strip() or "⚙️",
                            description=new_desc.strip(),
                            classes=selected_classes,
                            prompt=new_prompt,
                            extra_class_names=extra_class_names
                        )
                        if ok:
                            st.success(f"✅ {msg}")
                            if extra_class_names:
                                st.warning(
                                    f"⚠️ These objects aren't in YOLO's 80-class list: "
                                    f"**{', '.join(extra_class_names)}**. "
                                    "YOLOv8 won't draw bounding boxes on them, but Qwen will still describe them "
                                    "in the scene analysis. For bounding boxes on these objects, you'd need a "
                                    "custom-trained YOLO model or an open-vocabulary detector like YOLO-World."
                                )
                            st.info("Reload the page to see this use case in the Active Use Case dropdown at the top.")
                        else:
                            st.error(f"❌ {msg}")

    # Footer
    st.markdown("---")
    st.markdown("*Powered by YOLOv8 for object detection and Qwen2.5-VL for scene analysis. Horizontal by design.*")


if not os.getenv("YOLO_API_KEY") or not os.getenv("QWEN_API_KEY"):
    print("Warning: API keys not set. Configure them via env vars or the Settings tab.")

create_interface()

