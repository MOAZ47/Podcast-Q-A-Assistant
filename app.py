import streamlit as st
import os, sys
import time
import logging
from logging.handlers import RotatingFileHandler
from tempfile import NamedTemporaryFile
from main import initialize_pipeline
from agents.transcription import transcribe_audio

# Use Streamlit's current working directory
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_PATH = os.path.join(LOG_DIR, "app_latency.log")

# Get a logger specific to this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Ensure handlers are only added once
# Below ensures the log of this file takes precedence over others
if not logger.handlers:
    # File handler for app_latency.log
    file_handler = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=2, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # Stream handler for console output
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    # Ensure console output uses UTF-8 for compatibility
    stream_handler.stream.reconfigure(encoding='utf-8')

    # Define formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Set formatter for handlers
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


# --- Streamlit UI Setup ---
st.set_page_config(page_title="Podcast Fact Checker", layout="centered")
st.title("🎙️ Podcast Fact Checker Assistant")

st.markdown(
    """
    Upload a podcast episode, and this app will:
    1. 🔊 Transcribe the audio  
    2. 📝 Summarize the content  
    3. 🌐 Fact-check the summary via web search  
    4. 📊 Generate a markdown fact-check report with confidence and recommendation  
    """
)

# --- Session State ----
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "report" not in st.session_state:
    st.session_state.report = None

# --- Whisper Model Selection ---
model_size = st.selectbox("Select Whisper model for transcription:", ["base", "small", "medium", "large"], index=0)

# --- File Uploader ---
uploaded_file = st.file_uploader("🎧 Upload a podcast audio file (.mp3, .wav, .m4a)")

# --- Processing Function ---
def process_podcast(file_path, model_size="base"):
    overall_start = time.time()

    try:
        # --- Step 1: Transcription ---
        t1 = time.time()
        with st.spinner("Step 1: Transcribing audio..."):
            transcription = transcribe_audio(file_path, model_size=model_size)
            st.session_state.transcript = transcription
        t2 = time.time()
        transcription_time = t2 - t1
        logger.info(f"Transcription took {transcription_time:.2f} seconds")
        st.info(f"🕒 Transcription took {transcription_time:.2f} seconds")

        # --- Step 2: Report Generation ---
        t3 = time.time()
        with st.spinner("Step 2: Generating final report..."):
            report = initialize_pipeline(transcription)
            st.session_state.report = report
        t4 = time.time()
        report_time = t4 - t3
        logger.info(f"Report generation took {report_time:.2f} seconds")
        st.info(f"🕒 Report generation took {report_time:.2f} seconds")

        # --- Total ---
        total_time = time.time() - overall_start
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        st.success(f"✅ Total processing time: {total_time:.2f} seconds")

    finally:
        os.unlink(file_path)

# --- Handle Upload & Trigger Analysis ---
if uploaded_file is not None:
    if not uploaded_file.name.lower().endswith((".mp3", ".wav", ".m4a")):
        st.warning("⚠️ Please upload a valid audio file (.mp3, .wav, .m4a)")
    else:
        st.audio(uploaded_file)

        with NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1], dir="temp") as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        if st.button("🔍 Analyze Podcast"):
            try:
                process_podcast(file_path, model_size=model_size)
            except Exception as e:
                logger.exception("An error occurred during podcast processing")
                st.error(f"❌ An error occurred: {e}")
                os.unlink(file_path)

# --- Transcript Display ---
if st.session_state.transcript:
    st.subheader("📄 Transcript")
    st.text_area("Transcript", st.session_state.transcript, height=300)

# --- Report Display ---
if st.session_state.report:
    st.subheader("📊 Fact-Check Report")
    st.markdown(st.session_state.report, unsafe_allow_html=True)

    st.download_button(
        "⬇️ Download Report",
        st.session_state.report,
        file_name="podcast_fact_report.md"
    )

# --- Optional Log Viewer ---
with st.expander("📜 View Recent Log Output"):
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            logs = f.readlines()[-20:]
        st.text("".join(logs))
    else:
        st.info("No logs yet. Process a file to generate logs.")

# --- Reset Session ---
st.markdown("---")
if st.button("🔁 Rerun / Start Over"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Session reset. Please upload a new podcast file to begin.")
    st.rerun()
