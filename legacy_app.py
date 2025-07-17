import streamlit as st
import os
from tempfile import NamedTemporaryFile
from main import initialize_pipeline

from agents.transcription import transcribe_podcast
from langchain.agents import Tool

# Initialize session state
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "report" not in st.session_state:
    st.session_state.report = None

# Define Tool for transcription
transcriber_tool = Tool(
        name="PodcastTranscriber",
        func=transcribe_podcast,
        description="Use this to transcribe a podcast audio file."
    )

st.title("🎙️ Podcast Q&A Assistant")

st.write('LLM powered podcast analyser to check if the content of podcast are factual or not')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    st.audio(uploaded_file)

    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1],
                            dir="temp") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name
    
    print(f"file saved at: {file_path}")

    if st.button("🔍 Analyze Podcast"):
        with st.spinner("Transcribing and analyzing..."):
            try:
                # Run transcription
                transcription = transcriber_tool.run(file_path)
                st.session_state.transcript = transcription

                # Run analysis pipeline
                report = initialize_pipeline(transcription)
                st.session_state.report = report

                os.unlink(file_path)  # Clean up temp file
            except Exception as e:
                st.error(f"An error occurred: {e}")
                os.unlink(file_path)

# Display transcript and report if available
if st.session_state.transcript:
    st.subheader("📄 Transcript")
    st.text_area("Transcript", st.session_state.transcript, height=300)

if st.session_state.report:
    st.subheader("📊 Analysis Report")
    st.text_area("Report", st.session_state.report, height=300)

# Rerun/reset session
st.markdown("---")
if st.button("🔁 Rerun / Start Over"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Session reset. Please upload a new podcast file to begin.")
    st.experimental_rerun()