import streamlit as st
import os
from tempfile import NamedTemporaryFile
from main import initialize_pipeline  # make sure your pipeline supports a dynamic file path
import asyncio
import nest_asyncio

# Patch the event loop
nest_asyncio.apply()
asyncio.set_event_loop(asyncio.new_event_loop())

# Initialize session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "transcription" not in st.session_state:
    st.session_state.transcription = None
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

# Title
st.title("🎙️ Podcast Q&A Assistant")
st.write("Upload a podcast audio file and ask questions about its content.")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

# Handle file upload and pipeline initialization
if uploaded_file and not st.session_state.file_uploaded:
    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name
    print("\n File uploaded \n")
    with st.spinner("Initializing pipeline and transcribing audio..."):
        try:
            rag_chain, transcription = initialize_pipeline(file_path)
            st.session_state.rag_chain = rag_chain
            st.session_state.transcription = transcription
            st.session_state.file_uploaded = True
            os.unlink(file_path)  # Clean up temp file
        except Exception as e:
            st.error(f"Error: {e}")
            os.unlink(file_path)

# Show transcription and enable Q&A
if st.session_state.file_uploaded:
    st.success("Audio processed successfully!")
    
    st.subheader("Transcription Preview")
    st.text_area("Full Transcription", st.session_state.transcription, height=200)

    question = st.text_input("Ask a question about the podcast")
    if question and st.session_state.rag_chain:
        with st.spinner("Generating answer..."):
            response = st.session_state.rag_chain.invoke(question)
        st.subheader("Answer")
        st.write(response)

# Reset functionality
if st.button("Reset"):
    for key in ["rag_chain", "transcription", "file_uploaded"]:
        st.session_state[key] = None
    st.rerun()