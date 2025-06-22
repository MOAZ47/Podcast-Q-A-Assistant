# 🎙️ Podcast Q&A Assistant with Multimodal RAG + Agentic AI Framework

A Multimodal LLM Agentic AI application that allows users to upload podcast/audio files, transcribes them using Whisper, summarizes the content, fact-checks the claims using an LLM + internet search, and delivers a clear, confidence-rated report — all through a modular multi-agent pipeline.

---

## 🚀 Features

✅ Upload and analyze any podcast/audio file  
✅ Automatic transcription using OpenAI Whisper  
✅ Summary generation via LLM  
✅ Claim-level fact-checking using external evidence  
✅ Agent-based architecture: transcriber, summarizer, fact-checker, reporter  
✅ Interactive Streamlit UI  
✅ Few-shot prompting with example control  
✅ Session memory, rerun/reset support  
✅ Modular codebase for easy scaling

---

## 📦 Tech Stack

| Component       | Technology                                |
|------------------|--------------------------------------------|
| Transcription    | [OpenAI Whisper](https://github.com/openai/whisper) |
| LLM              | [Cohere LLM](https://cohere.com) via LangChain |
| Fact Checking    | Custom agent using LLM + external search   |
| Vector Store     | [Weaviate](https://weaviate.io/) (optional) |
| Prompting        | LangChain's `FewShotPromptTemplate`        |
| Frontend         | Streamlit                                  |
| Embeddings       | Sentence Transformers via HuggingFace      |

---

## 🧠 Architecture Overview

```plaintext
User Uploads Audio
        │
        ▼
[ Transcriber Agent ]  ← Whisper
        │
        ▼
[ Summarizer Agent ]   ← LLM (Cohere or others)
        │
        ▼
[ Fact-Checker Agent ] ← LLM + Internet
        │
        ▼
[ Reporter Agent ]     ← Few-shot LLM Report Generation
        │
        ▼
Streamlit UI → Transcript + Final Report

```

---

## Installation
```bash
git clone https://github.com/Moaz47/Podcast-Q-A-Assistant.git
cd agentic-audio-rag

# Optional: set up a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```
---

## Running the code
```bash
streamlit run app.py

```

---

## 📁 Project Structure
```plaintext
.
├── app.py               # Streamlit app UI
├── main.py              # Orchestrates the pipeline
├── config.py            # API keys
├── agents/
│   ├── transcription.py
│   ├── summarizer.py
│   ├── factchecker.py
│   └── reporter.py
├── requirements.txt
└── README.md

```
---

## ✨ Future Improvements
<ul>
    <li>Add chat interface for user questions over the podcast. </li>
    <li>Exportable PDF reports. </li>
    <li>Multi-turn memory for user corrections. </li>
    <li>OpenAI / Gemini model switching. </li>
</ul>