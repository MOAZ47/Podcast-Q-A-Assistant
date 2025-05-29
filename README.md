# 🎙️ Podcast Q&A Assistant

An intelligent web application that transforms podcast audio into searchable knowledge using Whisper transcription and Retrieval-Augmented Generation (RAG) technology.

<img src="img/img1.png" width="300" height="400">

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **Multi-format Support** | Upload `.mp3`, `.wav`, or `.m4a` files |
| **Accurate Transcription** | Whisper ASR with model size options (base/small/medium/large) |
| **Semantic Search** | Weaviate vector database with Cohere embeddings |
| **Intelligent Q&A** | RAG pipeline providing contextual answers |
| **User-Friendly UI** | Streamlit interface with audio playback |


## 🛠️ Quick Start

### Prerequisites
- Python 3.8+
- FFmpeg (`sudo apt install ffmpeg` or [download](https://ffmpeg.org/))
- Cohere API key ([sign up](https://dashboard.cohere.com/))

### Installation

Here's the corrected version with proper code block formatting:

```markdown
1. Clone the repository:
   ```bash
   git clone https://github.com/MOAZ47/Podcast-Q-A-Assistant.git
   cd Podcast-Q-A-Assistant
   ```

2. Set up virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/MacOS
   .\venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## 🧩 Example Usage
   ```bash
   streamlit run app.py
   ```

## ⚠️ Troubleshooting
| Issue	 | Solution |
|---------|-------------|
|Large file errors | Use smaller Whisper model or chunk audio files |
|WinError 2 | Install FFmpeg and add to PATH |
|API key errors | Verify keys in .env and service quotas |
	
	
	






