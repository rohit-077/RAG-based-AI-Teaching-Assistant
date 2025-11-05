# RAG-based-AI-Teaching-Assistant

## Overview
This project is an end-to-end Retrieval-Augmented Generation (RAG) pipeline that transforms any videos into a queryable knowledge base. It combines Automatic Speech Recognition (ASR) for audio transcription, text chunking, semantic embedding generation, and vector similarity search to enable context-aware question answering using a local LLM (DeepSeek-R1). The system uses Whisper for Speech-to-Text (STT) and bge-m3 for embedding generation, creating an efficient offline RAG workflow. 
Everything runs locally using [Ollama](https://ollama.com/) models, so there’s no cloud dependency.

Here’s roughly what happens:
1. Convert videos (in `/videos`) to MP3s using FFmpeg.  
2. Use Whisper to transcribe the audios into text chunks.  
3. Generate embeddings using the **bge-m3** model from Ollama.  
4. Ask a question — it finds the most relevant chunks.  
5. Finally, generate a response using the **deepseek-r1:1.5b** model.

---

## Folder Structure
project/
│
├── videos/ # put your .mp4 tutorial videos here
├── audios/ # will be created automatically, stores extracted mp3s
├── converted_texts/ # json files of transcriptions + chunks
├── outputs/ # final LLM answers
│
├── video_to_audio.py
├── create_chunks.py
├── create_embeddings.py
├── process_query.py
├── generate_response.py
├── main.py # orchestrates everything
└── README.md


---

## Prerequisites
Make sure you have the following installed before running anything:

- **Python 3.9+**
- **FFmpeg** (add it to PATH)
- **Ollama** (running locally)
- Ollama models:
  - `bge-m3`
  - `deepseek-r1:1.5b`
- Python libraries:
  ```bash
  pip install requests pandas joblib numpy scikit-learn openai-whisper

  
Steps to Execute
1. Pull required Ollama models
ollama pull bge-m3
ollama pull deepseek-r1:1.5b

2. Prepare your input videos

Place your .mp4 tutorial videos inside the videos/ folder.
Example:

videos/
│
├── 01 - Machine Learning Tutorial - What is Machine Learning？.mp4
├── 02 - Machine Learning Tutorial - Types of Learning.mp4
...

3. Run the pipeline

You can either run each script step-by-step or just execute everything from main.py.

Option 1 — Run all in one go:
python main.py

Option 2 — Run step-by-step:
python video_to_audio.py
python create_chunks.py
python create_embeddings.py
python process_query.py
python generate_response.py

4. Ask a Question

When prompted during process_query.py, just type your question related to the tutorial.
It will prepare a prompt.txt which the generate_response.py file will feed into the LLM.

Example:

Kindly ask your question related to the tutorial: what are neural networks?

5. Get Your Answer

Once done, your final answer will be saved in:

outputs/answer.txt
