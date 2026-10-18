# PepperLens AI

PepperLens AI is an English-language adaptation of a pepper leaf disease detection application using YOLOv11 and a retrieval-augmented language model (RAG).

## Features
- Upload a pepper leaf image and identify possible diseases with YOLOv11.
- View annotated detections, confidence scores, and AI-generated care guidance.
- Ask pepper cultivation and disease questions through a RAG chatbot.
- Run the Python FastAPI backend with a Streamlit interface or the HTML/JavaScript frontend.

## Run with Docker
```bash
docker compose up --build
```

The Streamlit app and API ports are defined in `docker-compose.yml`.

## Local frontend
Serve `frontend/` through a local HTTP server and run the FastAPI backend on port 8000. By default, the frontend sends requests to `http://localhost:8000`. To use another backend, set `window.PEPPERLENS_API_BASE` before loading `script.js`.

## Setup notes
- Configure `GROQ_API_KEY` for AI answers.
- This archive contains Git LFS pointer files rather than the actual YOLO model, source PDFs, and vector database. Download or regenerate these assets before expecting full inference/RAG functionality.
- The translated `data/data.json` is the English reference knowledge base. Rebuild the Chroma index using the project's ingestion workflow before querying it.
- Check model label mappings in `backend/api.py` against your actual model classes.
- Test with `pytest -v` after installing dependencies.

## Source and licensing
This project is adapted from the user-supplied *Chili Leaf Disease Detection with YOLOv11 and LLM-Based RAG* source archive. Renaming and translating do not transfer copyright or remove attribution/license obligations. Review the original repository license and permission terms before publishing or submitting as your own work.
