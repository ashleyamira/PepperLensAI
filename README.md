# PepperLensAI

🌶️ PepperLens AI

AI-Powered Chili Leaf Disease Detection and Plant Care Assistant

PepperLens AI is a web-based application that uses computer vision and artificial intelligence to identify chili leaf diseases and provide plant care recommendations. It combines YOLOv11 for image detection and Retrieval-Augmented Generation (RAG) for intelligent plant health assistance.

Features

Disease Detection – Identify chili leaf diseases using YOLOv11.

AI Diagnosis – View detection results, confidence scores, and disease information.

AI Chatbot – Ask questions about plant diseases, prevention, and treatment.

User-Friendly Interface – Simple and accessible web interface.

Tech Stack

Language: Python

Computer Vision: YOLOv11, OpenCV

AI Integration: LangChain, ChromaDB, RAG

Web Framework: Streamlit

Deployment: Docker

Getting Started

1. Build the Docker image

docker build -t pepperlens-ai .

2. Run the application

docker run --env-file .env -p 8501:8501 pepperlens-ai

3. Open in your browser

http://localhost:8501

Requires a configured .env file, trained YOLO model weights, and knowledge-base assets.
