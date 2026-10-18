import streamlit as st
import requests
from PIL import Image
import io
import base64


API_DETECT_URL = "http://backend:8000/detect"
API_ASK_URL = "http://backend:8000/ask"

st.set_page_config(page_title="PepperLens AI", page_icon="🌶️", layout="centered")

st.sidebar.title("🌶️ Menu PepperLens")
st.sidebar.markdown("Choose a feature to get started:")
menu = st.sidebar.radio(
    "====",
    ["📷 Disease Detection (Image)", "🤖 AI Chatbot (Text)"]
)

if menu == "📷 Disease Detection (Image)":
    st.title("🌶️🌿 Pepper Leaf Disease Detection")
    st.write("Upload a photo of a pepper leaf to identify possible diseases and get recommendations.")
    
    uploaded_file = st.file_uploader("Upload your pepper leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)

        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
        try:
            with st.spinner("Analyzing the image with YOLOv11..."):
                response = requests.post(API_DETECT_URL, files=files)
                
            if response.status_code == 200:
                data = response.json()
                    
                if "image_base64" in data:
                    image_bytes = base64.b64decode(data["image_base64"])
                    annotated_image = Image.open(io.BytesIO(image_bytes))
                    st.image(annotated_image, caption="YOLO Analysis Result", use_container_width=True)

                st.subheader("📋 Diagnosis Results: ")
                if data.get("total_detections", 0) > 0:
                    for item in data["results"]:
                        st.markdown(f"### 🦠 {item['class']} (Confidence: {item['confidence']:.0%})")
                        st.info(f"**Recommended Treatment:**\n\n{item['narrative']}")
                else:
                    st.success("No disease detected. The leaf appears healthy!")
            else:
                st.error(f"Unable to process the image. Error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API server. Make sure the backend is running.")

elif menu == "🤖 AI Chatbot (Text)":
    st.title("🤖 Pepper Care Assistant")
    st.write("Ask about pepper plant care, diseases, or fertilizers.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your question here... (e.g., When should I harvest peppers?)"):
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Searching for relevant information..."):
                try:
                    payload = {"question": prompt}
                    response = requests.post(API_ASK_URL, json=payload, stream=True)
                    
                    if response.status_code == 200:
                        def generate_chunks():
                            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                                if chunk:
                                    yield chunk
                        
                        jawaban = st.write_stream(generate_chunks())
                        st.session_state.messages.append({"role": "assistant", "content": jawaban})
                    else:
                        error_msg = f"Unable to get an answer. Error: {response.status_code}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        
                except requests.exceptions.ConnectionError:
                    error_msg = "Cannot connect to the API server. Make sure the backend is running."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.sidebar.markdown("---")

st.sidebar.markdown("""
    **1. Vision (Deteksi Penyakit)**
    * Model: `YOLOv11s` (Small)
    * Framework: `Ultralytics`
    
    **2. Retrieval-Augmented Generation (RAG)**
    * Vector Database: `ChromaDB`
    * Embedding: `Qwen3-Embedding-0.6B`
    * Re-ranker: `BAAI/bge-reranker-base`
    
    **3. Large Language Model (LLM)**
    * Model: `OpenAI / GPT-OSS-20B`
    * Provider: `Groq Cloud API`
    """)

st.sidebar.markdown("---")

st.sidebar.caption("© 2026 PepperLens AI Engineer Portfolio")