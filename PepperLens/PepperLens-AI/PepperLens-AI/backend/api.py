from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn
import base64 
import cv2 
import numpy as np 
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from src.chains.rag import generate_narrative
from src.chains.chain import create_rag_chain

app = FastAPI(
    title="PepperLens API",
    description="Pepper leaf disease detection API powered by YOLOv11 and LLM RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = YOLO("model/bestv11.pt")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the PepperLens API. Use /detect for inference."}

@app.post("/detect")
async def detect_disease(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="The uploaded file must be an image.")

    if not model:
        raise HTTPException(status_code=500, detail="The YOLO model is unavailable or not initialized.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        results = model(image)
        
        annotated_frame = results[0].plot()

        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(rgb_image)

        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        detected_labels = {}
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[cls_id]

            if label not in detected_labels or confidence > detected_labels[label]:
                detected_labels[label] = confidence

        detections = []
        for label, confidence in detected_labels.items():
            label_capitalized = {"kuning": "Yellowing", "keriting": "Leaf Curl", "bercak": "Leaf Spot", "whitefly": "Whitefly", "sehat": "Healthy"}.get(label.lower(), label.capitalize())
            narrative = generate_narrative(label_capitalized)
            
            detections.append({
                "class": label_capitalized,
                "confidence": round(confidence, 2),
                "narrative": narrative
            })

        return {
            "status": "success",
            "filename": file.filename,
            "total_detections": len(detections),
            "results": detections,
            "image_base64": img_base64 
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")
    
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_expert(request: QuestionRequest):
    try:
        rag_chain = create_rag_chain()
        
        def generate_response():
            for chunk in rag_chain.stream(request.question):
                yield chunk 
                
        return StreamingResponse(
            generate_response(), 
            media_type="text/event-stream"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An LLM error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)