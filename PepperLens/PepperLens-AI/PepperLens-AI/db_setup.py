import json
import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

JSON_FILE_PATH = "data/data.json"
CHROMA_DB_DIR = "chroma_data"

def main():
    if not os.path.exists(JSON_FILE_PATH):
        print(f"Error: File {JSON_FILE_PATH} not found.")
        return
    
    print("Reading JSON data...")
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        chili_data = json.load(f)

    print("Converting nested JSON data into LangChain Document format")
    documents = []
    
    for label, details in chili_data.items():
        teks_gabungan = f"Penyakit/Kondisi: {label}\n"
        
        if details.get("causes"):
            teks_gabungan += "Penyebab:\n- " + "\n- ".join(details["causes"]) + "\n"
        
        if details.get("symptoms"):
            teks_gabungan += "Gejala:\n- " + "\n- ".join(details["symptoms"]) + "\n"
            
        if details.get("treatment"):
            teks_gabungan += "Penanganan:\n- " + "\n- ".join(details["treatment"]) + "\n"
            
        if details.get("prevention"):
            teks_gabungan += "Pencegahan:\n- " + "\n- ".join(details["prevention"]) + "\n"

        doc = Document(
            page_content=teks_gabungan,
            metadata={"label": label}
        )
        documents.append(doc)

    print("Loading embedding model")
    embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

    print("Menyimpan data ke ChromaDB")
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name="pepperlens_kb"
    )
    print(f"Success! Database saved in folder: {CHROMA_DB_DIR}\n")


if __name__ == "__main__":
    main()