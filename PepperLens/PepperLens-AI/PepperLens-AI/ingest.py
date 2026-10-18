import os
import sys
root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir)

from src.ingestion.loader import load_data
from src.ingestion.chunker import split_documents
from src.retrieval.vector_store import get_vector_store

SOURCES = [
    "https://www.dgwfertilizer.co.id/8-hama-dan-penyakit-penting-pada-tanaman-cabai/",
    "https://mitrabertani.com/artikel/detail/Budidaya-Cabai-Sederhana-tapi-Penting-Cara-Tepat-Tanam-Cabai",
    "https://digitani.ipb.ac.id/bagaimana-langkah-langkah-budidaya-cabai/",
    "data/cabai.pdf",
    "data/budidaya_cabai_rawit.pdf",
    "data/penyakit.pdf"
]

def run_ingestion_pipeline():
    print("Memulai Data Ingestion Pipeline (Mode Semantik)\n")
    all_chunks = []
    
    for source in SOURCES:
        try:
            print(f"Reading source: {source}")
            raw_docs = load_data(source)
            
            chunks = split_documents(raw_docs, chunk_size=1000, chunk_overlap=200)
            
            all_chunks.extend(chunks)
            print(f"Successfully processed into {len(chunks)} semantic chunks.\n")
        except Exception as e:
            print(f"Failed to process {source}: {e}\n")
            
    if all_chunks:
        print(f"Menyiapkan penyisipan {len(all_chunks)} semantic chunks ke ChromaDB...")
        
        db = get_vector_store()
        db.add_documents(all_chunks)
        
        print("\nSelesai! Semua data web & PDF telah diproses secara semantik.")
        print("Vector database is ready for the FastAPI service.")
    else:
        print("\nNo data was successfully processed.")

if __name__ == "__main__":
    run_ingestion_pipeline()