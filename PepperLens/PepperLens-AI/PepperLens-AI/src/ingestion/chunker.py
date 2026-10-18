from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from typing import List
from src.ingestion.embedder import get_embedding_model 

def split_documents(documents: List[Document], chunk_size, chunk_overlap):
    print(f"Memulai proses pemotongan {len(documents)} dokumen...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"Successfully split documents into {len(chunks)} text chunks.")
    return chunks

    
    
    
