import sys
import os
from dotenv import load_dotenv

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_dir)

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI 
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.messages import SystemMessage
from langchain_classic.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from src.retrieval.vector_store import get_vector_store
from src.retrieval.retriever import get_retriever
from src.chains.prompt import get_rag_prompt

load_dotenv()

def create_rag_chain(disease_label=None):
    llm = ChatGroq(
        model="openai/gpt-oss-20b", 
        streaming=True,
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
    ) 
    
    vs = get_vector_store()
    base_retriever = get_retriever(
        vector_store=vs, 
        search_type="similarity", 
        k=5,
        filter_label=disease_label 
    )
    
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    print("Loading reranker model (BAAI/bge-reranker-v2-m3)...")
    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
    
    rerank_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=mq_retriever
    )
    
    prompt = get_rag_prompt()
    
    def format_docs(docs):
        print("\n" + "="*50)
        print("🎯 [DEBUG] 3 DOKUMEN TERBAIK SETELAH DI-RERANK:")
        for i, doc in enumerate(docs):
            sumber = doc.metadata.get('label') or doc.metadata.get('source') or 'Unknown source'
            skor = doc.metadata.get('relevance_score', 'N/A')
            print(f"  [{i+1}] Topic: {sumber} | Relevance Score: {skor}")
        print("="*50 + "\n")
        
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": rerank_retriever | format_docs, "input": RunnablePassthrough()} 
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
