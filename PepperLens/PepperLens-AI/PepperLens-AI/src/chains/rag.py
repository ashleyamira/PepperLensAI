import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.chains.prompt import DISEASE_PROMPT_TEMPLATE
from src.retrieval.vector_store import get_vector_store

load_dotenv()

def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b", 
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
    )

def generate_narrative(disease_name):
    print(f"Searching reference data for label: {disease_name}...")
    
    search_query = f"Causes, symptoms, treatment, and prevention of {disease_name} in pepper plants."
    
    try:
        vectorstore = get_vector_store()
        results = vectorstore.similarity_search(
            query=search_query, 
            k=3,
            filter={"label": disease_name}
        )
    except Exception as e:
        print(f"Warning: Vector store search failed: {e}")
        results = []

    if not results:
        return f"No reference information found for {disease_name}."

    retrieved_context = "\n\n".join([doc.page_content for doc in results])

    print("Reference data found. Generating an AI explanation...")

    llm = get_llm()
    chain = DISEASE_PROMPT_TEMPLATE | llm

    response = chain.invoke({
        "disease_name": disease_name,
        "context": retrieved_context
    })
    
    return response.content