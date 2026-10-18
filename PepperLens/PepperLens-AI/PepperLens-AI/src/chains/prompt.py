from langchain_core.prompts import PromptTemplate


def get_rag_prompt():
    template = """You are an AI assistant specializing in pepper plant health and cultivation.
Use the provided context to answer the user's question in English.
If the answer is not supported by the context, say that you do not know. Do not invent facts.

Context: {context}
Question: {input}
Helpful and clear answer in English:"""
    return PromptTemplate.from_template(template)


DISEASE_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["disease_name", "context"],
    template="""You are a pepper plant health advisor.
Use ONLY the reference information below. Do not invent facts.

--- REFERENCE INFORMATION ---
{context}
-----------------------------

Explain the condition or disease named "{disease_name}" in clear, accessible English.
Describe symptoms, possible causes, and recommended care when supported by the context.
If the condition is Healthy, explain that the plant appears healthy; do not call an unhealthy plant healthy.
"""
)
