from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

logger = get_logger(__name__)

# Prompt template: Strictly enforces medical domain boundary & complete sentences
CPT = """
You are Dr.Prompt, an AI Medical Assistant specialized exclusively in health, medicine, and clinical information.

Primary Context:
{context}

Question:
{question}

Instructions:
1. Strict Domain Restriction: You MUST ONLY answer questions related to medicine, health, human biology, symptoms, treatments, medications, or clinical conditions.
2. If the user asks a non-medical question (e.g., about programming, computer science, history, physics, finance, or general trivia), politely decline by stating: "I am Dr.Prompt, an AI Medical Assistant. I can only answer questions related to health, medicine, and medical documents."
3. For valid medical questions: Use the provided context first. If the context does not contain enough detail, use your general medical knowledge to provide a clear, accurate response (2-4 sentences). 
4. Ensure your response always consists of complete thoughts and finishes with a complete sentence. Never cut off mid-sentence.

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(
        template=CPT,
        input_variables=["context", "question"]
    )

def create_qa_chain():
    try:
        logger.info("Loading vectorstore for context...")
        db = load_vector_store()

        if db is None:
            raise CustomException("Vector store not present or empty.")

        logger.info("Loading LLM model...")
        llm = load_llm()

        if llm is None:
            raise CustomException("LLM failed to load.")

        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 4}),
            return_source_documents=False,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )

        logger.info("Successfully created the QA chain.")
        return qa_chain

    except Exception as e:
        error_message = f"Failed to create QA chain: {str(e)}"
        logger.error(error_message)
        raise CustomException(error_message)