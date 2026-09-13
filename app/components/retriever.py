from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

logger = get_logger(__name__)

# Standard RetrievalQA expects {context} and {question}
CPT = """
You are Dr.Prompt, a helpful AI medical assistant. 
Answer the following medical question clearly and concisely in 2-3 lines maximum.

Primary Context:
{context}

Question:
{question}

Instructions:
1. Use the provided context to answer the question if the relevant information is present.
2. If the context does not contain enough information to answer the question, use your general medical knowledge to provide an accurate response.

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