from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

logger = get_logger(__name__)

# Updated Prompt Template with Ambiguity Handling
CPT = """
You are Dr.Prompt, an AI Medical Assistant specialized exclusively in health, medicine, and clinical information.

Primary Context:
{context}

Question/Input:
{question}

Instructions:
1. Ambiguous/Incomplete Inputs: If the user uses pronouns or vague references without context (e.g., "Explain the side effects of that", "How do I treat it?", "What causes this?"), ask the user to clarify which condition, symptom, or medication they are asking about.
2. Medical Scope & Meta-Commands: Answer valid medical questions, health queries, or follow-up instructions modifying a previous medical answer (e.g., "explain simply", "in 2 bullet points").
3. Out-of-Scope Rejection: ONLY decline if the user asks a completely unrelated non-medical topic (e.g., programming, history, physics, finance). Respond with: "I am Dr.Prompt, an AI Medical Assistant. I can only answer questions related to health, medicine, and medical documents."
4. Quality: Provide clear, accurate responses (2-4 sentences). Never stop mid-sentence.

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