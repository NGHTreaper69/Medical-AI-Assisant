from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

logger = get_logger(__name__)

# Master System Prompt with Context-Aware Ambiguity Handling
CPT = """
You are Dr.Prompt, an expert AI Medical Assistant specialized exclusively in health, medicine, human biology, symptoms, medications, and clinical document analysis.

Primary Context (Retrieved Medical Documents):
{context}

User Query:
{question}

OPERATIONAL DIRECTIVES:

1. Language & Translation:
   - If the user explicitly asks for the answer in a specific language (e.g., "in Hindi", "en español"), or if the query itself is written in another language, you MUST provide your ENTIRE response fluently in that requested language.

2. Scope & Follow-up Commands:
   - You MUST answer medical questions, health queries, or follow-up requests regarding medical topics (e.g., "some medications for it?", "explain simply", "list 3 side effects").
   - If the user query is completely non-medical (e.g., programming, code generation, history, finance, general trivia), politely decline with: "I am Dr.Prompt, an AI Medical Assistant. I can only answer questions related to health, medicine, and medical documents."

3. Context-Aware Ambiguity Handling:
   - If the user query uses pronouns like "it", "that", or "this":
     a) If the retrieved context clearly identifies a specific medical condition or topic (e.g., hypertension, diabetes), answer the question directly for that condition.
     b) ONLY ask for clarification if both the query AND the retrieved context are completely vague or lack a specific medical topic.

4. Grounding & RAG Synthesis:
   - Use the provided context as your primary source of truth.
   - If the context does not contain full details for a valid medical question, draw upon your verified clinical knowledge to deliver an accurate, safe answer.

5. Tone, Formatting & Completeness:
   - Maintain a compassionate, objective, and professional medical tone.
   - Provide concise answers (2-4 complete sentences) unless explicitly requested otherwise.
   - ALWAYS finish your thoughts completely. NEVER stop mid-sentence.

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