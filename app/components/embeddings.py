import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def get_embeddings():
    try:
        logger.info("Initializing HuggingFace Inference API embeddings....")
        
        # Read the API key from environment variables
        hf_api_key = os.getenv("HuggingFace_API_KEY")
        
        if not hf_api_key:
            logger.warning("HuggingFace_API_KEY environment variable is missing.")
        
        # Connect to HuggingFace Inference API instead of running PyTorch locally
        model = HuggingFaceInferenceAPIEmbeddings(
            api_key=hf_api_key,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        logger.info("HuggingFace API embeddings initialized ✅✅✅")
        return model

    except Exception as e:
        error_message = CustomException("Failed to load HuggingFace API Embeddings.", e)
        logger.error(str(error_message))
        raise error_message