import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def get_embeddings():
    try:
        logger.info("Initializing HuggingFace Endpoint API embeddings....")
        
        # Read the API key from environment variables
        hf_api_key = os.getenv("HuggingFace_API_KEY")
        
        if not hf_api_key:
            logger.warning("HuggingFace_API_KEY environment variable is missing.")
        
        # Uses HuggingFace's active Serverless Inference endpoint
        model = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=hf_api_key,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        logger.info("HuggingFace API embeddings initialized ✅✅✅")
        return model

    except Exception as e:
        error_message = CustomException("Failed to load HuggingFace API Embeddings.", e)
        logger.error(str(error_message))
        raise error_message