import os
import warnings

# Suppress Pydantic warnings
warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince20.*")
try:
    from pydantic import PydanticDeprecatedSince20
    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
except ImportError:
    pass

from langchain_cohere import ChatCohere, CohereEmbeddings
from dotenv import load_dotenv

load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")

# Initialize Cohere
llm = ChatCohere(
    model="command-a-03-2025", 
    cohere_api_key=cohere_api_key,
    temperature=0.3
)

embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=cohere_api_key
)
