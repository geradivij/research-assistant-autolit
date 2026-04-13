import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def get_llm(model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.2):
    """
    Returns a configured LangChain ChatGroq instance.
    Reads GROQ_API_KEY from environment / .env file.
    """
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=os.environ.get("GROQ_API_KEY"),
    )
