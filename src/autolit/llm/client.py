from typing import List, Dict, Optional
from src.llm.llama_client import get_llm

llm = get_llm()

def chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
) -> str:
    """
    Universal chat helper using Groq.

    Accepts a list of messages:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]

    Returns the text content of the model's reply.
    """
    response = llm.invoke(messages)
    return response.content
