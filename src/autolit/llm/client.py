# src/autolit/llm/client.py

from typing import List, Dict
from src.llm.llama_client import get_llm

# Create a shared Llama instance (or recreate each call — both ok)
# But keeping a single instance is cleaner.
llm = get_llm()

def chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
) -> str:
    """
    Universal Llama chat helper.

    Accepts a list of messages:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]

    Returns the text content of the model's reply.
    """

    response = llm.invoke(
        messages,
        temperature=temperature,
    )

    # LangChain ChatOllama returns a BaseMessage
    # with `.content` attribute.
    return response.content
