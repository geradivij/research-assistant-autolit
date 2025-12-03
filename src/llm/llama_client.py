from langchain_community.chat_models import ChatOllama

def get_llm(model_name: str = "llama3", temperature: float = 0.2):
    """
    Returns a configured LangChain ChatOllama instance.
    Central place to tweak model, temperature, etc.
    """
    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
    )
    return llm
