from src.llm.llama_client import get_llm

def main():
    llm = get_llm()
    resp = llm.invoke("Give me 3 bullet points about what an AI agent is.")
    print("MODEL RESPONSE:\n")
    print(resp)

if __name__ == "__main__":
    main()
