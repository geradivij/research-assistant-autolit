from src.autolit.llm.client import chat

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Give me 2 emojis representing AI research."}
]

print(chat(messages))
