from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings=OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimension=64
)

texts=[
    "hello world",
    "What is the capital of India?",
    "What is the capital of France?"
]


vector=embeddings.embed_documents(texts)
print(vector)