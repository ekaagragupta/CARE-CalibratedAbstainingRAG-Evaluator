from langchain_huggingface import HuggingFaceEmbeddings

HuggingFaceEmbeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

text=[
    "hello world",
    "What is the capital of India?",
    "What is the capital of France?"
]

vector=HuggingFaceEmbeddings.embed_documents(text)
print(vector)