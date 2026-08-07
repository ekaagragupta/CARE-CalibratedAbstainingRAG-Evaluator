from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
    temperature=0.7,
    max_output_tokens=256,
)

response =embeddings.invoke("write a poem on mom")

print(response.content)