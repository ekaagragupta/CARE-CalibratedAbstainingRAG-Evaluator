from langchain_openai import OpenAIEmbeddings

embeddings=OpenAIEmbeddings(
    models="text-embedding-3-large",
    OPENAI_API_KEY="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",
    dimension=30
)
vector=embeddings.embed_query("What is the capital of india?")
print(vector)