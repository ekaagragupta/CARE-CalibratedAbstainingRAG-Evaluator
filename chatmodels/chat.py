import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_xxxxxxxxxxxxxxxxx"
from urllib import response

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint


llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    temperature=0.7
)
model=ChatHuggingFace(llm=llm)

response=model.invoke("What is the capital of France?")
print(response.content)