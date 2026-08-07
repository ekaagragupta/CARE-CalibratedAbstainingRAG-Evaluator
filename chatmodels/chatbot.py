import os
os.environ["HUGGINGFACE_API_TOKEN"] = "hf_xxxxxxxxxxxxxxxxx"

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint


llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    temperature=0.7
)
model=ChatHuggingFace(llm=llm)
while True:
    print("---------welcome to chat bot---------")
    prompt=input("you:")
    if(prompt=="0"):
        break
    response=model.invoke(prompt)
    print("bot",response.content)