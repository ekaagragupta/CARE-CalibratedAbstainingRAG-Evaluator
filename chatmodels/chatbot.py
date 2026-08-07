import os
os.environ["HUGGINGFACE_API_TOKEN"] = "hf_xxxxxxxxxxxxxxxxx"

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core import AIMessage, HumanMessage, SystemMessage

llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    temperature=0.7
)
model=ChatHuggingFace(llm=llm)
messages=[
    SystemMessage(content="You are a genz assistant.")
]
while True:
    print("---------welcome to chat bot---------")
    prompt=input("you:")
    messages.append(HumanMessage(content=prompt))
    if(prompt=="0"):
        break
    response=model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("bot",response.content)