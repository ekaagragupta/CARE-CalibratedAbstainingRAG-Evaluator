from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4.1")

response = model.invoke("Hello! Tell me a fun fact.")

print(response.content)