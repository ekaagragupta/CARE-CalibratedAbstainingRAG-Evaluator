from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

response = model.invoke("Hello! Tell me a fun fact.")

print(response.content)