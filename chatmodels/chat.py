from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain.chat_models import init_chat_model

init_chat_model("gpt-4.1")  # Initialize the chat model