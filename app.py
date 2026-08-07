import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Initialize model only once
@st.cache_resource
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1-0528",
        temperature=0.7,
    )
    return ChatHuggingFace(llm=llm)

model = load_model()

# Page configuration
st.set_page_config(
    page_title="AI Chat Bot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Chat Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a Gen Z assistant.")
    ]

# Display previous messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# User input
prompt = st.chat_input("Type your message...")

if prompt:

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Store user message
    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    # Generate response
    response = model.invoke(st.session_state.messages)

    # Store AI response
    st.session_state.messages.append(
        AIMessage(content=response.content)
    )

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(response.content)