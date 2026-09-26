import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

def load_css():
    css_path = Path(__file__).parent / "style.css"

    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    provider="groq",
    task="text-generation",
    max_new_tokens=1000
)

model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="chat_history")
])

st.set_page_config(page_title="My AI Chatbot")
with st.sidebar:
    st.markdown("Phoenix AI")
    st.write("Your personal AI assistant")

    st.divider()

    if st.button("＋ New Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.caption("Powered by LangChain")
st.markdown(
    """
    <h1 style="text-align:center;">
        Phoenix AI
    </h1>
    <p class="subtitle" style="text-align:center;">
        Your intelligent AI assistant, powered by LangChain
    </p>
    """,
    unsafe_allow_html=True
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)

    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

question = st.chat_input("Type your message...")

if question:
    with st.chat_message("user"):
        st.write(question)

    st.session_state.chat_history.append(
        HumanMessage(content=question)
    )

    message = prompt.invoke({
        "chat_history": st.session_state.chat_history
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = model.invoke(message)
            st.write(result.content)

    st.session_state.chat_history.append(
        AIMessage(content=result.content)
    )