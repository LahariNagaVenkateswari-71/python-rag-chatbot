import streamlit as st
from src.search import RAGSearch

st.set_page_config(
    page_title="Python AI Assistant",
    page_icon="🤖",
    layout="centered")

@st.cache_resource
def load_rag():
    return RAGSearch()

rag = load_rag()

st.title("🤖 Python AI Assistant")
st.markdown("Ask questions about Python programming and documentation.")

# Sidebar
with st.sidebar:

    st.title("⚙️ Settings")

    st.markdown("### About")
    st.write(
        "AI-powered Python documentation assistant using RAG."
    )

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input(
    "Ask anything..."
)

if user_input:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )
  
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):

        answer = rag.search_and_summarize(
            user_input,
            chat_history=st.session_state.messages
        )

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )