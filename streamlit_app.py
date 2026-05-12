import streamlit as st
from src.search import RAGSearch

st.set_page_config(
    page_title="Python AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Python AI Assistant")
st.markdown("Ask questions about Python documentation.")

rag = RAGSearch()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask something about Python...")
if user_input:

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    answer = rag.search_and_summarize(user_input)

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )