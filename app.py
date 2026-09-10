import streamlit as st
from agent import LearningAgent
from rag import KnowledgeBase

st.set_page_config(
    page_title="AI Learning & Study Assistant",
    page_icon="🎓"
)

st.title("🎓 AI Learning & Study Assistant")
st.write("Learn smarter with AI, RAG, Memory and Study Tools.")

# Initialize
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = KnowledgeBase()

if "agent" not in st.session_state:
    st.session_state.agent = LearningAgent(
        st.session_state.knowledge_base
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:

    st.header("📚 Course Material")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    if uploaded_file:
        if st.button("Add PDF"):
            result = st.session_state.knowledge_base.add_pdf(
                uploaded_file
            )
            st.success(result)

    st.divider()

    st.header("🧰 Available Tools")
    st.write("📖 Course Material Search")
    st.write("📅 Study Plan Generator")
    st.write("📝 Quiz Generator")
    st.write("🧠 Student Memory")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.agent.clear_memory()
        st.rerun()


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
user_input = st.chat_input(
    "Ask your study question..."
)

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("AI is thinking..."):

            response = st.session_state.agent.run(
                user_input
            )

        st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })