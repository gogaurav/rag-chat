import streamlit as st
import requests
import time

# ---------------------------
# Config
# ---------------------------

API_URL = "http://localhost:8000/ask"
APP_TITLE = "GitLab Handbook Assistant"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📘",
    layout="centered"
)

# ---------------------------
# Session State
# ---------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# Header
# ---------------------------

st.markdown(
    f"""
    <h2 style="text-align:center;">📘 {APP_TITLE}</h2>
    <p style="text-align:center; color:gray;">
        Ask questions about GitLab policies, HR guidelines, and company processes
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------------------
# Chat History
# ---------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------
# Input Box
# ---------------------------

user_input = st.chat_input("Ask a question from the GitLab handbook...")

if user_input:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching handbook..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": user_input},
                    timeout=60
                )
                response.raise_for_status()
                answer = response.json()["answer"]

            except Exception as e:
                answer = f"⚠️ Error contacting server: {e}"

        # Typing animation (optional but nice)
        placeholder = st.empty()
        typed_text = ""
        for char in answer:
            typed_text += char
            placeholder.markdown(typed_text)
            time.sleep(0.005)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

# ---------------------------
# Footer
# ---------------------------

st.divider()
st.caption(
    "Powered by FAISS + LangChain + LiteLLM | Built as a RAG demo"
)
