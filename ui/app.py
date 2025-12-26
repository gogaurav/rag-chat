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

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

# ---------------------------
# Sidebar (Sources Panel)
# ---------------------------

st.sidebar.title("📚 Retrieved Sources")
st.sidebar.caption("Documents used to generate the answer")

if st.session_state.last_sources:
    for src in st.session_state.last_sources:
        st.sidebar.markdown(f"- `{src}`")
else:
    st.sidebar.info("Ask a question to see sources here.")

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

                data = response.json()
                answer = data.get("answer", "No answer returned.")
                sources = data.get("sources", [])
                prompt_text = data.get("prompt", "")

                st.session_state.last_sources = sources
                st.session_state.last_prompt = prompt_text

            except Exception as e:
                answer = f"⚠️ Error contacting server: {e}"
                st.session_state.last_sources = []

        # Typing animation
        placeholder = st.empty()
        typed = ""
        for ch in answer:
            typed += ch
            placeholder.markdown(typed)
            time.sleep(0.004)

    with st.expander("🧠 View full prompt sent to GPT-4o", expanded=False):
        if st.session_state.last_prompt:
            st.text_area(
                label="Prompt",
                value=st.session_state.last_prompt,
                height=350
            )
            st.caption("This is the exact prompt sent to the LLM (question + retrieved context).")
        else:
            st.info("Ask a question to view the generated prompt.")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

# ---------------------------
# Footer
# ---------------------------

st.divider()
st.caption(
    "Powered by FAISS + LangChain + LiteLLM | Transparent RAG with source attribution"
)
