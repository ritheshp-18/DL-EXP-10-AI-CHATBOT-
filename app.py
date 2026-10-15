
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Chatbot")
st.write("Experiment 10 | Roll Number: 24BAD098")
st.caption("An AI chatbot for general questions and conversation.")

@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Chat Settings")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Ask me a question...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                chatbot = load_model()

 result = chatbot(
    "Give a helpful, accurate answer in 2 to 4 complete sentences. "
    "For greetings, respond warmly. For factual questions, explain "
    "the answer simply. For requests or plans, suggest useful next steps. "
    "Question: " + prompt,
    max_new_tokens=120,
    do_sample=False
)

                answer = result[0]["generated_text"].strip()

                if not answer:
                    answer = "I couldn't generate an answer. Please try again."

            except Exception as e:
                answer = (
                    "Sorry, the chatbot could not generate a response. "
                    "Please try again later."
                )
                st.error(f"Model error: {e}")

            st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
