
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

st.title("🤖 AI Chatbot")
st.write("Experiment 10 | Roll Number: 24BAD098")

@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2")

try:
    chatbot = load_model()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Type your message here...")

    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.write(prompt)

        context = "\n".join(
            f'{m["role"]}: {m["content"]}'
            for m in st.session_state.messages[-6:]
        )

        result = chatbot(
            context + "\nassistant:",
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            pad_token_id=chatbot.tokenizer.eos_token_id
        )

        answer = result[0]["generated_text"].split(
            "assistant:"
        )[-1].strip()

        if not answer:
            answer = "Please try asking that another way."

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.write(answer)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

except Exception as e:
    st.error(f"Unable to load the chatbot: {e}")
