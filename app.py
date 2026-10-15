
import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Application heading
st.title("🤖 AI Chatbot")
st.write("Experiment 10 | Roll Number: 24BAD098")
st.caption("An AI chatbot for general questions and conversation.")

# Load the model once and reuse it
@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar settings
with st.sidebar:
    st.subheader("Chat Settings")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
prompt = st.chat_input("Ask me a question...")

if prompt:
    # Display and save the user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.write(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chatbot = load_model()

                instruction = (
                    "You are a helpful AI assistant. "
                    "Give a helpful, accurate answer in 2 to 4 complete "
                    "sentences. For greetings, respond warmly. "
                    "For factual questions, explain the answer simply. "
                    "For requests or plans, suggest useful next steps. "
                    "If you do not know an answer, say so clearly. "
                    "Question: " + prompt
                )

                result = chatbot(
                    instruction,
                    max_new_tokens=120,
                    do_sample=False,
                    truncation=True
                )

                answer = result[0]["generated_text"].strip()

                if not answer:
                    answer = (
                        "I couldn't generate an answer. "
                        "Please try asking in a different way."
                    )

                st.write(answer)

            except Exception as e:
                answer = (
                    "Sorry, I couldn't generate a response. "
                    "Please try again later."
                )
                st.error(f"Model error: {e}")
                st.write(answer)

    # Save the assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
