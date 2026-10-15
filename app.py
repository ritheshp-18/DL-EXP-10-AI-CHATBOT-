
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

def get_response(question):
    q = question.lower().strip().rstrip("?!.")

    # Greetings
    if q in ["hi", "hello", "hey", "good morning", "good evening"]:
        return "Hello! Welcome to the AI Chatbot. How can I help you today?"

    # Common factual questions
    if "capital of france" in q:
        return "The capital of France is Paris."

    if "capital of india" in q:
        return "The capital of India is New Delhi."

    # Artificial intelligence
    if "what is artificial intelligence" in q or q == "define ai":
        return (
            "Artificial intelligence (AI) is a branch of computer science "
            "that enables machines to perform tasks that normally require "
            "human intelligence. Examples include speech recognition, "
            "image recognition, recommendation systems, and virtual assistants."
        )

    # Machine learning
    if "what is machine learning" in q or q == "define machine learning":
        return (
            "Machine learning (ML) is a branch of AI that allows computers "
            "to learn patterns from data and make predictions or decisions. "
            "For example, an email system can learn to identify spam messages "
            "from previously labelled emails."
        )

    # Deep learning
    if "what is deep learning" in q:
        return (
            "Deep learning is a type of machine learning that uses neural "
            "networks with multiple layers to learn complex patterns. "
            "It is used in image recognition, speech processing, and "
            "language applications."
        )

    # Job guidance
    if "looking for a job" in q or "looking for a job" in question.lower():
        return (
            "I can help you plan your job search. Update your resume, "
            "identify suitable roles, practise interview questions, and "
            "apply through trusted job portals. What type of job are you "
            "looking for?"
        )

    # Movie plans
    if "movies" in q or "movie" in q or "cinema" in q:
        return (
            "That sounds like a fun plan! You could choose a movie, check "
            "showtimes at nearby theatres, compare ticket prices, and book "
            "your seats. What kind of movie would you like to watch?"
        )

    # Other questions: use the language model
    try:
        chatbot = load_model()

        prompt = (
            "Answer the question helpfully in two or three complete "
            "sentences using simple English. Do not repeat the question. "
            "Question: " + question
        )

        result = chatbot(
            prompt,
            max_new_tokens=100,
            do_sample=False,
            truncation=True
        )

        answer = result[0]["generated_text"].strip()

        if not answer or answer.lower() == prompt.lower():
            return "I'm not sure about that. Could you ask in a different way?"

        return answer

    except Exception as e:
        return f"Sorry, I couldn't generate a response. Please try again. Details: {e}"


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.subheader("Chat Settings")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
question = st.chat_input("Ask me a question...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_response(question)
            st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
