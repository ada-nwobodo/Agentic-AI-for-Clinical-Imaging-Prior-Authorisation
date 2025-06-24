import streamlit as st
import requests
import os

# Hugging Face API setup
HF_API_KEY = os.getenv("HF_API_KEY")  # Secure way to store your token



API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

# System prompt
SYSTEM_PROMPT = """
You are a clinical assistant specializing in medical documentation for imaging prior authorizations. Your job is to help clinicians summarize patient symptoms, history, and guideline-based indications to justify imaging requests (e.g., MRI). Always structure your outputs clearly and include references to clinical criteria (e.g., NICE guidelines).
"""

# Query Hugging Face
def query_hf_model(prompt):
    payload = {
        "inputs": f"<s>[INST] {SYSTEM_PROMPT}\n\n{prompt} [/INST]",
        "options": {"wait_for_model": True}
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"].split("[/INST]")[-1].strip()
    else:
        return f"Error: {response.status_code} - {response.text}"

# Streamlit UI
st.title("🧠 Clinical Imaging Assistant (Mistral via Hugging Face)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask a clinical imaging question...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    reply = query_hf_model(user_input)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

