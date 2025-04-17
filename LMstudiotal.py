import streamlit as st
import json
import requests

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="UI Chatbot with LM Studio", page_icon="🤖", layout="wide")

# --- LM STUDIO CONFIG ---
LM_STUDIO_API = "http://localhost:1234/v1/chat/completions"
MODEL_ID = "mathstral-7b-v0.1"  # Replace with your LM Studio model identifier

# --- LOAD CHUNKS FROM JSON FILE ---
@st.cache_resource
def load_chunks(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks

chunks = load_chunks("all_chunks_logGPT.json")  # Replace with your actual path

# --- HELPER TO QUERY LLM ---
def ask_llm(query, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a frontend assistant trained to understand structured metadata chunks from UI pages.

The context consists of short text chunks describing:
- Pages and their descriptions.
- UI flows and step information (e.g., flow name, step name, step id).
- UI elements (e.g., input fields, buttons) with label, type, selector, and required info.

🚫 DO NOT hallucinate or infer structure like JSON or nested objects.
✅ Only use the actual chunks given — treat them as ground truth.
✅ When asked about a step, use the chunk that mentions the step name or step ID, and list the UI elements (inputs, buttons) that appear **after it** until the next step or page info chunk.

--- CHUNKS START ---
{context}
--- CHUNKS END ---

Now answer the following question based on the above chunks only:

Question: {query}
Answer:
"""

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        res = requests.post(LM_STUDIO_API, headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- MAIN UI ---
st.title("🤖 LM Studio UI Assistant")
st.markdown("Ask anything about UI page structure, steps, and fields!")

user_input = st.chat_input("Ask a question...")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("Thinking..."):
        response = ask_llm(user_input, chunks)
    st.session_state.chat_history.append(("assistant", response))

# --- DISPLAY CHAT HISTORY ---
for sender, message in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(message)

# --- OPTIONAL: Preview Raw Chunks ---
with st.expander("📄 View Raw Chunks"):
    for i, chunk in enumerate(chunks):
        st.markdown(f"**Chunk {i+1}:** {chunk}")
