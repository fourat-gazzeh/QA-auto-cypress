import streamlit as st
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import json

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title="UI Chatbot (from FAISS)", page_icon="🤖", layout="wide")

# --- CONFIGURATION ---
GEMINI_API_KEY = "AIzaSyBG3uQvT_K0bXsbFbHtIVsckwpvnKfc3m0"  # Replace with your own
EMBEDDING_DIR = "embedding_data"  # Folder from your previous script

# --- LOAD SAVED FAISS + EMBEDDINGS ---
@st.cache_resource
def load_faiss_and_chunks(path):
    index = faiss.read_index(os.path.join(path, "faiss.index"))

    with open(os.path.join(path, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)

    with open(os.path.join(path, "model_name.txt"), "r") as f:
        model_name = f.read().strip()

    model = SentenceTransformer(model_name)
    return model, index, chunks

embed_model, index, chunks = load_faiss_and_chunks(EMBEDDING_DIR)

with open("all_chunks_log.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)
# --- SETUP GEMINI ---
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-1.5-flash")

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "saved_chats" not in st.session_state:
    st.session_state.saved_chats = []

# --- GEMINI QUERY FUNCTION ---
def ask_gemini(query, top_k=10):
    q_embedding = embed_model.encode([query])
    _, indices = index.search(np.array(q_embedding), top_k)
    selected_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(selected_chunks)

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
    response = gemini.generate_content(prompt)
    
    # Display chunks used in Streamlit
    with st.expander("🧩 Chunks used for this query"):
        for chunk in selected_chunks:
            st.markdown(f"- {chunk}")
    
    return response.text.strip()
# --- SIDEBAR ---
with st.sidebar:
    st.header("💬 Chat Sessions")
    for i, chat in enumerate(st.session_state.saved_chats):
        if st.button(f"Chat {i + 1}"):
            st.session_state.chat_history = chat.copy()
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
    if st.button("💾 Save Current Chat"):
        st.session_state.saved_chats.append(st.session_state.chat_history.copy())

# --- MAIN UI ---
st.title("🤖 Gemini Chatbot for UI JSON")
st.markdown("Ask anything about UI structure, components, flows, or usage!")

with st.chat_message("assistant"):
    st.markdown("Hi! I'm your UI assistant. Ask me about the page structure, flows, elements or anything else.")

user_input = st.chat_input("Ask a question...")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("Thinking..."):
        response = ask_gemini(user_input)
    st.session_state.chat_history.append(("assistant", response))

# --- DISPLAY CHAT HISTORY ---
for sender, message in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(message)

# --- Optional chunk preview ---
with st.expander("📄 View Raw Chunks Used in Embedding"):
    for i, c in enumerate(chunks):
        st.markdown(f"**Chunk {i+1}:** {c}")
