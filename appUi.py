import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

st.set_page_config(page_title="JSON UI Chatbot", page_icon="🤖", layout="wide")

# --- CONFIGURATION ---
GEMINI_API_KEY = "AIzaSyBG3uQvT_K0bXsbFbHtIVsckwpvnKfc3m0"  # Replace with your key
JSON_PATH = "outputdesc3+cypress.json"  # Replace with your JSON file path

# --- SETUP GEMINI ---
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# --- SESSION STATE FOR CHAT HISTORY ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "saved_chats" not in st.session_state:
    st.session_state.saved_chats = []

# --- LOAD JSON DATA ---
@st.cache_data
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_data(JSON_PATH)

# --- EXTRACT TEXT CHUNKS ---
@st.cache_data
def extract_chunks(data):
    chunks = []
    if "page_info" in data:
        page = data["page_info"]
        chunks.append(f"Page Name: {page['name']}\nURL: {page['url_path']}\nDescription: {page['description']}")
    for el in data["elements"]:
        label = el.get("lable") or el.get("placeholder") or "No label"
        desc = f"{el['tag']} (type={el['type']}, label={label}, required={el['required']}) with selector: {el['cypress_selector']}"
        chunks.append(desc)
    chunks.extend(data.get("scenarios", []))
    return chunks

chunks = extract_chunks(data)

# --- EMBEDDINGS & FAISS INDEX ---
@st.cache_resource
def create_index(chunks):
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(chunks)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return embed_model, index, embeddings

embed_model, index, embeddings = create_index(chunks)

# --- QUERY FUNCTION ---
def ask_gemini(query, top_k=5):
    q_embedding = embed_model.encode([query])
    _, indices = index.search(np.array(q_embedding), top_k)
    context = "\n".join([chunks[i] for i in indices[0]])
    prompt = f"""You are a frontend UI assistant trained to understand page metadata and form structures.

Context:
{context}

Question: {query}
Answer:"""
    response = model.generate_content(prompt)
    return response.text.strip()

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Past Conversations")
    for i, chat in enumerate(st.session_state.saved_chats):
        if st.button(f"Chat {i + 1}"):
            st.session_state.chat_history = chat.copy()
    if st.button("❌ Clear Chat"):
        st.session_state.chat_history = []
    if st.button("📂 Save Current Chat"):
        st.session_state.saved_chats.append(st.session_state.chat_history.copy())

# --- MAIN INTERFACE ---
st.title("💬 Gemini Chatbot for UI Page JSON")
st.markdown("Ask any question about the UI page components, selectors, or behavior.")

with st.chat_message("assistant"):
    st.markdown("Hello! Ask me anything about your UI page JSON.")

user_input = st.chat_input("Your question...")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("Thinking..."):
        response = ask_gemini(user_input)
    st.session_state.chat_history.append(("assistant", response))

# --- DISPLAY CHAT ---
for sender, message in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(message)

# --- Optional JSON preview ---
with st.expander("🔍 View Raw JSON"):
    st.json(data)
