import streamlit as st
import json
import google.generativeai as genai

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="UI Gemini Chatbot", page_icon="🤖", layout="wide")

# --- GEMINI CONFIG ---
GEMINI_API_KEY = "AIzaSyBG3uQvT_K0bXsbFbHtIVsckwpvnKfc3m0"  # Replace with your own key
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-1.5-flash")

# --- LOAD JSON CHUNKS ---
@st.cache_resource
def load_chunks(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

chunks = load_chunks("all_chunks_logGPT.json")  # <-- change to your actual file

# --- GEMINI QUERY ---
def ask_gemini(query, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful and friendly testing assistant specialized in frontend QA.

Your role is to assist testers by understanding UI flows and generating Cypress test code based on structured metadata chunks.

The chunks describe:
- Pages and their purposes.
- UI flows and step info (flow name, step ID, step name).
- UI elements (like inputs and buttons) with label, type, selector, and required info.
- Valid (✅) and invalid (❌) user behavior examples.

🎯 Your main tasks:
1. Help testers understand the purpose and structure of UI flows and steps.
2. Explain what each UI step includes (fields, buttons, etc.).
3. Generate Cypress test code for valid and invalid user behavior scenarios based on the chunks.

🧠 You must only use the chunks provided. Do not invent any elements or details.
✅ Always base your answer strictly on the chunks.
✅ For Cypress code, use actual selectors and UI labels from the chunks.
✅ Include both success and failure test cases when user behavior chunks (✅ / ❌) exist.

🗨️ If the user's question is not related to the UI, flows, or testing (e.g. “hello”, “how are you”), respond politely as a friendly assistant would. Examples:
- “Hello! I’m here to help with UI testing — feel free to ask anything!”
- “I’m good! Ready to help you write Cypress tests whenever you are.”


--- CHUNKS START ---
{context}
--- CHUNKS END ---

Now answer the following question based on the above chunks only:

Question: {query}
Answer:
"""
    response = gemini.generate_content(prompt)
    
    # Show used chunks for transparency
    with st.expander("🧩 Chunks used in this query"):
        for chunk in context_chunks:
            st.markdown(f"- {chunk}")

    return response.text.strip()

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "saved_chats" not in st.session_state:
    st.session_state.saved_chats = []

# --- SIDEBAR CHAT MANAGEMENT ---
with st.sidebar:
    st.header("💬 Chat Sessions")
    for i, chat in enumerate(st.session_state.saved_chats):
        if st.button(f"Chat {i + 1}"):
            st.session_state.chat_history = chat.copy()
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
    if st.button("💾 Save Current Chat"):
        st.session_state.saved_chats.append(st.session_state.chat_history.copy())
def generate_documentation(context_chunks):
    context = "\n\n".join(context_chunks)

    doc_prompt = f"""
You are a frontend documentation expert.

Your task is to generate **clear and structured documentation** for QA testers based on the following UI flow metadata chunks.

The documentation should include:
1. 🧾 A high-level overview of what this flow is and its purpose.
2. 🗂️ A breakdown of each step (step name and ID) with the UI elements it contains (inputs, buttons) and any relevant notes.
3. 🎯 Valid and ❌ invalid user behavior examples (if available).
4. ✅ Practical usage hints and edge cases to consider when testing.
5. 🔍 Mention any mandatory fields and their selectors.
6. 🧪 This document will help testers understand what to test and how.

⚠️ You MUST ONLY use the info in the chunks. Do not invent UI or steps.

--- CHUNKS START ---
{context}
--- CHUNKS END ---

Now generate a full documentation guide for this UI flow based on the above chunks.
"""
    response = gemini.generate_content(doc_prompt)
    return response.text.strip()

# --- FLOW DOC GENERATOR BUTTON ---
st.markdown("---")
st.subheader("📘 Flow Documentation Generator")

if st.button("🛠️ Generate Documentation for Current Flow"):
    with st.spinner("Generating documentation..."):
        documentation = generate_documentation(chunks)
    st.markdown("### 📄 Flow Documentation")
    st.code(documentation, language="markdown")
    
# --- MAIN UI ---
st.title("🤖 Frontend QA Assistant")
st.markdown("Ask anything about UI structure, flows, or frontend metadata!")

with st.chat_message("assistant"):
    st.markdown("Hi! I'm your UI assistant. Ask me about the page structure, steps, components or behavior.")

user_input = st.chat_input("Ask a question...")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("Thinking..."):
        response = ask_gemini(user_input, chunks)
    st.session_state.chat_history.append(("assistant", response))

# --- DISPLAY CHAT HISTORY ---
for sender, message in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(message)

# --- OPTIONAL: Preview All Chunks ---
with st.expander("📄 View All Loaded Chunks"):
    for i, chunk in enumerate(chunks):
        st.markdown(f"**Chunk {i+1}:** {chunk}")
