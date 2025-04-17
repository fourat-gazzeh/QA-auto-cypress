import json
import numpy as np
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
JSON_FILES = ["MarketOrderStep1.json", "MarketOrderStep2.json"]  # Add as many as needed
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_DIR = "embedding_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD JSON DATA ---
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --- EXTRACT TEXT CHUNKS FROM A JSON FILE ---
def extract_chunks(data):
    chunks = []

    if "page_info" in data:
        page = data["page_info"]
        page_desc = f"Page Name: {page['name']}\nURL: {page.get('url_path', '')}\nDescription: {page['description']}"
        chunks.append(page_desc)

        # Extract flow info if present
        if "flow" in page:
            flow = page["flow"]
            flow_text = f"Flow Name: {flow['name']}\nStep ID: {flow['step_id']}\nStep Name: {flow['step_name']}"
            chunks.append(flow_text)

    for el in data.get("elements", []):
        label = el.get("lable") or el.get("placeholder") or "No label"
        desc = f"{el['tag']} (type={el['type']}, label={label}, required={el['required']}) with selector: {el['cypress_selector']}"
        chunks.append(desc)

    chunks.extend(data.get("scenarios", []))
    return chunks

# --- COMBINE CHUNKS FROM ALL FILES ---
all_chunks = []
for path in JSON_FILES:
    data = load_json(path)
    chunks = extract_chunks(data)
    all_chunks.extend(chunks)

# --- GENERATE EMBEDDINGS ---
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embeddings = embed_model.encode(all_chunks)
embeddings_np = np.array(embeddings)

# --- BUILD FAISS INDEX ---
dim = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings_np)

# --- SAVE ARTIFACTS ---
with open(os.path.join(OUTPUT_DIR, "chunks.pkl"), "wb") as f:
    pickle.dump(all_chunks, f)

with open(os.path.join(OUTPUT_DIR, "model_name.txt"), "w") as f:
    f.write(EMBEDDING_MODEL_NAME)

faiss.write_index(index, os.path.join(OUTPUT_DIR, "faiss.index"))

print(f"✅ Saved FAISS index, chunks, and model info to `{OUTPUT_DIR}`.")
