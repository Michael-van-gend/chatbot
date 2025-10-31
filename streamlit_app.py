# app.py
import streamlit as st
import torch
import json
import numpy as np
import csv
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Setup
# -----------------------------
st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("🧠 DPR + LLM Retrieval-Augmented Generation")
st.write("Ask a question. The app retrieves passages using DPR embeddings and generates an answer using TinyLlama.")

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_models():
    # DPR Encoder
    dpr_encoder = BertModel.from_pretrained("ielabgroup/StandardBERT-DR").to(device).eval()
    dpr_tokenizer = BertTokenizer.from_pretrained("ielabgroup/StandardBERT-DR")

    # LLM for generation
    llm = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    gen_model = AutoModelForCausalLM.from_pretrained(llm).to(device)
    gen_tokenizer = AutoTokenizer.from_pretrained(llm)

    return dpr_encoder, dpr_tokenizer, gen_model, gen_tokenizer

dpr_encoder, dpr_tokenizer, gen_model, gen_tokenizer = load_models()

# -----------------------------
# Load collection
# -----------------------------
@st.cache_resource
def load_collection():
    corpus = {}
    with open("collection/sampled_collection.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["id"]] = d["contents"]
    return corpus

corpus = load_collection()

# -----------------------------
# Precompute passage embeddings
# -----------------------------
@st.cache_resource
def compute_passage_embeddings(corpus):
    passage_embeddings = {}
    with torch.no_grad():
        for pid, text in tqdm(corpus.items(), desc="Encoding passages"):
            inputs = dpr_tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
            emb = dpr_encoder(**inputs)[0][:, 0, :]  # CLS token
            passage_embeddings[pid] = emb.cpu().numpy().flatten()
    return passage_embeddings

passage_embeddings = compute_passage_embeddings(corpus)

# -----------------------------
# Helper functions
# -----------------------------
def encode_query(query):
    inputs = dpr_tokenizer([query], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        emb = dpr_encoder(**inputs)[0][:, 0, :]
    return emb.cpu().numpy().flatten()

def retrieve_top_docs(query, top_k=3):
    q_emb = encode_query(query)
    scores = []
    for pid, p_emb in passage_embeddings.items():
        score = float(np.dot(q_emb, p_emb))
        scores.append((pid, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:top_k]
    return [{"id": pid, "content": corpus[pid], "score": round(score, 3)} for pid, score in top]

def generate_answer(query, retrieved_docs):
    context = ""
    for d in retrieved_docs:
        context += "- " + d["content"] + "\n"

    messages = [
        {"role": "system", "content": "You are a concise and factual assistant."},
        {"role": "user", "content": f"Answer this question using only the information below. Do not explain.\n\n{context}\nQuestion: {query}"}
    ]
    txt = gen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = gen_tokenizer.encode(txt, return_tensors="pt").to(device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        out = gen_model.generate(input_ids, max_new_tokens=256, temperature=0.7)
    resp = gen_tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    return resp.strip()

# -----------------------------
# Streamlit UI
# -----------------------------
query = st.text_input("💬 Your question:", placeholder="e.g. Who discovered penicillin?")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            top_docs = retrieve_top_docs(query, top_k=3)
            answer = generate_answer(query, top_docs)

        st.subheader("🧩 Generated Answer")
        st.success(answer)

        st.markdown("---")
        st.subheader("📚 Top-3 Retrieved Passages")
        for i, doc in enumerate(top_docs, 1):
            with st.expander(f"Passage {i} (score={doc['score']})"):
                st.write(doc["content"])
