# streamlit_app.py
import streamlit as st, pkg_resources
import torch
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

st.write("✅ Packages Installed:")
for pkg in ["sentence-transformers", "torch", "transformers"]:
    try:
        version = pkg_resources.get_distribution(pkg).version
        st.write(f"{pkg}: {version}")
    except Exception as e:
        st.write(f"{pkg}: not found ({e})")

# -----------------------------
# App setup
# -----------------------------
st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("🧠 Mini RAG (Local Embeddings + LLM)")
st.write("Ask a question. The app retrieves passages using local embeddings and generates an answer using TinyLlama — no API keys needed!")

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load embedding model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

encoder = load_embedding_model()

# -----------------------------
# Load LLM (TinyLlama)
# -----------------------------
@st.cache_resource
def load_llm():
    llm = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModelForCausalLM.from_pretrained(llm).to(device)
    tokenizer = AutoTokenizer.from_pretrained(llm)
    return model, tokenizer

gen_model, gen_tokenizer = load_llm()

# -----------------------------
# Load corpus
# -----------------------------
@st.cache_resource
def load_corpus():
    corpus = {}
    with open("collection/sampled_collection.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["id"]] = d["contents"]
    return corpus

corpus = load_corpus()

# -----------------------------
# Precompute embeddings
# -----------------------------
@st.cache_resource
def compute_embeddings(corpus):
    ids, texts = list(corpus.keys()), list(corpus.values())
    embs = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return dict(zip(ids, embs))

passage_embeddings = compute_embeddings(corpus)

# -----------------------------
# Retrieval
# -----------------------------
def retrieve_top_docs(query, top_k=3):
    q_emb = encoder.encode(query, normalize_embeddings=True)
    scores = [(pid, float(np.dot(q_emb, p_emb))) for pid, p_emb in passage_embeddings.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:top_k]
    return [{"id": pid, "content": corpus[pid], "score": round(score, 3)} for pid, score in top]

# -----------------------------
# Generation
# -----------------------------
def generate_answer(query, retrieved_docs):
    context = "\n".join([f"- {d['content']}" for d in retrieved_docs])
    messages = [
        {"role": "system", "content": "You are a concise and factual assistant."},
        {"role": "user", "content": f"Answer this question using only the context below. Be concise.\n\n{context}\n\nQuestion: {query}"}
    ]

    txt = gen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = gen_tokenizer.encode(txt, return_tensors="pt").to(device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        out = gen_model.generate(input_ids, max_new_tokens=200, temperature=0.7)
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
