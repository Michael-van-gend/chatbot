import streamlit as st
import os
import json, numpy as np
import torch
from sentence_transformers import SentenceTransformer
from openai import OpenAI

st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("Information Retrieval using DPR and RAG using Groq")

device = "cuda" if torch.cuda.is_available() else "cpu"

encoder = SentenceTransformer("all-MiniLM-L6-v2")

api_key = st.secrets["groq_key"]
os.environ["GROQ_API_KEY"] = api_key
client = OpenAI(api_key=os.environ.get("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")

corpus = {}
with open("collection/sampled_collection.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        corpus[d["id"]] = d["contents"]

emb_path = "collection/embeddings.npy"
if os.path.exists(emb_path):
    passage_embeddings = np.load(emb_path, allow_pickle=True).item()
else:
    ids, texts = list(corpus.keys()), list(corpus.values())
    embs = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    passage_embeddings = dict(zip(ids, embs))
    np.save(emb_path, passage_embeddings)

query = st.text_input("Your question:", placeholder="e.g. Who discovered penicillin?")

if st.button("Ask") and query.strip():
    q_emb = encoder.encode(query, normalize_embeddings=True)
    scores = [(pid, float(np.dot(q_emb, p_emb))) for pid, p_emb in passage_embeddings.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    top_docs = [{"id": pid, "content": corpus[pid], "score": round(score, 3)} for pid, score in scores[:3]]
    context = "\n".join([d['content'] for d in top_docs])
    prompt = f"Answer the question using only the context below.\n\n{context}\n\nQuestion: {query}\nAnswer:"

    try:
        response = client.responses.create(
            model="openai/gpt-oss-20b",
            input=prompt
        )
        answer = response.output_text
    except Exception as e:
        st.error(f"Groq API request failed: {e}")
        answer = "Could not generate an answer."

    st.subheader("Generated Answer")
    st.success(answer)

    if top_docs:
        st.markdown("---")
        st.subheader("Top-3 Retrieved Passages")
        for i, doc in enumerate(top_docs, 1):
            with st.expander(f"Passage {i} (score={doc['score']})"):
                st.write(doc["content"])
