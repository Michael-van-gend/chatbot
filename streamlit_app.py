import streamlit as st
import json
import numpy as np
from openai import OpenAI

# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="Cloud RAG Demo", layout="wide")
st.title("🧠 Cloud-Optimized RAG App (Fast & Lightweight)")
st.write("Ask a question. The app retrieves passages via precomputed embeddings and uses GPT for generation.")

# -----------------------------
# Load data + embeddings
# -----------------------------
@st.cache_resource
def load_corpus_and_embeddings():
    # Load corpus
    with open("collection/sampled_collection.jsonl", "r", encoding="utf-8") as f:
        corpus = {}
        for line in f:
            d = json.loads(line)
            corpus[d["id"]] = d["contents"]

    # Load precomputed embeddings (saved as a dict of {id: embedding})
    embeddings = np.load("collection/embeddings.npy", allow_pickle=True).item()
    return corpus, embeddings

corpus, embeddings = load_corpus_and_embeddings()

# -----------------------------
# Retrieval
# -----------------------------
def retrieve_top_docs(query_emb, embeddings, corpus, top_k=3):
    scores = []
    for pid, p_emb in embeddings.items():
        score = float(np.dot(query_emb, p_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(p_emb)))
        scores.append((pid, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:top_k]
    return [{"id": pid, "content": corpus[pid], "score": round(score, 3)} for pid, score in top]

# -----------------------------
# Query embedding (using OpenAI)
# -----------------------------
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["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"])

def encode_query_openai(query):
    client = get_openai_client()
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding)

# -----------------------------
# Answer generation (OpenAI)
# -----------------------------
def generate_answer_openai(query, retrieved_docs):
    client = get_openai_client()
    context = "\n\n".join([d["content"] for d in retrieved_docs])
    prompt = f"Answer the question based only on the passages below.\n\n{context}\n\nQuestion: {query}\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a concise and factual assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=256,
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()

# -----------------------------
# UI
# -----------------------------
query = st.text_input("💬 Your question:", placeholder="e.g. Who discovered penicillin?")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            query_emb = encode_query_openai(query)
            top_docs = retrieve_top_docs(query_emb, embeddings, corpus, top_k=3)
            answer = generate_answer_openai(query, top_docs)

        st.subheader("🧩 Generated Answer")
        st.success(answer)

        st.markdown("---")
        st.subheader("📚 Top-3 Retrieved Passages")
        for i, doc in enumerate(top_docs, 1):
            with st.expander(f"Passage {i} (score={doc['score']})"):
                st.write(doc["content"])
