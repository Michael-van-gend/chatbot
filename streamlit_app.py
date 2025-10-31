import streamlit as st
import os, json, numpy as np, torch
from sentence_transformers import SentenceTransformer
import openai

st.set_page_config(page_title="RAG + OpenAI", layout="wide")

st.title("RAG Information Retrieval App")
st.write("Ask a question and get the top-3 relevant passages along with an AI-generated answer.")

device = "cuda" if torch.cuda.is_available() else "cpu"

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

encoder = st.cache_resource(lambda: SentenceTransformer("all-MiniLM-L6-v2"))()

corpus = st.cache_resource(lambda: {
    d["id"]: d["contents"] for d in
    [json.loads(line) for line in open("collection/sampled_collection.jsonl", "r", encoding="utf-8")]
})()

emb_path = "collection/embeddings.npy"
if os.path.exists(emb_path):
    passage_embeddings = st.cache_resource(lambda: np.load(emb_path, allow_pickle=True).item())()
else:
    ids, texts = list(corpus.keys()), list(corpus.values())
    embs = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    passage_embeddings = dict(zip(ids, embs))
    np.save(emb_path, passage_embeddings)

query = st.text_input("Enter your question:", placeholder="e.g. Who discovered penicillin?")

if st.button("Submit"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving documents and generating answer..."):
            q_emb = encoder.encode(query, normalize_embeddings=True)
            scores = [(pid, float(np.dot(q_emb, p_emb))) for pid, p_emb in passage_embeddings.items()]
            scores.sort(key=lambda x: x[1], reverse=True)
            top_docs = [{"id": pid, "content": corpus[pid], "score": round(score, 3)} for pid, score in scores[:3]]

            if not top_docs:
                answer = "No relevant passages were found."
            else:
                context = "\n".join([f"- {d['content']}" for d in top_docs])
                prompt = f"Answer the question using only the context below.\n\n{context}\n\nQuestion: {query}\nAnswer:"

                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=250,
                        temperature=0.7
                    )
                    
                    answer = response.choices[0].message.content
                except Exception as e:
                    st.error(f"OpenAI API request failed: {e}")
                    answer = "Could not generate an answer."

        st.subheader("Generated Answer")
        st.success(answer)

        if top_docs:
            st.markdown("---")
            st.subheader("Top Retrieved Passages")
            for i, doc in enumerate(top_docs, 1):
                with st.expander(f"Passage {i} (score={doc['score']})"):
                    st.write(doc["content"])
