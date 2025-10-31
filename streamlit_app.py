import streamlit as st, os, json, numpy as np, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Basic setup
# -----------------------------
st.set_page_config(page_title="RAG Information Retrieval App", layout="wide")

st.title("RAG Information Retrieval App")
st.write(
    "Ask a question to retrieve the top-3 most relevant documents "
    "and generate a response using a small local language model."
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models 
encoder = st.cache_resource(lambda: SentenceTransformer("all-MiniLM-L6-v2"))()
llm_name = "TheBloke/TinyLlama‑1.1B‑intermediate‑step‑715k‑1.5T‑AWQ"
gen_tokenizer, gen_model = st.cache_resource(lambda: (
    AutoTokenizer.from_pretrained(llm_name),
    AutoModelForCausalLM.from_pretrained(
        llm_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(device)
))()
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

# -----------------------------
# Streamlit UI
# -----------------------------
query = st.text_input("Enter your question:", placeholder="e.g. Who discovered penicillin?")

if st.button("Submit"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            # Retrieval
            q_emb = encoder.encode(query, normalize_embeddings=True)
            scores = [(pid, float(np.dot(q_emb, p_emb))) for pid, p_emb in passage_embeddings.items()]
            scores.sort(key=lambda x: x[1], reverse=True)
            top_docs = [{"id": pid, "content": corpus[pid], "score": round(score, 3)} for pid, score in scores[:3]]

            # Generation
            if not top_docs:
                answer = "No retrieved passages available."
            else:
                context = "\n".join([f"- {d['content']}" for d in top_docs])
                prompt = f"Answer this question using only the context below.\n\n{context}\n\nQuestion: {query}\nAnswer:"
                try:
                    input_ids = gen_tokenizer.encode(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = gen_model.generate(input_ids, max_new_tokens=100, temperature=0.7)
                    answer = gen_tokenizer.decode(out[0], skip_special_tokens=True).strip()
                except Exception as e:
                    st.error("Answer generation failed:")
                    st.code(traceback.format_exc())
                    answer = "An error occurred during text generation."

        st.subheader("Generated Answer")
        st.success(answer)

        if top_docs:
            st.markdown("---")
            st.subheader("Top 3 Retrieved Documents")
            for i, doc in enumerate(top_docs, 1):
                with st.expander(f"Passage {i} (score={doc['score']})"):
                    st.write(doc["content"])
