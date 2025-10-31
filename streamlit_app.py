# streamlit_app.py
import streamlit as st, pkg_resources, traceback, os
import torch, json, numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Basic setup & diagnostics
# -----------------------------
st.set_page_config(page_title="RAG Demo", layout="wide")

st.title("🧠 Mini RAG (Local Embeddings + LLM)")
st.write("Ask a question. Retrieval uses precomputed local embeddings — no API keys needed!")

st.write("✅ Packages Installed:")
for pkg in ["sentence-transformers", "torch", "transformers"]:
    try:
        v = pkg_resources.get_distribution(pkg).version
        st.write(f"{pkg}: {v}")
    except Exception as e:
        st.write(f"{pkg}: ❌ {e}")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"🖥 Using device: `{device}`")

# -----------------------------
# Load embedding model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    try:
        st.write("⏳ Loading embedding model...")
        model = SentenceTransformer("HuggingFaceH4/zephyr-0.5b-beta")
        st.write("✅ Embedding model loaded.")
        return model
    except Exception as e:
        st.error("❌ Failed to load SentenceTransformer:")
        st.code(traceback.format_exc())
        raise e

encoder = load_embedding_model()

# -----------------------------
# Load tiny LLM (cloud-safe)
# -----------------------------
@st.cache_resource
def load_llm():
    try:
        st.write("⏳ Loading lightweight language model...")
        llm_name = "sshleifer/tiny-gpt2"
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
        model = AutoModelForCausalLM.from_pretrained(
            llm_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
        ).to(device)
        st.write("✅ LLM loaded successfully.")
        return model, tokenizer
    except Exception as e:
        st.error("❌ Failed to load LLM:")
        st.code(traceback.format_exc())
        raise e

gen_model, gen_tokenizer = load_llm()

# -----------------------------
# Load corpus
# -----------------------------
@st.cache_resource
def load_corpus():
    try:
        st.write("⏳ Loading corpus...")
        corpus = {}
        with open("collection/sampled_collection.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                corpus[d["id"]] = d["contents"]
        st.write(f"✅ Loaded {len(corpus)} passages.")
        return corpus
    except Exception as e:
        st.error("❌ Failed to load corpus file:")
        st.code(traceback.format_exc())
        return {}

corpus = load_corpus()

# -----------------------------
# Load or compute embeddings
# -----------------------------
@st.cache_resource
def load_or_compute_embeddings(corpus):
    try:
        emb_path = "collection/embeddings.npy"

        if os.path.exists(emb_path):
            st.write("📂 Found precomputed embeddings. Loading...")
            data = np.load(emb_path, allow_pickle=True).item()
            st.write(f"✅ Loaded {len(data)} embeddings.")
            return data
        else:
            st.write("⚙️ No precomputed embeddings found — computing...")
            ids, texts = list(corpus.keys()), list(corpus.values())
            embs = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
            emb_dict = dict(zip(ids, embs))
            np.save(emb_path, emb_dict)
            st.write("✅ Embeddings computed and saved to disk.")
            return emb_dict
    except Exception as e:
        st.error("❌ Failed to load or compute embeddings:")
        st.code(traceback.format_exc())
        return {}

passage_embeddings = load_or_compute_embeddings(corpus)

# -----------------------------
# Retrieval
# -----------------------------
def retrieve_top_docs(query, top_k=3):
    if not passage_embeddings:
        return []
    q_emb = encoder.encode(query, normalize_embeddings=True)
    scores = [(pid, float(np.dot(q_emb, p_emb))) for pid, p_emb in passage_embeddings.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:top_k]
    return [{"id": pid, "content": corpus[pid], "score": round(score, 3)} for pid, score in top]

# -----------------------------
# Generation
# -----------------------------
def generate_answer(query, retrieved_docs):
    if not retrieved_docs:
        return "⚠️ No retrieved passages available."
    context = "\n".join([f"- {d['content']}" for d in retrieved_docs])
    prompt = f"Answer this question using only the context below.\n\n{context}\n\nQuestion: {query}\nAnswer:"
    try:
        input_ids = gen_tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = gen_model.generate(input_ids, max_new_tokens=100, temperature=0.7)
        resp = gen_tokenizer.decode(out[0], skip_special_tokens=True)
        return resp.strip()
    except Exception as e:
        st.error("❌ Generation failed:")
        st.code(traceback.format_exc())
        return "Generation error."

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

        if top_docs:
            st.markdown("---")
            st.subheader("📚 Top-3 Retrieved Passages")
            for i, doc in enumerate(top_docs, 1):
                with st.expander(f"Passage {i} (score={doc['score']})"):
                    st.write(doc["content"])
                    
