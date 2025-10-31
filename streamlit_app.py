import streamlit as st, pkg_resources, traceback, os
import torch, json, numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Basic setup & diagnostics
st.set_page_config(page_title="RAG Information Retrieval App", layout="wide")

st.write("Ask a question and return the top-3 most relevant documents and an LLM-generated response based on these.")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Using device: `{device}`")

# Load embedding model

@st.cache_resource
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLM

@st.cache_resource
llm_name = "sshleifer/tiny-gpt2"
gen_tokenizer = AutoTokenizer.from_pretrained(llm_name)
gen_model = AutoModelForCausalLM.from_pretrained(
    llm_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
).to(device)

# Load corpus
@st.cache_resource
corpus = {}
with open("collection/sampled_collection.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        corpus[d["id"]] = d["contents"]

corpus = load_corpus()

# Load or compute embeddings
@st.cache_resource
def load_or_compute_embeddings(corpus):
    try:
        emb_path = "collection/embeddings.npy"
        if os.path.exists(emb_path):
            st.write("Found precomputed embeddings. Loading...")
            data = np.load(emb_path, allow_pickle=True).item()
            st.write(f"Loaded {len(data)} embeddings.")
            return data
        else:
            st.write("No precomputed embeddings found. Computing new embeddings...")
            ids, texts = list(corpus.keys()), list(corpus.values())
            embs = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
