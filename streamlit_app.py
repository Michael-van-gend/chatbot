import streamlit as st
import random

# ---------------------------
# Dummy data (replace with real retrieval & generation later)
# ---------------------------

DUMMY_DOCUMENTS = [
    {
        "id": "doc1",
        "content": "Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital in London.",
    },
    {
        "id": "doc2",
        "content": "Penicillin is derived from the Penicillium mold and was the first true antibiotic.",
    },
    {
        "id": "doc3",
        "content": "The discovery of penicillin revolutionized modern medicine and earned Fleming a Nobel Prize.",
    },
    {
        "id": "doc4",
        "content": "Marie Curie discovered polonium and radium and won two Nobel Prizes for her work in radioactivity.",
    },
    {
        "id": "doc5",
        "content": "The theory of relativity was proposed by Albert Einstein in the early 20th century.",
    },
]

# ---------------------------
# Dummy retrieval and generation functions
# ---------------------------

def retrieve_top_docs(query, k=3):
    """Simulate retrieval by picking top-k documents with random scores."""
    random.shuffle(DUMMY_DOCUMENTS)
    top_docs = DUMMY_DOCUMENTS[:k]
    return [
        {
            "id": doc["id"],
            "content": doc["content"],
            "score": round(random.uniform(0.7, 1.0), 3),
        }
        for doc in top_docs
    ]

def generate_answer(query, docs):
    """Simulate generation by combining key sentences from retrieved docs."""
    joined = " ".join([d["content"] for d in docs])
    answer = f"Based on the documents, the answer to '{query}' is: {joined.split('.')[0]}."
    return answer

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Mini RAG Demo", layout="wide")

st.title("🧠 Retrieval-Augmented Generation (RAG) Demo")
st.write("Enter a question below. The app retrieves the top-3 relevant passages and generates an answer.")

# Query input
query = st.text_input("💬 Your question:", placeholder="e.g. Who discovered penicillin?")

# Button to run the pipeline
if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            # Backend logic
            top_docs = retrieve_top_docs(query, k=3)
            answer = generate_answer(query, top_docs)

        # Display results
        st.subheader("🧩 Generated Answer")
        st.success(answer)

        st.markdown("---")
        st.subheader("📚 Top-3 Retrieved Passages")
        for i, doc in enumerate(top_docs, 1):
            with st.expander(f"Passage {i} (score={doc['score']})"):
                st.write(doc["content"])
