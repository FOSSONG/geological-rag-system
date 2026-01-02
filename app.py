import streamlit as st
from utils import extract_text_from_pdf, add_document, retrieve, synthesize_answer

st.set_page_config("Offline RAG System", layout="wide")
st.title("📚 Offline Local RAG System")

uploaded = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

if uploaded:
    for f in uploaded:
        text = extract_text_from_pdf(f)
        add_document(text, f.name)
    st.success("Documents indexed successfully.")

query = st.text_input("Ask a question about your documents")

if st.button("Ask") and query.strip():
    hits = retrieve(query)
    answer, sources = synthesize_answer(query, hits)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Sources"):
        for h in sources:
            st.markdown(f"**{h['source']}**")
            st.write(h["text"])
