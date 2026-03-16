import streamlit as st
import tempfile
import os
from utils import load_and_chunk_pdf, build_vector_store, build_rag_chain

st.set_page_config(
    page_title="AI Research Paper Explainer",
    layout="wide"
)

st.title("AI research Paper Explainer")
st.caption("Upload a research paper and chat with it using AI")


if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chathistory = []

if "paper_loaded" not in st.session_state:
    st.session_state.paper_loaded = False

with st.sidebar:
    st.header("Upload Paper")
    uploaded_file = st.file_uploader("Choose a pdf", type="pdf")

    if uploaded_file and not st.session_state.paper_loaded:
        with st.spinner("Reading and indexing paper..."):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            chunks = load_and_chunk_pdf(tmp_path)
            vector_store = build_vector_store(chunks)
            st.session_state.rag_chain = build_rag_chain(vector_store)
            st.session_state.paper_loaded = True
            st.session_state.chat_history = []

            os.unlink(tmp_path)
        
        st.success(f"Indexed{len(chunks)} chunks from paper")

    if st.session_state.paper_loaded:
        if st.button("Load a different paper"):
            st.session_state.rag_chain = None
            st.session_state.chat_history = []
            st.session_state.paper_loaded = False
            st.rerun()

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload A PDF")
    st.markdown("2. Paper gets chunked & embedded into vectors")
    st.markdown("3. Your question finds the most relevant chunks")
    st.markdown("Gemini answers using only those chunks")

if not st.session_state.paper_loaded:

    st.info("Upload a research paper in the sidebar to get started")

    st.markdown("### What you can ask:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- What is the main contribution?")
        st.markdown("- What problem does this solve?")
        st.markdown("- Explain the methodology")

    with col2:
        st.markdown("- What are the results?")
        st.markdown("- Who are the authors?")
        st.markdown("- Summarize section 3")

    
else:

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    if question := st.chat_input("Ask anything about the paper..."):

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if st.session_state.rag_chain is None:
                st.error("No paper loaded. Please upload a PDF first.")
            else:
                with st.spinner("Thinking..."):
                    answer = st.session_state.rag_chain.invoke(question)
                st.markdown(answer)
        st.session_state.chat_history.append({
            "role": "assitant",
            "content": answer
        })
            