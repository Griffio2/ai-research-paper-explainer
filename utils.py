import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

def load_and_chunk_pdf(file_path):

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(pages)

    return chunks

def build_vector_store(chunks):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vector_store = FAISS.from_documents(chunks,embeddings)

    return vector_store

def build_rag_chain(vector_store):

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    prompt = PromptTemplate.from_template("""
    You are an expert research assistant helping users understand academic papers.
    Answer the question based ONLY on the following context from the paper.
    If the answer isn't in the context, say "I couldn't find that in the paper."
    Be clear and explain technical terms in simple language.

    Context:
    {context}

    Question: {question}

    Answer:
    """)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

if __name__ == "__main__":
    chunks = load_and_chunk_pdf("test.pdf")
    print(f"Total chunks: {len(chunks)}")

    vector_store = build_vector_store(chunks)
    print("Vector store ready")

    rag_chain = build_rag_chain(vector_store)
    print("RAG chain ready\n")

    questions = [
        "What is the main contribution of this paper?",
        "What problem does this method solve?",
        "Who wrote this paper"
    ]

    for q in questions:
        print(f"Q: {q}")
        answer = rag_chain.invoke(q)
        print(f"A: {answer}")
        print("-"*60)