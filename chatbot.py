import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq

# ✅ Set your GROQ API key (never expose this in production!)
os.environ["GROQ_API_KEY"] = "gsk_KN7SE54vg6666dmfM9V0AWGdyb3FYJfzEeM2SkARNTMfYDutBDe4h"

st.set_page_config(page_title="AI PDF Chatbot", layout="wide")
st.header("🤖 Chat with Your PDF")

# Sidebar upload
with st.sidebar:
    st.title("📄 Upload your document")
    file = st.file_uploader("Upload a file", type=["pdf"])

# Process file
if file is not None:
    # Read PDF
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content

    st.subheader("📜 Extracted Text Preview")
    st.write(text[:1000] + "..." if len(text) > 1000 else text)

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)

    # Embeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Vector Store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User Input
    user_query = st.text_input("💬 Ask a question about the document:")

    if user_query:
        # Similarity search
        matched_docs = vector_store.similarity_search(user_query, k=3)

        # ChatGroq LLM
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.1)

        # QA chain
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=matched_docs, question=user_query)

        # Output
        st.subheader("🤖 Chatbot Response")
        st.write(response)

else:
    st.info("Please upload a PDF file to begin.")
