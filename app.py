import streamlit as st
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from google import genai

st.set_page_config(page_title="Multimodal Document RAG Engine", layout="wide")
st.title("📄 Multimodal Document Intelligence & RAG Engine")

api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")

uploaded_file = st.file_uploader("Upload a Document (PDF)", type=["pdf"])

if uploaded_file and api_key:
    client = genai.Client(api_key=api_key)
    
    with st.spinner("Extracting text and building vector index..."):
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_text(text)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_texts(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
    st.success("Document indexed successfully! Ask your questions below.")

    query = st.text_input("Ask a question about the document:")
    if query:
        with st.spinner("Retrieving relevant context and generating answer..."):
            relevant_docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nProvide a precise, comprehensive answer using only the provided context."
            
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            )
            
            st.subheader("Answer:")
            st.write(response.text)
            
            with st.expander("Retrieved Context Chunks"):
                for idx, doc in enumerate(relevant_docs):
                    st.markdown(f"**Chunk {idx+1}:**\n{doc.page_content}")
elif not api_key:
    st.info("Please enter a Google Gemini API Key in the sidebar to get started.")
