import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from google import genai

st.set_page_config(page_title="Multimodal Document RAG Engine", layout="wide")
st.title("📄 Conversational Document RAG Engine")

# Sidebar for API key and document management
api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("Upload a Document (PDF)", type=["pdf"])

# Process PDF only once per upload
if uploaded_file and api_key and st.session_state.vectorstore is None:
    with st.spinner("Extracting text and building vector index..."):
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vectorstore = Chroma.from_texts(chunks, embeddings)
        st.success("Document indexed! You can now chat continuously.")

# Display existing chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User chat input
if prompt := st.chat_input("Ask a question about the document..."):
    if not api_key:
        st.warning("Please enter your Gemini API key in the sidebar.")
    elif st.session_state.vectorstore is None:
        st.warning("Please upload and process a PDF first.")
    else:
        # Display user message immediately
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant context
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(prompt)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Construct conversation-aware prompt
        history_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-5:]])
        full_prompt = (
            f"Conversation History:\n{history_context}\n\n"
            f"Document Context:\n{context}\n\n"
            f"User Question: {prompt}\n\n"
            "Answer the question accurately based on the context and previous messages. If the answer cannot be found in the context, clearly state that."
        )

        client = genai.Client(api_key=api_key)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=full_prompt,
                )
                st.markdown(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})