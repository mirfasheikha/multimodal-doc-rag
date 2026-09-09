# Multimodal Document Intelligence & RAG Engine

A production-grade Retrieval-Augmented Generation (RAG) system capable of parsing documents, embedding semantic representations into a vector space, and performing contextually grounded question-answering with Google Gemini.

---

## System Architecture

```text
[User Document (PDF)] ──> [pypdf Extraction] 
                               │
                               ▼
            [RecursiveCharacterTextSplitter] (Chunking)
                               │
                               ▼
        [sentence-transformers/all-MiniLM-L6-v2] (Embeddings)
                               │
                               ▼
                     [ChromaDB Vector Store]
                               │
[User Query] ──────────> [Cosine Retrieval (Top-k)]
                               │
                               ▼
                     [Augmented Prompt Context]
                               │
                               ▼
                    [Google Gemini API] ──> [Grounded Output]