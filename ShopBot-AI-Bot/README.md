# 1Mart E-Commerce RAG Bot

1Mart E-Commerce RAG Bot is a professional, offline-capable Retrieval-Augmented Generation chatbot built for e-commerce customer support and analytics. 

## Overview
This architecture ensures secure data processing by avoiding third-party APIs. It answers user questions based strictly on local Excel, CSV, or text files, entirely preventing AI hallucinations.

## Key Technologies
- **Local LLM**: Google FLAN-T5 (Base/Small) for text generation.
- **Embedded Search**: MiniLM-L6 for generating semantic vector embeddings.
- **Vector Database**: FAISS (Facebook AI Similarity Search) for rapid and precise context retrieval.
- **Frontend App**: Streamlit, customized via CSS for a polished, corporate-ready interface.

## Core Features
1. **Offline Inference**: Data uploaded and queries raised are entirely processed within the environment.
2. **Context-Aware Reasoning**: Directly searches the product catalog and client orders databases before generating a response.
3. **Data Agnostic**: Upload `.csv`, `.xlsx`, or `.txt` datasets directly through the interface to continuously append to the knowledge base.
4. **Professional UI**: Tailored interface to match modern e-commerce dashboard aesthetics.

## Installation
Run the following from the root directory:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
This project is configured to run efficiently on Streamlit Cloud using CPU-bound models, ensuring it stays well within memory guidelines while delivering high-fidelity AI performance.
