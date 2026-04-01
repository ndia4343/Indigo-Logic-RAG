# 🤖 Indigo Logic | Precision RAG Engine

A professional-grade, dark-themed **Retrieval-Augmented Generation (RAG)** chatbot built to analyze local Excel, CSV, and Text files using the **Gemini 1.5 Flash** model. 

Designed for data analysts who need quick insights from local documents without complex infrastructure.

## 🚀 Live Demo (Streamlit)
**[Link to your Streamlit App will appear here after deployment]**

## ✨ Key Features
- **Data Ingestion:** Supports `.xlsx`, `.csv`, `.txt`, and `.md` files.
- **Contextual Memory:** Remembers the last 10 messages for deep analysis.
- **Privacy First:** Processes documents locally before sending grounded queries to the Gemini API.
- **Premium UI:** Custom dark-themed dashboard built with Streamlit and CSS.
- **Parameter Control:** Real-time temperature slider to adjust AI creativity.

## 🛠️ Tech Stack
- **Frontend/App framework:** Streamlit (Python)
- **AI Backend:** Google Gemini 1.5 Flash / 2.5 Flash
- **Data Parsing:** Pandas & OpenPyXL

## 📦 How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/ndia4343/Indigo-Logic-RAG.git
   cd Indigo-Logic-RAG
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

## 🔐 API Configuration
You will need a **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/) to power the chat.

---
Built by [Nadia](https://github.com/ndia4343) 🚀
