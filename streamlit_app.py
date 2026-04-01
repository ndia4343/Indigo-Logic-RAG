import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Indigo Logic | Knowledge Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR PROFESSIONAL DARK THEME ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0f1116;
        color: #e2e8f0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1a1d23;
        border-right: 1px solid #2d3748;
    }
    
    /* Input Fields */
    .stTextInput>div>div>input {
        background-color: #2d3748;
        color: white;
        border: 1px solid #4a5568;
    }
    
    /* Custom Sidebar Headers */
    .sidebar-header {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    
    /* Blue Accents */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    /* Chat Bubbles */
    .chat-bubble {
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        line-height: 1.6;
        font-family: 'Inter', sans-serif;
    }
    .user-bubble {
        background-color: #1e293b;
        border: 1px solid #334155;
    }
    .assistant-bubble {
        background-color: #111827;
        border-left: 4px solid #2563eb;
        border: 1px solid #1f2937;
        border-left: 4px solid #2563eb;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = ""
if "processed" not in st.session_state:
    st.session_state.processed = False

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">AI CONFIGURATION</div>', unsafe_allow_html=True)
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your key from Google AI Studio")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    st.markdown('<div class="sidebar-header">DATA INGESTION</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["csv", "xlsx", "txt", "md"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if st.button("🚀 Process Data"):
        if uploaded_files:
            combined_context = ""
            with st.spinner("Indexing knowledge base..."):
                for file in uploaded_files:
                    filename = file.name
                    file_ext = filename.split('.')[-1].lower()
                    
                    try:
                        if file_ext == "csv":
                            df = pd.read_csv(file)
                            combined_context += f"\n--- DATASET: {filename} ---\n{df.to_csv(index=False)[:80000]}\n"
                        elif file_ext == "xlsx":
                            df = pd.read_excel(file)
                            combined_context += f"\n--- SPREADSHEET: {filename} ---\n{df.to_csv(index=False)[:80000]}\n"
                        else:
                            content = file.read().decode("utf-8")
                            combined_context += f"\n--- DOCUMENT: {filename} ---\n{content}\n"
                    except Exception as e:
                        st.error(f"Error parsing {filename}: {e}")
                
                st.session_state.knowledge_base = combined_context
                st.session_state.processed = True
                st.success("Database Updated Successfully!")
        else:
            st.warning("Please upload files first.")

# --- MAIN INTERFACE ---
st.title("Knowledge Explorer")
st.caption("Indigo Logic Precision ML Engine | Context-Aware RAG Chatbot")

# Display Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
    elif not st.session_state.processed:
        st.warning("Please upload and 'Process' data before chatting.")
    else:
        with st.chat_message("assistant"):
            chat_placeholder = st.empty()
            with st.spinner("Analyzing context..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(
                        model_name="gemini-2.5-flash", # Fixed to match your specific API account
                        system_instruction="You are a strict data analysis chatbot. Use the provided Document Context to answer questions. If the answer is not in the context, say 'I cannot find the answer in the provided documents.' Do not halluncinate."
                    )
                    
                    # Create context-aware history for Gemini
                    history = []
                    for m in st.session_state.messages[:-1]:
                        history.append({"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]})
                    
                    # Core prompt with document data
                    full_query = f"Document Context:\n{st.session_state.knowledge_base}\n\nUser Question: {prompt}"
                    
                    chat = model.start_chat(history=history)
                    response = chat.send_message(full_query, generation_config={"temperature": temperature})
                    
                    chat_placeholder.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"API Error: {e}")

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; color: #4a5568; font-size: 0.7rem;">Indigo Logic v1.0.0 | Built with Gemini Pro & Streamlit</div>', unsafe_allow_html=True)
