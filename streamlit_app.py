import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Indigo Logic | Precision RAG Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- GOOGLE FONTS & MATERIAL SYMBOLS ---
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Manrope:wght@700;800&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

# --- PROFESSIONAL UI INJECTION (CUSTOM CSS) ---
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background: #0d0e10;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Overhaul */
    [data-testid="stSidebar"] {
        background-color: #111216;
        border-right: 1px solid #1f2127;
        padding-top: 1rem;
    }
    
    /* Branding */
    .brand-container {
        padding: 1.5rem;
        border-bottom: 1px solid #1f2127;
        margin-bottom: 2rem;
    }
    .brand-title {
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: 1.1rem;
        letter-spacing: 0.05em;
        color: #fff;
    }
    .brand-tag {
        font-size: 0.65rem;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Message Bubbles - Precise Replication */
    .chat-container {
        max-width: 850px;
        margin: auto;
    }
    
    .msg-bot {
        background: #16171d;
        border: 1px solid #23252e;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .msg-user {
        background: rgba(37, 99, 235, 0.1);
        border: 1px solid rgba(37, 99, 235, 0.2);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
        text-align: right;
    }

    /* Sidebar Items */
    .sidebar-label {
        font-size: 0.7rem;
        font-weight: 700;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
    }

    /* Glassmorphism Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1e40af, #2563eb);
        border: none;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = ""
if "processed" not in st.session_state:
    st.session_state.processed = False

# --- SIDEBAR CONTENT ---
with st.sidebar:
    st.markdown("""
        <div class="brand-container">
            <div class="brand-title">INDIGO LOGIC</div>
            <div class="brand-tag">PRECISION ML ENGINE</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="sidebar-label">API CONFIGURATION</p>', unsafe_allow_html=True)
    api_key = st.text_input("Gemini API Key", type="password", label_visibility="collapsed", help="Get your key from AI Studio")
    
    st.markdown('<p class="sidebar-label">PARAMETERS</p>', unsafe_allow_html=True)
    temp = st.slider("Core Temperature", 0.0, 1.0, 0.7)

    st.markdown('<p class="sidebar-label">DATA INGESTION</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Drop Zone", type=["csv", "xlsx", "txt", "md"], accept_multiple_files=True, label_visibility="collapsed")
    
    if st.button("🔄 PROCESS DATA"):
        if uploaded_files:
            combined_context = ""
            with st.spinner("Indexing..."):
                for file in uploaded_files:
                    ext = file.name.split('.')[-1].lower()
                    if ext == "csv" or ext == "xlsx":
                        df = pd.read_csv(file) if ext == "csv" else pd.read_excel(file)
                        combined_context += f"\n--- DATA: {file.name} ---\n{df.to_csv(index=False)[:80000]}\n"
                    else:
                        combined_context += f"\n--- DOC: {file.name} ---\n{file.read().decode('utf-8')}\n"
                st.session_state.knowledge_base = combined_context
                st.session_state.processed = True
                st.sidebar.success("Database Updated")
        else:
            st.sidebar.warning("Upload files first")

# --- MAIN CHAT AREA ---
st.markdown('<h2 style="font-family:Manrope; font-weight:800; letter-spacing:-0.02em;">Knowledge Explorer</h2>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="msg-bot">
                {msg["content"]}
                <div style="margin-top:15px; padding-top:10px; border-top: 1px solid #23252e; font-size: 0.7rem; color: #4a5568;">
                    <span class="material-symbols-outlined" style="font-size:12px; vertical-align:middle;">dataset</span> SOURCE: <span style="color:#2563eb">Selected Context</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Processing actual LLM call if the last message is from user
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    if not api_key:
        st.error("API Key Required")
    elif not st.session_state.processed:
        st.warning("Please process your data first")
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction="You are an Expert Portfolio Analyst. \n\n1. KNOWLEDGE: CarPrices.csv = UCI Dataset (USD). ecommerce_data.csv = Retail data (USD). \n\n2. RULES: \n- Professional reporting format.\n- Use Dollars ($) automatically for prices. \n- Always provide a SUMMARY TABLE if possible."
            )
            
            # History stays as just text for efficiency
            history = [{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]} for m in st.session_state.messages[:-1]]
            
            # Core prompt with document data (ONLY for the latest message to save tokens)
            currentQuery = f"Reference the following Document Context to answer: \n{st.session_state.knowledge_base}\n\nUser Question: {st.session_state.messages[-1]['content']}"
            
            chat = model.start_chat(history=history)
            response = chat.send_message(currentQuery, generation_config={"temperature": temp})
            
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
