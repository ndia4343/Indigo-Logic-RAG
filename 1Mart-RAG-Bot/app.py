import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ShopBot AI | Professional RAG Support",
    page_icon="🤖",
    layout="wide",
)

# --- PREMIUM DASHBOARD STYLING (Blue Corporate Theme) ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap" rel="stylesheet">
<style>
    /* Global Background */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Overhaul (Dark Blue Sidebar) */
    [data-testid="stSidebar"] {
        background-color: #0c1e3e;
        border-right: 1px solid #1e293b;
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* Professional Header */
    .header-container {
        background-color: #1e3a8a;
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: 1.8rem;
        margin: 0;
    }
    
    .header-tag {
        font-size: 0.8rem;
        color: #93c5fd;
    }

    /* Navigation / Sections */
    .nav-label {
        font-size: 0.7rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 1.5rem;
    }

    /* Chat Styling */
    .chat-bubble {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        max-width: 80%;
    }
    
    .bot-bubble {
        background: white;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2563eb;
        color: #1e293b;
    }
    
    .user-bubble {
        background: #2563eb;
        color: white;
        margin-left: auto;
    }

    /* Stats Card */
    .stats-card {
        background: #162447;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #1f3a93;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA & RAG LOGIC ---
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return embedder

@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        data = {
            'Product': ['Premium Wireless Headphones', 'Smart Fitness Watch', 'Organic Arabica Coffee', 'Mechanical Gaming Keyboard', 'Ergonomic Office Chair', 'UltraWide Monitor 34"', 'Stainless Steel Water Bottle'],
            'Category': ['Electronics', 'Wearables', 'Grocery', 'Accessories', 'Furniture', 'Electronics', 'Home'],
            'Price': [249.99, 129.50, 18.00, 89.00, 350.00, 499.99, 25.00],
            'Stock Status': ['In Stock', 'In Stock', 'Low Stock', 'Out of Stock', 'In Stock', 'In Stock', 'In Stock'],
            'Description': ['Noise canceling headphones.', 'Tracks health.', 'Organic beans.', 'RGB Keyboard.', 'Ergonomic.', '4K Display.', 'Safe and reusable.']
        }
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
    return pd.read_csv(path)

# Initialize
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "ecommerce_sales.csv")
df = load_data(CSV_PATH)
embedder = load_models()

# Build FAISS locally
texts = df.apply(lambda r: f"Item: {r['Product']}, Category: {r['Category']}, Price: ${r['Price']}, Stock: {r['Stock Status']}, Details: {r['Description']}", axis=1).tolist()
embeddings = embedder.encode(texts)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings).astype('float32'))

# --- SIDEBAR (DASHBOARD STYLE) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=80)
    st.markdown("<h2 style='margin-bottom:0;'>ShopBot AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#93c5fd;font-size:0.8rem;'>Customer Support · RAG Powered</p>", unsafe_allow_html=True)
    
    st.markdown("<p class='nav-label'>NAVIGATION</p>", unsafe_allow_html=True)
    st.markdown("💬 Chat\n📂 Upload files\n📜 Chat history\n📊 Analytics")
    
    st.markdown("<p class='nav-label'>KNOWLEDGE BASE</p>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='stats-card'>
        <div style='font-size:0.75rem; color:#93c5fd;'>ecommerce_sales.csv</div>
        <div style='font-size:0.9rem; font-weight:700;'>{len(df)} products indexed</div>
    </div>
    """, unsafe_allow_html=True)
    
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.markdown("<p class='nav-label'>SYSTEM INFO</p>", unsafe_allow_html=True)
    st.caption("Model: Gemini 1.5 Flash\nEmbedder: MiniLM-L6-v2\nVector DB: FAISS")

# --- MAIN CONTENT ---
st.markdown(f"""
    <div class="header-container">
        <div>
            <div class="header-title">ShopBot AI</div>
            <div class="header-tag">Customer support · RAG powered</div>
        </div>
        <div style="font-size: 0.8rem;">● Online</div>
    </div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your e-commerce support assistant. I have access to your full product database. How can I help you today?"}]

# Display chat history
for msg in st.session_state.messages:
    cls = "bot-bubble" if msg["role"] == "assistant" else "user-bubble"
    st.markdown(f'<div class="chat-bubble {cls}">{msg["content"]}</div>', unsafe_allow_html=True)

# Input
if prompt := st.chat_input("Ask a question about orders or products..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="chat-bubble user-bubble">{prompt}</div>', unsafe_allow_html=True)
    
    with st.spinner("ShopBot AI typing..."):
        # 1. FAISS Search
        query_emb = embedder.encode([prompt]).astype('float32')
        _, I = index.search(query_emb, k=3)
        context = "\n".join([texts[i] for i in I[0]])
        
        # 2. Add Analytical Helpers
        total_rev = df['Price'].sum()
        avg_price = df['Price'].mean()
        
        # 3. Request LLM
        if api_key:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                sys_prompt = f"""
                You are ShopBot AI, a professional e-commerce support bot.
                Use the CSV Context and System Stats below to answer queries.
                
                System Stats:
                - Total Inventory Value: ${total_rev:,.2f}
                - Average Product Price: ${avg_price:,.2f}
                - Total Products: {len(df)}
                
                CSV Context:
                {context}
                """
                response = model.generate_content(f"{sys_prompt}\n\nUser: {prompt}")
                ans = response.text
            except Exception as e:
                ans = f"Error: {e}"
        else:
            ans = f"Context found:\n{context}\n\n(Tip: Enter API key in sidebar for analytical answers)"
            
    st.markdown(f'<div class="chat-bubble bot-bubble">{ans}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": ans})
