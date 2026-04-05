"""
1Mart ShopAI — Streamlit RAG Chatbot
Fixed: total revenue, top customers, currency symbols, basic math
"""

import os, json, re, tempfile
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="ShopAI – 1Mart", page_icon="🛒", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 300
CHUNK_OVERLAP = 50
MODEL_NAME    = "google/flan-t5-base"
EMBEDDER_NAME = "all-MiniLM-L6-v2"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_currency(value, symbol="$"):
    try:
        return f"{symbol}{float(value):,.2f}"
    except (ValueError, TypeError):
        return f"{symbol}{value}"


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    step  = max(size - overlap, 1)
    return [" ".join(words[i:i+size])
            for i in range(0, len(words), step) if words[i:i+size]]


def create_text_blob(row):
    return (
        f"Customer {row.get('customer_name','N/A')} from {row.get('country','N/A')} purchased "
        f"{row.get('quantity','N/A')} unit(s) of {row.get('product','N/A')} "
        f"for {fmt_currency(row.get('price', 0))} each. "
        f"Total order value was {fmt_currency(row.get('order_value', 0))}. "
        f"Transaction ID: {row.get('order_id','N/A')} on {row.get('date','N/A')}."
    )


def parse_tabular(filepath):
    ext = filepath.rsplit(".", 1)[-1].lower()
    try:
        if ext == "csv":             df = pd.read_csv(filepath)
        elif ext in ("xlsx","xlsm"): df = pd.read_excel(filepath, engine="openpyxl")
        elif ext == "xls":           df = pd.read_excel(filepath, engine="xlrd")
        else: return []
    except Exception as e:
        st.warning(f"Could not read {filepath}: {e}"); return []
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.fillna("N/A")
    return [" | ".join(f"{c.replace('_',' ').title()}: {v}" for c, v in row.items())
            for _, row in df.iterrows()]


# ─────────────────────────────────────────────────────────────────────────────
# Data-aware query handler  ← KEY FIX for "total revenue" / "top customers"
# ─────────────────────────────────────────────────────────────────────────────

def try_data_query(query: str, df: pd.DataFrame):
    """Answer analytics questions directly from the dataframe — no LLM needed."""
    if df is None or df.empty:
        return None

    q = query.lower().strip()

    # Normalise column names
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ── Total revenue ─────────────────────────────────────────────────────────
    if any(k in q for k in ["total revenue", "overall revenue", "sum of revenue",
                              "total sales", "overall sales", "total order value"]):
        if "order_value" in df.columns:
            total = pd.to_numeric(df["order_value"], errors="coerce").sum()
            return f"The total revenue is ${total:,.2f}"

    # ── Average order value ───────────────────────────────────────────────────
    if any(k in q for k in ["average order", "avg order", "mean order",
                              "average revenue", "avg revenue"]):
        if "order_value" in df.columns:
            avg = pd.to_numeric(df["order_value"], errors="coerce").mean()
            return f"The average order value is ${avg:,.2f}"

    # ── Total orders count ────────────────────────────────────────────────────
    if any(k in q for k in ["how many orders", "total orders", "number of orders",
                              "count of orders", "order count"]):
        return f"There are {len(df):,} total orders in the dataset."

    # ── Top country by orders ─────────────────────────────────────────────────
    if any(k in q for k in ["which country", "top country", "most orders",
                              "highest orders", "country has most"]):
        if "country" in df.columns:
            top = df["country"].value_counts().head(3)
            lines = [f"{i+1}. {c} — {n:,} orders" for i,(c,n) in enumerate(top.items())]
            return "Top countries by number of orders:\n" + "\n".join(lines)

    # ── Top country by revenue ────────────────────────────────────────────────
    if any(k in q for k in ["country by revenue", "highest revenue country",
                              "most revenue", "top revenue country"]):
        if "country" in df.columns and "order_value" in df.columns:
            df["order_value"] = pd.to_numeric(df["order_value"], errors="coerce")
            top = df.groupby("country")["order_value"].sum().sort_values(ascending=False).head(3)
            lines = [f"{i+1}. {c} — ${v:,.2f}" for i,(c,v) in enumerate(top.items())]
            return "Top countries by revenue:\n" + "\n".join(lines)

    # ── Orders / revenue from a specific country ──────────────────────────────
    country_match = re.search(
        r"(orders?|sales?|revenue).{0,15}(from|in|for)\s+([a-zA-Z\s]+?)(\?|$)", q
    )
    if country_match and "country" in df.columns:
        country_name = country_match.group(3).strip().title()
        filtered = df[df["country"].str.title() == country_name]
        if not filtered.empty:
            rev = pd.to_numeric(filtered["order_value"], errors="coerce").sum()
            return (f"{country_name} has {len(filtered):,} orders "
                    f"with total revenue of ${rev:,.2f}.")
        else:
            return f"No orders found for '{country_name}' in the dataset."

    # ── Most popular product ──────────────────────────────────────────────────
    if any(k in q for k in ["popular product", "top product", "best selling",
                              "most ordered", "most sold"]):
        if "product" in df.columns:
            top = df["product"].value_counts().head(3)
            lines = [f"{i+1}. {p} — {n:,} orders" for i,(p,n) in enumerate(top.items())]
            return "Most popular products:\n" + "\n".join(lines)

    # ── Revenue for a specific product ────────────────────────────────────────
    product_rev = re.search(
        r"(revenue|sales|value).{0,15}(of|for)\s+([a-zA-Z0-9\s]+?)(\?|$)", q
    )
    if product_rev and "product" in df.columns:
        prod_name = product_rev.group(3).strip().title()
        filtered  = df[df["product"].str.title().str.contains(prod_name, na=False)]
        if not filtered.empty:
            rev = pd.to_numeric(filtered["order_value"], errors="coerce").sum()
            return (f"{prod_name} has {len(filtered):,} orders "
                    f"with total revenue of ${rev:,.2f}.")

    # ── Top customers ─────────────────────────────────────────────────────────
    if any(k in q for k in ["top customer", "best customer", "highest spending",
                              "most valuable customer", "who are top"]):
        if "customer_name" in df.columns and "order_value" in df.columns:
            df["order_value"] = pd.to_numeric(df["order_value"], errors="coerce")
            top = df.groupby("customer_name")["order_value"].sum().sort_values(ascending=False).head(3)
            lines = [f"{i+1}. {c} — ${v:,.2f}" for i,(c,v) in enumerate(top.items())]
            return "Top customers by total spend:\n" + "\n".join(lines)

    return None  # Not a data question — fall through to LLM


# ─────────────────────────────────────────────────────────────────────────────
# Simple arithmetic handler
# ─────────────────────────────────────────────────────────────────────────────

def try_math(query: str):
    q    = query.lower().strip()
    nums = [float(n.replace(",", ""))
            for n in re.findall(r"\d[\d,]*(?:\.\d+)?", q)]
    if not nums:
        return None
    if any(k in q for k in ["sum", "add", "plus", "+"]) and len(nums) > 1:
        return f"The total is ${sum(nums):,.2f}"
    if any(k in q for k in ["multiply", "times", "×"]) and len(nums) > 1:
        r = nums[0]
        for n in nums[1:]: r *= n
        return f"The result is ${r:,.2f}"
    if any(k in q for k in ["subtract", "minus", "difference"]) and len(nums) > 1:
        return f"The result is ${nums[0] - sum(nums[1:]):,.2f}"
    if any(k in q for k in ["divide", "divided by"]) and len(nums) >= 2 and nums[1] != 0:
        return f"The result is {nums[0] / nums[1]:,.4f}"
    if any(k in q for k in ["average", "avg", "mean"]) and len(nums) > 1:
        return f"The average is ${sum(nums)/len(nums):,.2f}"
    if ("%" in q or "percent" in q) and len(nums) >= 2:
        return f"{(nums[0]/nums[1])*100:.2f}%"
    return None


def format_currency_in_answer(text: str) -> str:
    return re.sub(
        r"(?i)(revenue|price|cost|value|total|amount|order)[\s:]*([$£€]?)(\d[\d,]*(?:\.\d{1,2})?)",
        lambda m: f"{m.group(1)} ${m.group(3)}" if not m.group(2) else m.group(0),
        text
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    embedder  = SentenceTransformer(EMBEDDER_NAME)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    llm       = T5ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    return embedder, tokenizer, llm


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge base builder (cached per file hash)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Building knowledge base...")
def build_kb(file_bytes: bytes, filename: str, _embedder):
    # Write to temp file
    suffix = "." + filename.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Parse
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        df = pd.read_csv(tmp_path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df = df.fillna("N/A")
        # Use natural language blobs for embeddings
        chunks = [create_text_blob(row) for _, row in df.iterrows()]
    elif ext in ("xlsx", "xlsm", "xls"):
        chunks = parse_tabular(tmp_path)
        df = pd.read_excel(tmp_path)
    else:
        with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
            chunks = chunk_text(f.read())
        df = None

    os.unlink(tmp_path)

    if not chunks:
        return None, None, None

    # Embed
    embeddings = _embedder.encode(
        chunks, batch_size=256, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    # FAISS index
    dim = embeddings.shape[1]
    n   = len(chunks)
    if n < 10_000:
        index = faiss.IndexFlatIP(dim)
    else:
        nlist = min(256, n // 100)
        q     = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(q, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.nprobe = 16
    index.add(embeddings)

    return chunks, index, df


# ─────────────────────────────────────────────────────────────────────────────
# RAG answer
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_context(query, chunks, index, embedder, top_k=5):
    q_vec = embedder.encode([query], convert_to_numpy=True,
                             normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_vec, top_k)
    return [chunks[i] for s, i in zip(scores[0], indices[0])
            if i != -1 and float(s) > 0.25]


def build_prompt(query, ctx):
    body = "\n".join(f"- {c}" for c in ctx[:5])
    return (
        "You are ShopAI, a helpful 1Mart e-commerce support assistant.\n"
        "Use ONLY the context below to answer accurately.\n"
        "Always include the $ currency symbol for any price, revenue, or order value.\n"
        "If the answer is not in the context, say: "
        "'I don't have that information. Please contact 1Mart support.'\n\n"
        f"Context:\n{body}\n\nCustomer question: {query}\n\nAnswer:"
    )


def generate_answer(prompt, tokenizer, llm, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt",
                       max_length=1024, truncation=True).to(DEVICE)
    with torch.no_grad():
        out = llm.generate(
            **inputs, max_new_tokens=max_new_tokens,
            num_beams=4, early_stopping=True, no_repeat_ngram_size=3,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def get_answer(query, chunks, index, df, embedder, tokenizer, llm):
    """Full pipeline: data → math → RAG"""
    if not query.strip():
        return "Please enter a question."

    # 1. Direct data query
    ans = try_data_query(query, df)
    if ans:
        return ans

    # 2. Simple math
    ans = try_math(query)
    if ans:
        return ans

    # 3. RAG
    ctx = retrieve_context(query, chunks, index, embedder)
    if not ctx:
        return "I couldn't find relevant information. Please contact 1Mart support."
    answer = generate_answer(build_prompt(query, ctx), tokenizer, llm)
    return format_currency_in_answer(answer)


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## 🛒 SHOPAI")
        st.caption("Your intelligent shopping assistant")
        st.divider()

        st.markdown("### SYSTEM STATUS")
        col1, col2 = st.columns(2)
        col1.markdown("**Bot**");      col2.markdown("🟢 Ready")
        col1.markdown("**Store**");    col2.markdown("🟠 1Mart")
        col1.markdown("**LLM**");      col2.markdown("🟠 Flan-T5")
        col1.markdown("**Embedder**"); col2.markdown("🟠 MiniLM-L6")
        col1.markdown("**Vector DB**");col2.markdown("🟠 FAISS")
        st.divider()

        st.markdown("### UPLOAD DATA")
        uploaded_files = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True
        )

        build_btn = st.button("⚡ Build Knowledge Base", use_container_width=True)
        st.divider()

        st.markdown("### ℹ️ ABOUT")
        st.caption("Upload ecommerce_sales.csv or any Excel/text file to power the bot with your own data.")

    # Load models
    embedder, tokenizer, llm = load_models()

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Welcome to 1Mart! I'm ShopAI, your AI shopping assistant. Ask me anything about products, orders, or pricing."}
        ]
    if "kb_built" not in st.session_state:
        st.session_state.kb_built  = False
        st.session_state.chunks    = None
        st.session_state.index     = None
        st.session_state.df        = None

    # Build KB
    if build_btn and uploaded_files:
        all_chunks, combined_df = [], []
        final_index = None

        for uf in uploaded_files:
            file_bytes = uf.read()
            chunks, idx, df = build_kb(file_bytes, uf.name, embedder)
            if chunks:
                all_chunks.extend(chunks)
                if df is not None:
                    combined_df.append(df)

        if all_chunks:
            # Rebuild single FAISS index over all chunks
            all_emb = embedder.encode(
                all_chunks, batch_size=256, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True
            ).astype("float32")
            dim, n = all_emb.shape[1], len(all_chunks)
            if n < 10_000:
                final_index = faiss.IndexFlatIP(dim)
            else:
                nlist = min(256, n // 100)
                q     = faiss.IndexFlatIP(dim)
                final_index = faiss.IndexIVFFlat(q, dim, nlist, faiss.METRIC_INNER_PRODUCT)
                final_index.train(all_emb)
                final_index.nprobe = 16
            final_index.add(all_emb)

            st.session_state.chunks   = all_chunks
            st.session_state.index    = final_index
            st.session_state.df       = pd.concat(combined_df, ignore_index=True) if combined_df else None
            st.session_state.kb_built = True
            st.sidebar.success(f"✅ Knowledge base ready — {len(all_chunks):,} chunks")

    # Suggested questions
    with st.sidebar:
        st.markdown("### 💡 Try asking")
        suggestions = [
            "What is total revenue?",
            "Which country has most orders?",
            "Who are top customers?",
            "Most popular product?",
            "Orders from India",
            "Products available?",
        ]
        for s in suggestions:
            if st.button(s, key=f"sug_{s}", use_container_width=True):
                st.session_state._pending = s

    # Main chat area
    st.markdown("## 🛒 ShopAI — 1Mart Assistant")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Handle suggestion click
    pending = st.session_state.pop("_pending", None)
    user_input = st.chat_input("Ask ShopAI anything...") or pending

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        if not st.session_state.kb_built:
            answer = "Please upload your data file and click **Build Knowledge Base** first."
        else:
            with st.spinner("ShopAI is thinking..."):
                answer = get_answer(
                    user_input,
                    st.session_state.chunks,
                    st.session_state.index,
                    st.session_state.df,
                    embedder, tokenizer, llm
                )

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)


if __name__ == "__main__":
    main()
