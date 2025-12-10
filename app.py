import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import csv
from datetime import datetime

# ---------------------
#        SETUP
# ---------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ Missing OPENAI_API_KEY in .env file.")
    st.stop()

client = OpenAI(api_key=api_key)
st.set_page_config(page_title="🤖 GGS Chatbot", layout="wide", initial_sidebar_state="collapsed")

# ---------------------
#         CSS
# ---------------------
st.markdown("""
<style>
/* --- PAGE BACKGROUND --- */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 20% 20%, #001a33, #000814 85%) !important;
    color: #E6F1FF;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    height: 100vh;
    overflow: hidden;
}

/* --- HEADER --- */
h1 {
    color: #FFFFFF;
    text-shadow: 0 0 25px rgba(0,180,255,0.8), 0 0 40px rgba(0,180,255,0.5);
    text-align: center;
    font-weight: 800;
    margin-top: 15px;
    margin-bottom: 5px;
}

/* --- USER MESSAGE --- */
.msg-user {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    margin-bottom: 15px;
}
.user-icon {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background-image: url('https://cdn-icons-png.flaticon.com/512/456/456212.png');
    background-size: cover;
    margin-left: 10px;
    box-shadow: 0 0 10px rgba(255, 0, 0, 0.8);
}
.bubble-user {
    background: rgba(255, 0, 0, 0.08);
    color: #ffcccc;
    padding: 12px 18px;
    border: 2px solid rgba(255, 0, 0, 0.7);
    border-radius: 16px 16px 0 16px;
    box-shadow: 0 0 15px rgba(255, 0, 0, 0.5);
    max-width: 70%;
}

/* --- BOT MESSAGE --- */
.msg-bot {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    margin-bottom: 15px;
}
.bot-icon {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background-image: url('https://cdn-icons-png.flaticon.com/512/4712/4712027.png');
    background-size: cover;
    margin-right: 10px;
    box-shadow: 0 0 10px rgba(0, 191, 255, 0.8);
}
.bubble-bot {
    background: rgba(0, 191, 255, 0.08);
    color: #cce7ff;
    padding: 12px 18px;
    border: 2px solid rgba(0, 191, 255, 0.7);
    border-radius: 16px 16px 16px 0;
    box-shadow: 0 0 15px rgba(0, 191, 255, 0.5);
    max-width: 70%;
}

/* --- FIXED INPUT BAR --- */
.input-area {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: rgba(0, 0, 30, 0.85);
    padding: 15px 30px;
    display: flex;
    justify-content: center;
    box-shadow: 0 -2px 20px rgba(0, 180, 255, 0.3);
}
.input-inner {
    background:#192959;
    padding:10px;
    border-radius:10px;
    display:flex;
    align-items:center;
    gap:10px;
    width:60%;
}
[data-testid="stTextInputRootElement"] input {
    flex:1;
    background: rgba(0, 30, 60, 0.6);
    border: 1px solid #00BFFF;
    border-radius: 12px;
    color: #E6F1FF;
    padding: 10px 14px;
    box-shadow: 0 0 12px rgba(0,180,255,0.5);
}
#send-btn button {
    height:44px !important; width:44px !important;
    border-radius:8px; background:#00BFFF !important; color:#fff !important;
    border:none !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------
#  LOAD DOCX KNOWLEDGE (BACKEND FILE)
# ---------------------
# Replace with the path to your docx file (existing)
backend_docx_path = r"C:\Users\jeevashreer\Downloads\Chatbot_QAs.docx"

def extract_qa(path):
    doc = Document(path)
    qa_pairs = []
    q, a = None, []

    for p in doc.paragraphs:
        t = p.text.strip()
        if not t:
            continue

        lower = t.lower()

        # Detect likely question lines (common patterns)
        # We allow more flexible detection: lines ending with '?' or starting with 'q' tokens
        if t.endswith("?") or (lower.startswith("q") and (len(t) > 1 and (t[1].isdigit() or t[1] in ".:- ")) ) or lower.startswith("q:") or lower.startswith("q."):
            if q and a:
                qa_pairs.append((q, " ".join(a)))
            q = t
            a = []
        elif lower.startswith("a:") or lower.startswith("a.") or (lower.startswith("answer") and ":" in lower):
            # treat the rest of the line as answer start
            a.append(t)
        elif q:
            a.append(t)

    if q and a:
        qa_pairs.append((q, " ".join(a)))

    return qa_pairs

@st.cache_data
def load_vectorized(path, mtime):
    qa = extract_qa(path)
    vect = TfidfVectorizer()
    texts = [ (q + " " + a) for q, a in qa ] or [""]
    X = vect.fit_transform(texts)
    return qa, vect, X

# Ensure file exists
if not os.path.exists(backend_docx_path):
    st.error(f"❌ Docx not found at: {backend_docx_path}")
    st.stop()

# Use file modification time to refresh cache when doc is updated
mtime = os.path.getmtime(backend_docx_path)

# Load QA data
qa_data, vectorizer, X = load_vectorized(backend_docx_path, mtime)
answers = [a for _, a in qa_data]

# Enquiry storage path
ENQUIRIES_CSV = "/mnt/data/ggs_enquiries.csv"

def save_enquiry(original_question, name, email, phone, raw_message):
    header = ["timestamp","original_question","name","email","phone","raw_message"]
    row = [datetime.utcnow().isoformat(), original_question, name, email, phone, raw_message]
    os.makedirs(os.path.dirname(ENQUIRIES_CSV), exist_ok=True)
    write_header = not os.path.exists(ENQUIRIES_CSV)
    with open(ENQUIRIES_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

# Basic regex extractors
email_re = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
phone_re = re.compile(r"((?:\+?\d{1,3}[-\s]?)?(?:\d{6,14}))")
# A naive name extractor: first token(s) capitalized — this is a simple heuristic.
name_re = re.compile(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)")

# ---------------------
#     CHAT DISPLAY
# ---------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hi there! 👋 How can I help you today?"}
    ]

# Flags to control enquiry flow
if "awaiting_contact" not in st.session_state:
    st.session_state.awaiting_contact = False
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""
if "enquiries" not in st.session_state:
    st.session_state.enquiries = []

st.markdown("<h1> GGS Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div class="msg-user">
                <div class="bubble-user">{st.session_state.get('escape_html',False) and st._utils.html.escape(msg['content']) or msg["content"]}</div>
                <img src="https://cdn-icons-png.flaticon.com/512/9131/9131529.png" class="user-icon"/>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="msg-bot">
                <img src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png" class="bot-icon"/>
                <div class="bubble-bot">{st.session_state.get('escape_html',False) and st._utils.html.escape(msg['content']) or msg["content"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------
#   FIXED CHAT INPUT
# ---------------------
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

def try_extract_contact(text):
    """
    Try find email, phone and name heuristically from a text blob.
    Returns (name, email, phone) or (None, None, None) if not found.
    """
    email_match = email_re.search(text)
    phone_match = phone_re.search(text)
    name_match = None

    # Try simple approach:
    # If user types "Name: John Doe, Email: x, Phone: y" it will pick up easily
    # else we try to find capitalized tokens near email/phone
    if "name" in text.lower():
        nm = re.search(r"name[:\-]\s*([A-Za-z\s]{2,50})", text, flags=re.IGNORECASE)
        if nm:
            name_match = nm.group(1).strip()

    if not name_match:
        # fallback to first reasonable capitalized pair
        nm = name_re.search(text)
        if nm:
            name_match = nm.group(1).strip()

    email = email_match.group(0) if email_match else None
    phone = phone_match.group(1) if phone_match else None
    name = name_match if name_match else None

    return name, email, phone

def send_message():
    user_input = st.session_state.chat_input.strip()
    if not user_input:
        return

    # Save user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # If we are currently awaiting contact details for a previous unknown question
    if st.session_state.awaiting_contact:
        # Try to parse the provided message for contact info
        name, email, phone = try_extract_contact(user_input)

        if name and email and phone:
            # save enquiry
            save_enquiry(st.session_state.pending_question, name, email, phone, user_input)
            st.session_state.enquiries.append({
                "question": st.session_state.pending_question,
                "name": name,
                "email": email,
                "phone": phone,
                "raw": user_input,
                "ts": datetime.utcnow().isoformat()
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Thanks {name}! 🎉 We've recorded your enquiry. Our team will reach out to {email} / {phone} shortly."
            })
            # clear awaiting state
            st.session_state.awaiting_contact = False
            st.session_state.pending_question = ""
            st.session_state.chat_input = ""
            return
        else:
            # If parsing failed, ask explicitly for missing pieces
            missing = []
            if not name:
                missing.append("name")
            if not email:
                missing.append("email address")
            if not phone:
                missing.append("contact number")
            missing_text = ", ".join(missing)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"I couldn't detect your {missing_text}. Please provide your Name, Email, and Contact number (e.g. John Doe, john@example.com, +919876543210)."
            })
            st.session_state.chat_input = ""
            return

    # ---- Find relevant context ----
    user_vec = vectorizer.transform([user_input])
    sims = cosine_similarity(user_vec, X)
    best_idx = int(np.argmax(sims))
    score = float(sims[0][best_idx])
    context = answers[best_idx] if score > 0.25 and len(answers)>0 else ""

    # If no matching answer is found, start enquiry capture flow
    if not context:
        st.session_state.pending_question = user_input
        st.session_state.awaiting_contact = True
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": (
                "I don't have a direct answer in my knowledge base for that. "
                "If you'd like, our enquiries team can assist you further — please provide your **Name**, **Email**, and **Contact Number** "
                "so they can reach out. Example format: `John Doe, john@example.com, +919876543210`"
            )
        })
        st.session_state.chat_input = ""
        return

    # ---- Prepare messages for model ----
    messages = [{"role": "system", "content": "You are GGS Smart Chatbot, helpful and concise."}]
    for m in st.session_state.chat_history[-10:]:
        messages.append({"role": m["role"], "content": m["content"]})
    if context:
        messages.append({"role": "system", "content": f"Company context: {context}"})

    # ---- Generate reply using OpenAI ----
    with st.spinner("🤖 Bot is typing..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.4,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            reply = "Sorry — I'm having trouble reaching the model right now. Please try again later."

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.session_state.chat_input = ""

# open the input-area div
st.markdown("<div class='input-area'>", unsafe_allow_html=True)

col1, col2 = st.columns([12, 1], gap="small")
with col1:
    st.text_input(
        "Type your message...",
        key="chat_input",
        placeholder="Type your message...",
        label_visibility="collapsed",
        on_change=send_message
    )
with col2:
    st.button("➤", on_click=send_message, key="send-btn")

# close the inner and outer divs
st.markdown("</div>", unsafe_allow_html=True)  # close input-inner
st.markdown("</div>", unsafe_allow_html=True)  # close input-area
