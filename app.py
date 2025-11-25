import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
from openai import OpenAI
from dotenv import load_dotenv
import os

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

# Name of the docx file inside your GitHub repo
backend_docx_path = "GGS_Chatbot_150_QA.docx"

def extract_qa(path):
    doc = Document(path)   # read docx from repo
    qa_pairs = []
    q, a = None, []

    for p in doc.paragraphs:
        t = p.text.strip()
        if not t:
            continue

        if t.lower().startswith("q"):  # New question
            if q and a:
                qa_pairs.append((q, " ".join(a)))
            q, a = t, []

        elif t.lower().startswith("a"):  # Answer line
            a.append(t)

        elif q:
            a.append(t)

    if q and a:
        qa_pairs.append((q, " ".join(a)))

    return qa_pairs


@st.cache_data
def load_vectorized(path):
    qa = extract_qa(path)
    vect = TfidfVectorizer()
    X = vect.fit_transform([q + " " + a for q, a in qa])
    return qa, vect, X


# Load backend docx file (NO user upload)
qa_data, vectorizer, X = load_vectorized(backend_docx_path)
answers = [a for _, a in qa_data]


# ---------------------
#     CHAT DISPLAY
# ---------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hi there! 👋 How can I help you today?"}
    ]

st.markdown("<h1> GGS Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div class="msg-user">
                <div class="bubble-user">{msg["content"]}</div>
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
                <div class="bubble-bot">{msg["content"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------
#  PRE-EXISTING FORMAL QUESTIONS
# ---------------------
st.markdown("### 📘 Quick General Questions")

general_questions = [
    "What services does GGS provide?",
    "How can I contact GGS team?",
    "What are the working hours?",
]

cols = st.columns(3)

for i, q in enumerate(general_questions):
    with cols[i % 3]:
        if st.button(q, key=f"qbtn_{i}"):

            # Use TF-IDF to find answer directly from DOCX
            user_vec = vectorizer.transform([q])
            sims = cosine_similarity(user_vec, X)
            best_idx = np.argmax(sims)
            score = sims[0][best_idx]

            if score > 0.25:
                answer = answers[best_idx]
                st.success(f"**Answer:**\n\n{answer}")
            else:
                st.error("❌ Sorry, no matching answer found in the document.")

# ---------------------
#   FIXED CHAT INPUT
# ---------------------
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

def send_message():
    user_input = st.session_state.chat_input.strip()
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        # ---- Find relevant context ----
        user_vec = vectorizer.transform([user_input])
        sims = cosine_similarity(user_vec, X)
        best_idx = np.argmax(sims)
        score = sims[0][best_idx]
        context = answers[best_idx] if score > 0.25 else ""
        # ---- Prepare messages for model ----
        messages = [{"role": "system", "content": "You are GGS Smart Chatbot, helpful and concise."}]
        for m in st.session_state.chat_history[-10:]:
            messages.append({"role": m["role"], "content": m["content"]})
        if context:
            messages.append({"role": "system", "content": f"Company context: {context}"})
        # ---- Generate reply ----
        with st.spinner("🤖 Bot is typing..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.4,
            )
            reply = response.choices[0].message.content.strip()
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
