import streamlit as st
import json
import matplotlib.pyplot as plt
import pandas as pd
import google.generativeai as genai
import io
import sys
import graphviz
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

# ✅ Securely Fetch API Key from Streamlit Secrets
API_KEY = st.secrets.get("GEMINI_API_KEY")

# ✅ Validate API Key
if not API_KEY:
    st.error("⚠️ Google GenAI API key is missing! Please add it to `.streamlit/secrets.toml`.")
    st.stop()

# ✅ Configure Google GenAI
genai.configure(api_key=API_KEY)

def get_ai_response(user_input):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(user_input)
        if response and response.text:
            return "\n- " + response.text.replace("\n", "\n- ")
        return "⚠️ Error: AI could not generate a response."
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

# Load chat history from file
def load_chat_history():
    try:
        with open("chat_history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_chat_history():
    with open("chat_history.json", "w") as f:
        json.dump(st.session_state.chat_history, f, indent=4)

# Load saved preferences
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

st.set_page_config(page_title="AI Data Science Tutor", page_icon="🤖", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔑 Login to AI Data Science Tutor")
    username = st.text_input("Enter your username:")
    if st.button("Login"):
        if not username:
            st.warning("Please enter your username to proceed.")
        else:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
    st.stop()

st.sidebar.title("🔑 User")
st.sidebar.write(f"👋 Welcome, {st.session_state.username}!")

# Apply Dark Mode Styling
if st.session_state.dark_mode:
    st.markdown("""
        <style>
            body { background-color: #1E1E1E; color: white; }
            .stButton>button { background-color: #444; color: white; border-radius: 5px; }
        </style>
    """, unsafe_allow_html=True)

# Sidebar - Settings
st.sidebar.title("⚙️ Settings")
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
st.session_state.dark_mode = dark_mode


st.sidebar.title("📜 Chat History")
if st.sidebar.button("🗑 Clear Chat History"):
    st.session_state.chat_history = []
    save_chat_history()

if st.sidebar.button("📥 Download Chat History"):
    formatted_chat = "\n".join([f"**{st.session_state.username}:** {q}\n**AI:** {a}" for q, a in st.session_state.chat_history])
    st.sidebar.download_button(label="Download", data=formatted_chat, file_name="chat_history.txt", mime="text/plain")

st.title("🧠 Conversational AI Data Science Tutor")

quick_questions = [
    "What is overfitting in ML?",
    "Explain bias-variance tradeoff.",
    "Types of regression?",
    "Supervised vs. Unsupervised learning?",
]
cols = st.columns(len(quick_questions))
for idx, question in enumerate(quick_questions):
    if cols[idx].button(question):
        st.session_state.chat_history.append((st.session_state.username, question))
        response = get_ai_response(question)
        st.session_state.chat_history.append(("assistant", response))
        save_chat_history()
        st.rerun()

st.subheader("🗨 Chat")
chat_container = st.container()

with chat_container:
    for role, text in st.session_state.chat_history:
        if role == st.session_state.username:
            st.markdown(f"**👤 {st.session_state.username}:** {text}")
        else:
            st.markdown(f"**🤖 AI:** {text}")

user_input = st.chat_input("Ask a Data Science question...")
if user_input:
    st.session_state.chat_history.append((st.session_state.username, user_input))
    response = get_ai_response(user_input)
    st.session_state.chat_history.append(("assistant", response))
    save_chat_history()
    st.rerun()

st.sidebar.title("📝 Python Code Editor")
if "code" not in st.session_state:
    st.session_state.code = ""
st.session_state.code = st.sidebar.text_area("Write your Python code here:", value=st.session_state.code, height=200)
code_col1, code_col2 = st.sidebar.columns([0.5, 0.5])
if code_col1.button("Run Code"):
    st.subheader("📝 Python Code Execution")
    st.code(st.session_state.code, language="python")
    try:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        exec_globals = {}
        exec(st.session_state.code, exec_globals)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        st.subheader("📤 Output:")
        st.code(output, language="python")
        if plt.get_fignums():
            st.subheader("📊 Visualization Output:")
            st.pyplot(plt.gcf())
        st.subheader("🧐 Code Explanation:")
        explanation = get_ai_response(f"Explain this Python code: {st.session_state.code}")
        st.markdown(explanation)
    except Exception as e:
        st.error(f"Error: {e}")
if code_col2.button("Clear Code"):
    st.session_state.code = ""
    st.rerun()


st.sidebar.title("📊 Data Comparisons")
data_option = st.sidebar.selectbox("Select comparison", ["None", "ML Models", "Algorithms"])

def get_comparison_table(option):
    if option == "ML Models":
        return pd.DataFrame({
            "Model": ["Linear Regression", "Decision Tree", "SVM"],
            "Accuracy": [85, 78, 82],
            "Training Time": ["Fast", "Medium", "Slow"]
        })
    elif option == "Algorithms":
        return pd.DataFrame({
            "Algorithm": ["K-Means", "DBSCAN", "Hierarchical"],
            "Scalability": ["High", "Medium", "Low"],
            "Use Case": ["Clustering", "Anomaly Detection", "Dendrogram Analysis"]
        })
    return None

comparison_table = get_comparison_table(data_option)
if comparison_table is not None:
    st.table(comparison_table)

st.sidebar.title("📊 Data Science Visualizations")
visualization_option = st.sidebar.selectbox("Select visualization", ["None", "Decision Tree", "Neural Network", "K-Means Clustering"])

def get_visualization(option):
    visualizations = {
        "Decision Tree": "digraph G {A -> B; A -> C;}",
        "Neural Network": "digraph G {A -> B; B -> C; C -> D;}",
        "K-Means Clustering": "digraph G {Cluster1 -> Point1; Cluster1 -> Point2; Cluster2 -> Point3;}"
    }
    return visualizations.get(option, None)

visualization = get_visualization(visualization_option)
if visualization:
    st.graphviz_chart(visualization)
