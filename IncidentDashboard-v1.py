import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter

# ---------------------------
# Load Model (local embedding)
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("incidents.csv")
    df["text"] = df["title"] + " " + df["description"]
    return df

df = load_data()

# ---------------------------
# Create Embeddings
# ---------------------------
@st.cache_resource
def create_index(texts):
    embeddings = model.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

index, embeddings = create_index(df["text"].tolist())

# ---------------------------
# Search Function
# ---------------------------
def search_incident(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    results = df.iloc[indices[0]].copy()
    results["score"] = distances[0]
    return results

# ---------------------------
# KPI Calculation
# ---------------------------
def compute_kpis():
    categories = df["category"].value_counts()
    total = len(df)
    return categories, total

categories, total_incidents = compute_kpis()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🚀 AIOps Incident Assistant (Local)")

st.sidebar.header("📊 KPIs")
st.sidebar.write(f"Total Incidents: {total_incidents}")
st.sidebar.write("Category Distribution:")
st.sidebar.bar_chart(categories)

# ---------------------------
# User Input
# ---------------------------
st.subheader("🔍 Search Incident")

title = st.text_input("Incident Title")
desc = st.text_area("Description")

if st.button("Analyze Incident"):
    query = title + " " + desc

    results = search_incident(query)

    st.subheader("🔎 Top Matches")

    best_match = results.iloc[0]
    similarity_score = 1 / (1 + best_match["score"])  # convert distance

    for i, row in results.iterrows():
        st.write(f"**Title:** {row['title']}")
        st.write(f"Category: {row['category']}")
        st.write(f"Resolution: {row['resolution']}")
        st.write("---")

    st.subheader("🧠 Decision")

    if similarity_score > 0.5:
        st.success("✅ Known Issue Detected")

        st.write(f"**Category:** {best_match['category']}")
        st.write(f"**Resolution:** {best_match['resolution']}")
        st.write(f"**Confidence:** {round(similarity_score*100,2)}%")

        # ---------------------------
        # Auto Action (Optional)
        # ---------------------------
        if best_match["type"] == "restart":
            st.warning("⚡ Auto Action Available: Restart Service")

            if st.button("Execute Restart"):
                st.write("🔧 Simulating restart...")
                # Add real script here
                # os.system("restart_script.sh")

    else:
        st.error("❌ Unknown Issue - Escalate to L2")

# ---------------------------
# Feedback Section
# ---------------------------
st.subheader("👍 Feedback")

feedback = st.radio("Was this helpful?", ["Yes", "No"])

if st.button("Submit Feedback"):
    st.write("✅ Feedback recorded (extend this to DB/logging)")