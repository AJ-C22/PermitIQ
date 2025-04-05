import streamlit as st
import os
import random
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="PermitIQ - Dashboard", layout="wide")

PRIMARY_BLUE = "#1E90FF"
SECONDARY_GREEN = "#32CD32"

# --- Header ---
st.markdown(
    f"""
    <h1 style='font-size: 2.8em;'>
        <span style='color:{PRIMARY_BLUE};'>Permit</span>
        <span style='color:{SECONDARY_GREEN};'>IQ</span> - Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)

# --- Fake Departments and Files ---
departments = ["Zoning", "Planning", "Building", "Public Works", "Fire"]

def generate_dummy_docs(dept):
    return [
        {
            "department": dept,
            "filename": f"{dept.lower()}_doc_{i+1}.pdf",
            "received": (datetime.now() - timedelta(hours=random.randint(1, 72))).strftime("%Y-%m-%d %H:%M"),
            "status": random.choice(["New", "In Review", "Archived"])
        }
        for i in range(random.randint(2, 5))
    ]

# Generate all dummy data for home view
document_data = []
for dept in departments:
    document_data.extend(generate_dummy_docs(dept))
df_docs = pd.DataFrame(document_data)

# --- Sidebar Navigation ---
st.sidebar.header("📁 Navigation")
view = st.sidebar.selectbox("Go to", ["Home", *departments])

# --- Home View ---
if view == "Home":
    st.subheader("📊 Document Count by Department")
    count_by_dept = df_docs.groupby("department").size().reset_index(name="count")
    fig = px.bar(
        count_by_dept,
        x="department",
        y="count",
        color="department",
        color_discrete_sequence=[PRIMARY_BLUE, SECONDARY_GREEN, PRIMARY_BLUE, SECONDARY_GREEN, PRIMARY_BLUE],
        title="Documents per Department"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 All Documents")
    st.dataframe(df_docs[["department", "filename", "received", "status"]], use_container_width=True)

# --- Department View ---
else:
    st.subheader(f"📂 Documents for {view} Department")
    docs = df_docs[df_docs["department"] == view]

    cols = st.columns([3, 2, 2])
    cols[0].markdown("**Filename**")
    cols[1].markdown("**Received**")
    cols[2].markdown("**Status**")

    for _, doc in docs.iterrows():
        cols = st.columns([3, 2, 2])
        cols[0].markdown(f"🔗 {doc['filename']}")
        cols[1].markdown(doc["received"])
        status_color = (
            SECONDARY_GREEN if doc["status"] == "New" else
            "orange" if doc["status"] == "In Review" else
            "gray"
        )
        cols[2].markdown(f"<span style='color:{status_color}; font-weight:bold'>{doc['status']}</span>", unsafe_allow_html=True)

    if any(doc['status'] == 'New' for _, doc in docs.iterrows()):
        st.markdown(f"""
        <div style='padding: 10px; background-color: #e6f7ff; border-left: 5px solid {PRIMARY_BLUE}; margin-top: 30px;'>
            <strong>🔔 New documents have arrived! </strong> Please review and assign them.
        </div>
        """, unsafe_allow_html=True)

    if st.button("✅ Mark All as Read"):
        st.success("Marked all documents as read (simulation only)")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("🚧 This is a demo dashboard using dummy data. Backend integration coming soon!")
