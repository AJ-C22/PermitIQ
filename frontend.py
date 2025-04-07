import streamlit as st
import os
import random
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="PermitIQ - Dashboard", layout="wide")
base="light"

hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

PRIMARY_BLUE = "#17A9CE"
SECONDARY_GREEN = "#6BC856"
colors = [
    "#17A9CE",  
    "#6BC856",  
    "#75D0E6",  
    "#3A8BB3", 
    "#7FCA7F", 
    "#2C8D32"   
]

col1, col2 = st.columns([0.8, 0.2])

# --- Fake Data ---
departments = ["Zoning", "Planning", "Building", "Public Works", "Fire"]

# --- Sidebar Navigation ---
st.sidebar.header("📁 Navigation")
view = st.sidebar.selectbox("Go to", ["Home", *departments])

with col1:
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
    if view == "Home":
        st.subheader("📊 Document Count by Department")

with col2:
    
    uploaded_file = st.file_uploader(
        label="Upload File",
        type=["csv", "txt", "xlsx"],
        label_visibility="collapsed"
    )

dummy_documents = {
    "Zoning": [
        {"filename": "zoning_doc_1.pdf", "received": "2025-04-01 10:00", "status": "New"},
        {"filename": "zoning_doc_2.pdf", "received": "2025-04-02 11:30", "status": "In Review"},
        {"filename": "zoning_doc_3.pdf", "received": "2025-04-02 11:45", "status": "In Review"},
        {"filename": "zoning_doc_4.pdf", "received": "2025-04-02 9:30", "status": "In Review"},
    ],
    "Planning": [
        {"filename": "planning_doc_1.pdf", "received": "2025-04-01 09:15", "status": "Archived"},
        {"filename": "planning_doc_2.pdf", "received": "2025-04-03 14:45", "status": "New"},
        {"filename": "planning_doc_3.pdf", "received": "2025-04-02 14:44", "status": "New"}
    ],
    "Building": [
        {"filename": "building_doc_1.pdf", "received": "2025-04-02 08:20", "status": "In Review"},
    ],
    "Public Works": [
        {"filename": "public_works_doc_1.pdf", "received": "2025-04-03 13:00", "status": "New"},
        {"filename": "public_works_doc_2.pdf", "received": "2025-04-05 09:45", "status": "In Review"},
        {"filename": "public_works_doc_3.pdf", "received": "2025-04-06 10:43", "status": "In Review"},
        {"filename": "public_works_doc_4.pdf", "received": "2025-04-07 23:45", "status": "New"},
        {"filename": "public_works_doc_5.pdf", "received": "2025-04-11 01:45", "status": "New"}
    ],
    "Fire": [
        {"filename": "fire_doc_1.pdf", "received": "2025-04-04 07:30", "status": "Archived"},
        {"filename": "fire_doc_2.pdf", "received": "2025-04-05 12:20", "status": "New"},
    ],
}

document_data = []
for department, docs in dummy_documents.items():
    for doc in docs:
        document_data.append({
            "department": department,
            "filename": doc["filename"],
            "received": datetime.strptime(doc["received"], "%Y-%m-%d %H:%M"),
            "status": doc["status"]
        })

df_docs = pd.DataFrame(document_data)



# --- Home View ---
if view == "Home":
    count_by_dept = df_docs.groupby("department").size().reset_index(name="count")
    fig = px.bar(
        count_by_dept,
        x="department",
        y="count",
        color="department",
        color_discrete_sequence=colors,
        title="Documents per Department"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig.update_layout(
    xaxis=dict(fixedrange=True),
    yaxis=dict(fixedrange=True)
    )

    selected_dept = st.selectbox("🔍 Filter documents by department", ["All"] + departments)

    if selected_dept == "All":
        filtered_docs = df_docs
    else:
        filtered_docs = df_docs[df_docs["department"] == selected_dept]

    st.subheader("📋 Filtered Documents")
    st.dataframe(filtered_docs[["department", "filename", "received", "status"]], use_container_width=True)

    # if selected_dept != "All":
    #     if st.button(f"➡️ Go to {selected_dept} Department View"):
    #         st.session_state.view = selected_dept
    #         st.experimental_rerun()


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
