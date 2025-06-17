# The following code is commented out because pysqlite3-binary is not available for Python 3.11 on macOS ARM
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
import joblib
from dotenv import load_dotenv
import os
import random
import time
from preprocessing import clean_description
from mockdata import MOCK_PERMIT_DATA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

st.set_page_config(layout="wide", page_title="PermitIQ", page_icon="logo.png")

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PDF_FOLDER = "LA County Permitting" 
PDF_PERSIST_DIRECTORY = "chroma_db_permitting"
PDF_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PDF_GEMINI_MODEL = "gemini-1.5-flash-latest"

# --- Load Classifier Model ---
MODEL_PATH = "permit_classifier_pipeline.joblib"
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        model = None
else:
    st.sidebar.warning("⚠️ Model file not found. Type prediction disabled.")
    model = None

# --- Mock Data / Configuration ---
PRIMARY_BLUE = "#17A9CE"
SECONDARY_GREEN = "#6BC856"
DEPARTMENTS = ["Department of Public Works", 
    "Fire Department",
    "Regional Planning",
    "Department of Environmental Health",
    "Department of Public Health",
    "Department of Parks and Recreation",]
DEPARTMENTS_NEEDING_REVIEW = {
    "Building Permit": ["Fire", "Public Works"],
    "ADU Conversion": ["Regional Planning", "Public Works"],
    "Site Plan Review": ["Regional Planning"],
}
PERMIT_TYPES = ['unincorporated building residential',
    'road',
    'unincorporated electrical',
    'unincorporated mechanical',
    'fire',
    'unincorporated sewer']
# Define status groups
ACTIVE_STATUSES = ["Submitted", "In Review", "Needs Info", "External Review"]
COMPLETED_STATUSES = ["Approved", "Rejected", "Withdrawn"]
PERMIT_STATUS_OPTIONS = ACTIVE_STATUSES + COMPLETED_STATUSES # Combined list
CONFIDENCE_THRESHOLD = 0.93 

# --- Session State Initialization ---
if "requests" not in st.session_state or not st.session_state.requests: 
    st.session_state.requests = MOCK_PERMIT_DATA.copy() 
if "reviewer_idx" not in st.session_state:
    st.session_state.reviewer_idx = 0
if "view" not in st.session_state:
    st.session_state.view = "form"
if "pdf_response" not in st.session_state: 
    st.session_state.pdf_response = None
if "dashboard_detail" not in st.session_state:
    st.session_state.dashboard_detail = "Home"
if 'rejecting_classification_id' not in st.session_state:
    st.session_state.rejecting_classification_id = None

LOGO_BLUE = "#17A9CE"
LOGO_GREEN = "#6BC856"
CUSTOM_COLORS = ["#1B3F7D", "#2D66A7", "#498ABA", "#6AB1CF", "#8ECAC4", "#B3DBB8"]

# --- Function to Load PDF QA Chain ---
@st.cache_resource 
def load_pdf_qa_chain():
    """Loads the vector store and initializes the QA chain with Gemini."""
    if not os.path.exists(PDF_PERSIST_DIRECTORY):
        st.sidebar.error(f"⚠️ PDF Vector Store ('{PDF_PERSIST_DIRECTORY}') not found.")
        st.sidebar.caption("Ensure the vector store is in your GitHub repo or run build script locally if testing.")
        return None
    try:
        # 1. Get Google API Key from Streamlit Secrets
        google_api_key = os.environ.get("GEMINI_API_KEY")
        if not google_api_key:
            st.sidebar.error("⚠️ GEMINI_API_KEY not found in Streamlit Secrets.")
            return None

        # 2. Load Embeddings (for retrieving from Chroma)
        embeddings = HuggingFaceEmbeddings(model_name=PDF_EMBEDDING_MODEL)

        # 3. Load Vector Store
        vectorstore = Chroma(
            persist_directory=PDF_PERSIST_DIRECTORY,
            embedding_function=embeddings
        )

        # 4. Initialize the Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model=PDF_GEMINI_MODEL,
            google_api_key=google_api_key,
            convert_system_message_to_human=True,
            temperature=0.5
        )

        # 5. Create the Retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # 6. Create the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        st.sidebar.success("✅ Chatbot System Loaded")
        return qa_chain
    except Exception as e:
        st.sidebar.error(f"⚠️ Failed to load Chatbot System")
        st.sidebar.caption(f"Error: {e}")
        return None

# --- Helper Functions ---
def get_next_reviewer():
    """Simple round-robin assignment."""
    reviewer = DEPARTMENTS[st.session_state.reviewer_idx % len(DEPARTMENTS)]
    st.session_state.reviewer_idx += 1
    return reviewer

def generate_mock_checklist(permit_type):
    """Generates placeholder checklist text."""
    items = ["Application Form", "Site Plan", "Floor Plan", "Proof of Ownership"]
    if "Electrical" in permit_type: items.append("Load Calculation Sheet")
    if "Plumbing" in permit_type: items.append("Plumbing Riser Diagram")
    if "ADU" in permit_type: items.append("Existing Structure Photos")
    return "\n".join([f"- [ ] {item}" for item in items])

def generate_mock_issuance_doc(permit_data):
    """Generates placeholder issuance document text."""
    return f"""
    **PERMIT ISSUED**
    -------------------
    Permit Type: {permit_data.get('Permit Type', 'N/A')}
    Project: {permit_data.get('Project Name', 'N/A')}
    Applicant: {permit_data.get('Applicant', 'N/A')}
    Issue Date: {datetime.now().strftime('%Y-%m-%d')}
    Valid Until: {(datetime.now() + timedelta(days=180)).strftime('%Y-%m-%d')}
    Assigned Inspector: TBD
    Conditions: Standard conditions apply. Refer to approved plans.
    """

# --- Navigation Sidebar ---
logo_col1, logo_col2, logo_col3 = st.sidebar.columns([1, 4, 1])
with logo_col2:
    try:
        st.image("logo.png", width=240)
    except Exception as e:
        st.error(f"Logo not found: {e}")

title_col1, title_col2, title_col3 = st.sidebar.columns([1, 4, 1])
with title_col2:
    st.markdown(
    f"""
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;700&display=swap" rel="stylesheet">
    <style>
        .permit-iq-title {{
            font-family: 'Montserrat', sans-serif;
            text-align: center;
            font-size: 2.5rem !important;
            margin-top: 0px;
            margin-bottom: 0px;
        }}
        .permit-iq-title .blue {{
            font-weight: 700;
            color: {LOGO_BLUE};
        }}
        .permit-iq-title .green {{
            font-weight: 450;
            color: {LOGO_GREEN};
        }}
    </style>

    <h1 class="permit-iq-title">
        <span class="blue">PERMIT</span><span class="green">IQ</span>
    </h1>
    """,
    unsafe_allow_html=True
    )

st.sidebar.divider() 

selected_view = st.sidebar.radio(
    "Main View",
    ["📝 Submit New Request", "📊 Internal Dashboard"],
    key="nav_radio",
    captions=["Fill out a permit application.", "Track and manage requests."]
)
st.session_state.view = "form" if selected_view == "📝 Submit New Request" else "dashboard"
st.sidebar.divider()

pdf_qa_chain = load_pdf_qa_chain()
st.sidebar.divider()

if st.session_state.view == "dashboard":
    dashboard_views = ["Home"] + DEPARTMENTS
    current_detail_index = dashboard_views.index(st.session_state.dashboard_detail) if st.session_state.dashboard_detail in dashboard_views else 0
    selected_detail_view = st.sidebar.selectbox(
        "Dashboard View",
        options=dashboard_views,
        index=current_detail_index,
        key="dashboard_select"
    )
    if selected_detail_view != st.session_state.dashboard_detail:
        st.session_state.dashboard_detail = selected_detail_view
        st.rerun()
    st.sidebar.divider()

form_submitted_this_run = False

# ========== FORM PAGE ==========

if st.session_state.view == "form":
    
    # --- PDF Query Section ---
    # okay
    st.subheader(f"Ask Ethica AI about Procedures & Permits")
    
    if pdf_qa_chain: 
        pdf_query_col, pdf_btn_col = st.columns([20, 1])
        with pdf_query_col:
            pdf_question_input = st.text_input("Ask the PDFs...", key="pdf_input", placeholder="e.g., What is the setback for Zone R1?", label_visibility="collapsed")
        with pdf_btn_col:
            pdf_submit_button = st.button("➡️", key="pdf_submit", use_container_width=True)
        st.caption("Ethica AI is available to assist with your permitting application.")

        if pdf_submit_button and pdf_question_input:
            st.session_state.pdf_response = None 
            with st.spinner(f"Ethica AI is processing your query..."):
                try:
                    answer = pdf_qa_chain.run(pdf_question_input)
                    st.session_state.pdf_response = answer 
                except Exception as e:
                     st.error(f"Error querying PDFs: {e}")
                     st.session_state.pdf_response = None 
    else:
        st.warning("Ethica AI's PDF Question Answering system could not be loaded. Check sidebar errors.")

    if st.session_state.pdf_response:
        st.markdown("**Ethica AI Response:**")
        with st.expander("View PDF Query Response", expanded=True):
            st.markdown(st.session_state.pdf_response)

    st.divider()

    st.title("Submit a Permit Request")
    st.markdown("Fill in the details below to submit your permit application.")

    # Initialize variables to store outcomes for display after submission
    classification_outcome_msg = None
    classification_confidence_val = None
    final_permit_type = None
    needs_approval_flag = None
    assigned_dept = None
    submitted_project_name = None
    form_submitted_this_run = False

    with st.form("permit_form", clear_on_submit=True):
        st.subheader("Applicant & Project Information")
        col1, col2 = st.columns(2)
        with col1:
            applicant_name = st.text_input("Applicant Name*", key="applicant_name")
            project_name = st.text_input("Project Name*", key="project_name")
        with col2:
            email = st.text_input("Applicant Email*", key="email")
            project_location = st.text_input("Project Location/Address*", key="project_location", placeholder="e.g., 123 Main St, Anytown")

        st.subheader("Permit Details")
        col3, col4 = st.columns(2)
        with col3:
            department_choice = st.selectbox("Primary Department*", ["Building & Safety", "Fire", "Public Works", "Regional Planning"], key="department")
        with col4:
            user_permit_type = st.selectbox("Requested Permit Type*", PERMIT_TYPES, key="user_permit_type")

        description = st.text_area("Detailed Project Description*", height=150, key="description", placeholder="Describe the scope of work...")
        submission_date = datetime.today()

        submitted = st.form_submit_button("➡️ Submit Permit Request", use_container_width=True, type="primary")
        
        if submitted:
                if not all([applicant_name, project_name, email, project_location, description]):
                    st.error("⚠️ Please fill in all required fields marked with *.")
                else:
                    form_submitted_this_run = True
                    with st.status("Processing submission...", expanded=True) as status:
                        st.write("⏳ Validating input...")
                        time.sleep(0.5)

                        # --- Modified Classification Logic ---
                        predicted_type = user_permit_type # Start with user's choice ALWAYS
                        confidence = None
                        classification_approved = True # Default to approved, as we now prioritize user input on low confidence
                        classification_status_message = "N/A"

                        if model and description:
                            try:
                                st.write("✨ Cleaning description...")
                                cleaned_input_description = clean_description(description)

                                st.write("🤖 Auto-classifying...")
                                model_prediction = model.predict([cleaned_input_description])[0]
                                prediction_proba = None

                                if hasattr(model, "predict_proba"):
                                    prediction_proba = model.predict_proba([cleaned_input_description])
                                    confidence = prediction_proba.max()

                                    # CASE 1: High Confidence & AI agrees with user OR AI differs but meets threshold
                                    if confidence >= CONFIDENCE_THRESHOLD:
                                        if model_prediction != user_permit_type:
                                            st.write(f"✅ AI suggested type **{model_prediction}** used (Confidence: {confidence:.2f} >= {CONFIDENCE_THRESHOLD})")
                                            predicted_type = model_prediction # Use AI prediction
                                            classification_status_message = "AI prediction used (High Confidence)"
                                        else:
                                            
                                            predicted_type = user_permit_type
                                            classification_status_message = f"AI confirmed user type (Confidence: {confidence:.2f})"
                                        classification_approved = True

                                    # CASE 2: Low Confidence (< Threshold)
                                    else: 
                                        predicted_type = user_permit_type # ALWAYS use user type if confidence is low
                                        classification_approved = False
                                        if model_prediction != user_permit_type:
                                            st.write(f"⚠️ AI suggested type **{model_prediction}** ignored (Confidence: {confidence:.2f} < {CONFIDENCE_THRESHOLD}). Using user-selected type.")
                                            classification_status_message = f"User type kept (AI suggestion '{model_prediction}'. Permit Flagged)"
                                        else:
                                            # AI agreed but confidence was low
                                            classification_status_message = f"User type kept (AI agreed but low confidence: {confidence:.2f})"

                                else: 
                                    confidence = None 
                                    if model_prediction != user_permit_type:
                                        st.write(f"ℹ️ AI suggested type: **{model_prediction}** (Confidence N/A). Needs manual check.")
                                        # Even without confidence, if model differs significantly, maybe still flag? Or stick to user? Let's stick to user for now.
                                        predicted_type = user_permit_type # Stick to user input if confidence unknown
                                        classification_approved = True # Approved as we use user input
                                        classification_status_message = "User type kept (AI suggestion ignored, confidence N/A)"
                                        # Alternative: Set predicted_type = model_prediction, classification_approved = False here if desired
                                    else:
                                        predicted_type = user_permit_type
                                        classification_approved = True
                                        classification_status_message = "User type kept (AI confirmed, confidence N/A)"

                                time.sleep(0.5)
                            except Exception as e:
                                st.warning(f"⚠️ Classification failed: {e}. Using user-selected type.")
                                predicted_type = user_permit_type; confidence = None; classification_approved = True; classification_status_message = "Classification failed"
                        else:
                            predicted_type = user_permit_type; confidence = None; classification_approved = True; classification_status_message = "Classification skipped"
                        # --- End Modified Classification ---

                        st.write("Assigning reviewer...")
                        assigned_reviewer = get_next_reviewer()
                        time.sleep(0.5)

                        # Prepare new entry (Classification Approved is now mostly True unless model lacks predict_proba AND differs)
                        new_entry = {
                            "ID": f"P{random.randint(10000, 99999)}", "Project Name": project_name, "Applicant": applicant_name,
                            "Email": email, "Location": project_location, "Department": department_choice,
                            "Permit Type": predicted_type, # Final type determined above
                            "Description": description, "Submission Date": submission_date.strftime("%Y-%m-%d"), "Status": "Submitted",
                            "Assigned To": assigned_reviewer, "Last Update": submission_date.strftime("%Y-%m-%d %H:%M"),
                            "Needs Review By": DEPARTMENTS_NEEDING_REVIEW.get(predicted_type, []),
                            "Review History": [f"{submission_date.strftime('%Y-%m-%d %H:%M')}: Submitted, assigned to {assigned_reviewer}."],
                            "External Review Status": "N/A",
                            "Classification Confidence": confidence,
                            "Classification Approved": classification_approved # Reflects if AI override occurred or needs check
                        }

                        # Store details for display
                        classification_outcome_msg = classification_status_message
                        classification_confidence_val = confidence
                        final_permit_type = predicted_type
                        # Needs approval flag is now tied directly to the classification_approved variable from the logic above
                        needs_approval_flag = not classification_approved
                        assigned_dept = assigned_reviewer
                        submitted_project_name = project_name

                        st.write("💾 Saving request...")
                        st.session_state.requests.append(new_entry)
                        time.sleep(0.5)

                        status.update(label="✅ Submission Complete!", state="complete", expanded=False)

        # --- Display Results AFTER the form processing ---
        if form_submitted_this_run:
            st.success(f"✅ Permit request '{submitted_project_name}' submitted successfully! Assigned to {assigned_dept}.")
            st.balloons()

            # --- Always Display Classification Outcome ---
            st.markdown("---")
            st.subheader("Classification Summary")
            conf_display = f"{classification_confidence_val:.2f}" if classification_confidence_val is not None else "N/A"
            st.markdown(f"**Final Permit Type Used:** `{final_permit_type}`")
            st.markdown(f"**AI Confidence:** {conf_display}")
            st.markdown(f"**Notes:** {classification_outcome_msg}")

            # Update the warning/info message based on the final needs_approval_flag
            if needs_approval_flag:
                st.warning("⚠️ This classification requires manual review and approval on the dashboard (likely due to missing confidence score).")
            else:
                st.info("✅ Classification determined automatically (High confidence AI override or user selection used).")
            st.markdown("---")

# ========== DASHBOARD PAGE ==========
elif st.session_state.view == "dashboard":
    st.title(f"{st.session_state.dashboard_detail}")

    if not st.session_state.requests:
        st.info("No permit requests submitted yet.")
    else:
        try:
            df_all = pd.DataFrame(st.session_state.requests)

            # --- FIX: Robust date parsing AND Timezone Handling ---
            try:
                # Convert to datetime first, coercing errors
                df_all['Submission Date'] = pd.to_datetime(df_all['Submission Date'], errors='coerce')
                df_all['Last Update DT'] = pd.to_datetime(df_all['Last Update'], errors='coerce')

                # Now handle timezones: Ensure both are UTC
                if 'Submission Date' in df_all.columns and pd.api.types.is_datetime64_any_dtype(df_all['Submission Date']):
                    if df_all['Submission Date'].dt.tz is None:
                        # If naive, assume UTC and localize
                        df_all['Submission Date'] = df_all['Submission Date'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
                    else:
                        # If already aware, convert to UTC
                        df_all['Submission Date'] = df_all['Submission Date'].dt.tz_convert('UTC')

                if 'Last Update DT' in df_all.columns and pd.api.types.is_datetime64_any_dtype(df_all['Last Update DT']):
                     if df_all['Last Update DT'].dt.tz is None:
                         # If naive, assume UTC and localize
                         df_all['Last Update DT'] = df_all['Last Update DT'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
                     else:
                         # If already aware, convert to UTC
                         df_all['Last Update DT'] = df_all['Last Update DT'].dt.tz_convert('UTC')

            except Exception as date_e:
                 st.error(f"⚠️ Error parsing or handling date/timezones: {date_e}")
                 # Fallback: Attempt calculations anyway, but they might fail if dates are problematic
                 pass
            # --- End FIX ---

            now_utc = datetime.now(timezone.utc) # Use this consistent timezone-aware 'now'

            if st.session_state.dashboard_detail == "Home":
                st.subheader(f"Overall Process KPIs")

                # --- KPI Calculations ---
                total_requests = len(df_all)
                active_requests_df = df_all[df_all['Status'].isin(ACTIVE_STATUSES)].copy()
                completed_requests_df = df_all[df_all['Status'].isin(COMPLETED_STATUSES)].copy()

                # Calculate Avg Completed Cycle Time (uses Last Update DT and Submission Date)
                completed_requests_df = completed_requests_df.dropna(subset=['Submission Date', 'Last Update DT'])
                avg_cycle_time = (completed_requests_df['Last Update DT'] - completed_requests_df['Submission Date']).mean() if not completed_requests_df.empty else None

                # Calculate Approval Rate
                approved_count = completed_requests_df[completed_requests_df['Status'] == 'Approved'].shape[0]
                total_completed = len(completed_requests_df)
                approval_rate = (approved_count / total_completed * 100) if total_completed > 0 else 0

                # Counts for specific statuses
                active_reviews = len(active_requests_df)
                needs_info_count = active_requests_df[active_requests_df['Status'] == 'Needs Info'].shape[0]
                external_review_count = active_requests_df[active_requests_df['Status'] == 'External Review'].shape[0]

                # --- Display KPIs ---
                kp_col1, kp_col2, kp_col3 = st.columns(3)
                kp_col1.metric("Total Requests", total_requests)
                kp_col1.metric("Active Reviews", active_reviews)
                # Use .days from timedelta for display
                kp_col2.metric("Avg. Completed Cycle Time (Days)", f"{avg_cycle_time.days:.1f}" if pd.notna(avg_cycle_time) else "N/A")
                kp_col2.metric("Approval Rate", f"{approval_rate:.1f}%" if total_completed > 0 else "N/A")
                kp_col3.metric("Requests Needing Info", needs_info_count)
                kp_col3.metric("External Reviews Pending", external_review_count)

                st.divider()

                # --- Bottleneck Chart Calculation ---
                # Calculate age of active requests
                active_requests_df = active_requests_df.dropna(subset=['Submission Date']) # Ensure no NaT dates
                # This subtraction should now work as both are timezone-aware (UTC)
                active_requests_df['Age'] = (now_utc - active_requests_df['Submission Date'])

                # Group by department and calculate mean age in days
                # Use .dt.total_seconds() for accurate mean then convert to days
                avg_age_seconds = active_requests_df.groupby('Assigned To')['Age'].apply(lambda x: x.dt.total_seconds().mean())
                avg_age_by_dept = (avg_age_seconds / (24 * 3600)).reset_index() # Convert mean seconds to days
                avg_age_by_dept.columns = ['Department', 'Average Age (Days)']
                avg_age_by_dept = avg_age_by_dept.sort_values('Average Age (Days)', ascending=False)

                # --- Display Charts (No changes needed here) ---
                st.subheader("Overall Reporting Charts")
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.write("**Status Distribution (All Requests)**"); status_counts = df_all['Status'].value_counts().reset_index(); status_counts.columns = ['Status', 'Count']; status_spec = {"data": {"values": status_counts.to_dict('records')},"mark": "bar","encoding": {"x": {"field": "Status", "type": "nominal", "axis": {"labelAngle": -45}, "sort":"-y"},"y": {"field": "Count", "type": "quantitative"},"color": {"field": "Status","type": "nominal","scale": {"range": CUSTOM_COLORS},"legend": None},"tooltip": [{"field": "Status"}, {"field": "Count"}]},"config": {"view": {"stroke": "transparent"},"axis": {"domainWidth": 1}}}; st.vega_lite_chart(status_spec, use_container_width=True);
                    st.write("**Average Age of Active Requests by Department**"); st.caption("(Indicates potential current bottlenecks)");
                    if not avg_age_by_dept.empty: bottleneck_spec = {"data": {"values": avg_age_by_dept.to_dict('records')},"mark": "bar","encoding": {"x": {"field": "Department", "type": "nominal", "axis": {"labelAngle": -45}, "sort":"-y"},"y": {"field": "Average Age (Days)", "type": "quantitative"},"color": {"field": "Department","type": "nominal","scale": {"range": CUSTOM_COLORS},"legend": None},"tooltip": [{"field": "Department"}, {"field": "Average Age (Days)", "format": ".1f"}]},"config": {"view": {"stroke": "transparent"}, "axis": {"domainWidth": 1}}}; st.vega_lite_chart(bottleneck_spec, use_container_width=True)
                    else: st.info("No active requests with valid dates to calculate average age.")
                with chart_col2:
                    st.write("**Assignments Distribution (All Requests)**"); assignee_counts = df_all['Assigned To'].value_counts().reset_index(); assignee_counts.columns = ['Assignee', 'Count']; assignee_spec = {"data": {"values": assignee_counts.to_dict('records')},"mark": "bar","encoding": {"x": {"field": "Assignee", "type": "nominal", "axis": {"labelAngle": -45}, "sort":"-y"},"y": {"field": "Count", "type": "quantitative"},"color": {"field": "Assignee","type": "nominal","scale": {"range": CUSTOM_COLORS},"legend": None},"tooltip": [{"field": "Assignee"}, {"field": "Count"}]},"config": {"view": {"stroke": "transparent"},"axis": {"domainWidth": 1}}}; st.vega_lite_chart(assignee_spec, use_container_width=True);

                st.divider()
                st.subheader(f"All Submitted Requests"); st.info(f"Displaying {len(df_all)} total requests."); display_cols_home = ["ID", "Project Name", "Applicant", "Permit Type", "Status", "Assigned To", "Submission Date", "Last Update"]; df_display_home = df_all.copy();
                # Format dates for display AFTER calculations
                if 'Submission Date' in df_display_home.columns and pd.api.types.is_datetime64_any_dtype(df_display_home['Submission Date']): df_display_home['Submission Date'] = df_display_home['Submission Date'].dt.strftime('%Y-%m-%d');
                st.dataframe(df_display_home[display_cols_home], use_container_width=True, hide_index=True)

            # --- DEPARTMENT VIEW ---
            else:
                # ... (Department view logic - Needs similar date formatting for display if Submission Date is shown) ...
                selected_dept = st.session_state.dashboard_detail
                st.markdown(f"Managing requests assigned to **{selected_dept}**.")

                df_dept = df_all[df_all['Assigned To'] == selected_dept].copy() # Inherits corrected dates from df_all

                if df_dept.empty:
                    st.info(f"No requests currently assigned to {selected_dept}.")
                else:
                    st.subheader("Filter Department Requests")
                    filter1, filter3 = st.columns(2)
                    with filter1:
                        # Use list comprehension for default active statuses
                        status_filter = st.multiselect(f"Filter {selected_dept} by Status", options=PERMIT_STATUS_OPTIONS, default=[s for s in ACTIVE_STATUSES], key=f"dept_status_{selected_dept}")
                    with filter3:
                        date_filter = st.date_input(f"Filter {selected_dept} by Submission Date", value=(), key=f"dept_date_{selected_dept}", min_value=df_dept['Submission Date'].min().date() if df_dept['Submission Date'].notna().any() else None, max_value=df_dept['Submission Date'].max().date() if df_dept['Submission Date'].notna().any() else None)

                    filtered_df_dept = df_dept.copy()
                    if status_filter:
                        filtered_df_dept = filtered_df_dept[filtered_df_dept['Status'].isin(status_filter)]
                    # Filter by date using .dt.date to compare dates only
                    if len(date_filter) == 2 and filtered_df_dept['Submission Date'].notna().any():
                         # Ensure column is datetime before using .dt accessor
                        date_col_dt = pd.to_datetime(filtered_df_dept['Submission Date'], errors='coerce')
                        # Filter only where dates are valid (not NaT)
                        valid_dates_mask = date_col_dt.notna()
                        filtered_df_dept = filtered_df_dept[valid_dates_mask & (date_col_dt[valid_dates_mask].dt.date >= date_filter[0]) & (date_col_dt[valid_dates_mask].dt.date <= date_filter[1])]


                    st.subheader(f"Submitted Requests for {selected_dept}")
                    if filtered_df_dept.empty and not df_dept.empty:
                        st.warning("No requests match the current filters for this department.")
                    elif filtered_df_dept.empty:
                        st.info(f"No requests match filters for {selected_dept}.")
                    else:
                        st.info(f"Displaying {len(filtered_df_dept)} of {len(df_dept)} total requests for {selected_dept}.")
                        display_cols = ["ID", "Project Name", "Applicant", "Permit Type", "Status", "Submission Date", "Last Update"]
                        filtered_df_display = filtered_df_dept.copy()
                        # Format Submission Date for display AFTER filtering/calculations
                        if 'Submission Date' in filtered_df_display.columns and pd.api.types.is_datetime64_any_dtype(filtered_df_display['Submission Date']):
                            filtered_df_display['Submission Date'] = filtered_df_display['Submission Date'].dt.strftime('%Y-%m-%d')
                        st.dataframe(filtered_df_display[display_cols], use_container_width=True, hide_index=True)

                        # --- Manage Selected Request (rest of department view logic) ---
                        # ... (This section remains largely the same, just ensure dates displayed read from the original string or formatted version) ...
                        st.subheader("Manage Selected Request")
                        request_ids = filtered_df_dept['ID'].tolist()
                        if request_ids:
                            selected_id = st.selectbox(f"Select Request ID to Manage ({selected_dept})", options=request_ids, key=f"dept_select_{selected_dept}")
                            selected_indices = [i for i, req in enumerate(st.session_state.requests) if req['ID'] == selected_id]

                            if selected_indices:
                                selected_idx = selected_indices[0]; selected_data = st.session_state.requests[selected_idx]

                                with st.expander("View/Edit Details", expanded=True):
                                    detail1, detail2 = st.columns(2)
                                    with detail1: # Displaying original string dates is fine here
                                        st.write(f"**Project:** {selected_data['Project Name']}")
                                        st.write(f"**Applicant:** {selected_data['Applicant']} ({selected_data['Email']})")
                                        st.write(f"**Location:** {selected_data['Location']}")
                                        st.write(f"**Submitted:** {selected_data['Submission Date']}") # Original string
                                        st.write(f"**Type:** {selected_data['Permit Type']}")
                                        # Use text_area for potentially long descriptions
                                        st.markdown("**Description:**")
                                        st.text_area("DescDisp", value=selected_data['Description'], height=120, disabled=True, label_visibility="collapsed", key=f"desc_disp_{selected_id}")

                                        # --- Classification display logic (no changes needed) ---
                                        confidence_score = selected_data.get('Classification Confidence'); classification_approved = selected_data.get('Classification Approved', True);
                                        if confidence_score is not None: confidence_text = f"AI Confidence: **{confidence_score:.2f}**";
                                        if not classification_approved: st.warning(f"⚠️ {confidence_text if confidence_score else 'Confidence N/A'} - Classification requires manual review below.")
                                        else: st.success(f"✅ {confidence_text if confidence_score else 'Confidence N/A'} - Classification approved.")
                                        

                                    # --- Detail 2: Actions + Approval/Rejection (no changes needed here) ---
                                    with detail2:
                                        st.write("**Status & Assignment:**"); current_status = selected_data['Status']; current_assignee = selected_data['Assigned To']; new_status = st.selectbox("Update Status", options=PERMIT_STATUS_OPTIONS, index=PERMIT_STATUS_OPTIONS.index(current_status), key=f"status_{selected_id}"); new_assignee = st.selectbox("Reassign To", options=DEPARTMENTS, index=DEPARTMENTS.index(current_assignee) if current_assignee in DEPARTMENTS else 0, key=f"assignee_{selected_id}");
                                        st.write("**Classification Approval:**"); classification_approved = selected_data.get('Classification Approved', True); classification_approved_status = "Approved" if classification_approved else "Needs Approval"; st.write(f"Status: **{classification_approved_status}**");
                                        if st.session_state.rejecting_classification_id == selected_id:
                                            st.markdown("**Rejecting AI Classification:**");
                                            try: current_type_index = PERMIT_TYPES.index(selected_data['Permit Type'])
                                            except ValueError: current_type_index = 0
                                            new_manual_type = st.selectbox("Select Correct Permit Type*", options=PERMIT_TYPES, index=current_type_index, key=f"manual_type_select_{selected_id}")
                                            confirm_col, cancel_col = st.columns(2);
                                            with confirm_col:
                                                if st.button("Confirm Rejection & Update Type", type="primary", key=f"confirm_reject_{selected_id}"): now_str = datetime.now().strftime("%Y-%m-%d %H:%M"); old_type = selected_data['Permit Type']; st.session_state.requests[selected_idx]['Permit Type'] = new_manual_type; st.session_state.requests[selected_idx]['Classification Approved'] = True; st.session_state.requests[selected_idx]['Review History'].append(f"{now_str}: Classification rejected. Type manually changed from '{old_type}' to '{new_manual_type}'."); st.session_state.requests[selected_idx]['Last Update'] = now_str; st.session_state.rejecting_classification_id = None; st.success(f"Classification rejected. Permit type updated to '{new_manual_type}'."); st.rerun()
                                            with cancel_col:
                                                if st.button("Cancel Rejection", key=f"cancel_reject_{selected_id}"): st.session_state.rejecting_classification_id = None; st.rerun()
                                        elif not classification_approved:
                                            approve_col, reject_col = st.columns(2);
                                            with approve_col:
                                                if st.button("✅ Approve Classification", key=f"approve_class_{selected_id}"): now_str = datetime.now().strftime("%Y-%m-%d %H:%M"); st.session_state.requests[selected_idx]['Classification Approved'] = True; st.session_state.requests[selected_idx]['Review History'].append(f"{now_str}: AI classification manually approved."); st.session_state.requests[selected_idx]['Last Update'] = now_str; st.success("AI Classification Approved!"); st.rerun() # Rerun inside button
                                            with reject_col:
                                                if st.button("❌ Reject Classification", key=f"reject_class_{selected_id}"): st.session_state.rejecting_classification_id = selected_id; st.rerun() # Rerun inside button
                                        st.write("**Review Tracking:**"); st.write(f"Needs Review By: {', '.join(selected_data.get('Needs Review By', [])) or 'None'}"); st.write(f"External Review: {selected_data['External Review Status']}")
                                        if st.button("🔄 Update Status/Assignment", key=f"update_{selected_id}", type="primary"):
                                            now_str = datetime.now().strftime("%Y-%m-%d %H:%M"); update_log = []
                                            if new_status != current_status: st.session_state.requests[selected_idx]['Status'] = new_status; update_log.append(f"{now_str}: Status -> {new_status}.")
                                            if new_assignee != current_assignee: st.session_state.requests[selected_idx]['Assigned To'] = new_assignee; update_log.append(f"{now_str}: Reassigned -> {new_assignee}.")
                                            if update_log:
                                                st.session_state.requests[selected_idx]['Last Update'] = now_str; st.session_state.requests[selected_idx]['Review History'].extend(update_log); st.success("Request updated!"); st.rerun() # Rerun inside button
                                            else: st.info("No changes detected.")
                                        st.write("**Simulate Actions:**"); sim1, sim2 = st.columns(2);
                                        with sim1:
                                            if st.button("➡️ Send External", key=f"external_{selected_id}"): st.session_state.requests[selected_idx]['Status'] = "External Review"; st.session_state.requests[selected_idx]['External Review Status'] = "Pending"; st.session_state.requests[selected_idx]['Last Update'] = datetime.now().strftime("%Y-%m-%d %H:%M"); st.session_state.requests[selected_idx]['Review History'].append(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}: Sent External."); st.success("Sent external."); st.rerun() # Rerun inside button
                                        with sim2:
                                            if st.button("🚩 Trigger Interdept.", key=f"interdept_{selected_id}"): needed = selected_data.get('Needs Review By', []); st.info(f"Notify: {', '.join(needed)}") if needed else st.warning("No interdept needed.")
                                        st.write("**Review History:**"); st.text_area("History Log", value="\n".join(selected_data['Review History']), height=150, disabled=True, key=f"history_{selected_id}")
                                        st.divider(); st.write("**Docs (Placeholders):**"); doc1, doc2 = st.columns(2);
                                        with doc1: st.text_area("Checklist", value=generate_mock_checklist(selected_data['Permit Type']), height=150, disabled=True, key=f"checklist_{selected_id}");
                                        with doc2:
                                            if selected_data['Status'] == 'Approved': st.text_area("Issuance Doc", value=generate_mock_issuance_doc(selected_data), height=150, disabled=True, key=f"issuance_{selected_id}"); st.button("Download Doc", key=f"dl_{selected_id}")
                                            else: st.info("Issuance doc on approval.")
                            else:
                                st.warning(f"Selected ID '{selected_id}' not found.")
                        else:
                            st.info("No requests to manage in this view.")

                    # --- Department Reporting Charts (No changes needed) ---
                    st.divider(); st.subheader(f"Reporting Charts for {selected_dept}"); rep1, rep2 = st.columns(2);
                    with rep1: st.write(f"**Status Distribution ({selected_dept})**"); status_counts_dept = df_dept['Status'].value_counts().reset_index(); status_counts_dept.columns = ['Status', 'Count']; dept_status_spec = {"data": {"values": status_counts_dept.to_dict('records')},"mark": "bar","encoding": {"x": {"field": "Status", "type": "nominal", "axis": {"labelAngle": -45}},"y": {"field": "Count", "type": "quantitative"},"color": {"field": "Status","type": "nominal","scale": {"range": CUSTOM_COLORS},"legend": None},"tooltip": [{"field": "Status"}, {"field": "Count"}]},"config": {"view": {"stroke": "transparent"},"axis": {"domainWidth": 1}}}; st.vega_lite_chart(dept_status_spec, use_container_width=True)
                    with rep2: st.write(f"**Permit Type Distribution ({selected_dept})**"); type_counts_dept = df_dept['Permit Type'].value_counts().reset_index(); type_counts_dept.columns = ['Permit Type', 'Count']; dept_type_spec = {"data": {"values": type_counts_dept.to_dict('records')},"mark": "bar","encoding": {"x": {"field": "Permit Type", "type": "nominal", "axis": {"labelAngle": -45}},"y": {"field": "Count", "type": "quantitative"},"color": {"field": "Permit Type","type": "nominal","scale": {"range": CUSTOM_COLORS},"legend": None},"tooltip": [{"field": "Permit Type"}, {"field": "Count"}]},"config": {"view": {"stroke": "transparent"},"axis": {"domainWidth": 1}}}; st.vega_lite_chart(dept_type_spec, use_container_width=True)


        except Exception as e:
            st.error(f"An error occurred while rendering the dashboard: {e}")
            st.exception(e)
                        