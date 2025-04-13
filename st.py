import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import joblib
from dotenv import load_dotenv
import os
import google.generativeai as genai
import random 
import time
from preprocessing import clean_description
from mockdata import MOCK_PERMIT_DATA
# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="PermitIQ", page_icon="logo.png")

# --- Load Environment Variables & Configure Gemini ---
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_configured = False
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_configured = True
    except Exception as e:
        st.sidebar.error(f"⚠️ Failed to configure Gemini: {e}")
else:
    st.sidebar.error("⚠️ GEMINI_API_KEY not found in environment variables.")

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
PERMIT_STATUS_OPTIONS = ["Submitted", "In Review", "Needs Info", "Approved", "Rejected", "Withdrawn", "External Review"]
CONFIDENCE_THRESHOLD = 0.93 

# --- Session State Initialization ---
if "requests" not in st.session_state or not st.session_state.requests: 
    st.session_state.requests = MOCK_PERMIT_DATA.copy() 
if "reviewer_idx" not in st.session_state:
    st.session_state.reviewer_idx = 0
if "view" not in st.session_state:
    st.session_state.view = "form"
if "ai_question" not in st.session_state:
    st.session_state.ai_question = ""
if "ai_response" not in st.session_state:
    st.session_state.ai_response = None
if "dashboard_detail" not in st.session_state:
    st.session_state.dashboard_detail = "Home"

# Add session state for rejection workflow if it doesn't exist
if 'rejecting_classification_id' not in st.session_state:
    st.session_state.rejecting_classification_id = None

# Define Logo Colors
LOGO_BLUE = "#17A9CE"
LOGO_GREEN = "#6BC856"

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

if not gemini_configured:
    st.sidebar.error("AI Features Disabled")

form_submitted_this_run = False

# ========== FORM PAGE ==========

if st.session_state.view == "form":

    st.subheader("✨ Ask Ethica AI about Procedures or Data")
    query_col, btn_col = st.columns([20, 1])

    with query_col:
        ai_question_input = st.text_input("Ask a question...", key="ai_input", placeholder="e.g., What are the requirements for an ADU conversion?", label_visibility="collapsed")
    with btn_col:
        ai_submit_button = st.button("➡️", key="ai_submit", use_container_width=True, disabled=not gemini_configured)

    if ai_submit_button and ai_question_input:
        st.session_state.ai_question = ai_question_input
        st.session_state.ai_response = None
        prompt = st.session_state.ai_question

        with st.spinner("🧠 Ethica AI is thinking..."):
            try:
                gemini_model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")
                response = gemini_model.generate_content(prompt)
                st.session_state.ai_response = response.text

            except Exception as e:
                st.error(f"🤕 Sorry, couldn't get an answer from the AI: {e}")
                st.session_state.ai_response = None 

    if st.session_state.ai_response:
        st.markdown("**EthicaAI Response:**")
        with st.expander("View AI Response", expanded=True):
            st.markdown(st.session_state.ai_response) 

    st.caption("Ethica AI can provide answers about general procedures or specific permit data if available.")
    st.divider() 

    st.title("📝 Submit a Permit Request")
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

                    st.write("�� Assigning reviewer...")
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
        st.info("📪 No permit requests submitted yet.")
    else:
        try:
            df_all = pd.DataFrame(st.session_state.requests)
            try: # Robust date parsing
                df_all['Submission Date'] = pd.to_datetime(df_all['Submission Date'], errors='coerce')
                df_all['Last Update DT'] = pd.to_datetime(df_all['Last Update'], errors='coerce')
            except Exception: pass # Ignore errors, keep as object

            if st.session_state.dashboard_detail == "Home":
                st.subheader(f"📋 Total Submitted Requests")

                st.divider()

                st.subheader("📈 KPIs & Overview")
                kp1, kp2, kp3, kp4 = st.columns(4)
                kp1.metric("Total Requests", len(df_all))

                valid_times = df_all.dropna(subset=['Submission Date', 'Last Update DT'])
                valid_times = valid_times[pd.to_datetime(valid_times['Submission Date'], errors='coerce').notna()]
                valid_times = valid_times[pd.to_datetime(valid_times['Last Update DT'], errors='coerce').notna()]
                valid_times['Submission Date'] = pd.to_datetime(valid_times['Submission Date'])
                valid_times['Last Update DT'] = pd.to_datetime(valid_times['Last Update DT'])
                avg_time = (valid_times['Last Update DT'] - valid_times['Submission Date']).mean() if not valid_times.empty else None
                kp2.metric("Avg. Time (Days)", f"{avg_time.days + avg_time.seconds/86400:.1f}" if pd.notna(avg_time) else "N/A")

                approved_count = df_all[df_all['Status'] == 'Approved'].shape[0]
                kp3.metric("Approved Permits", approved_count)
                in_progress = df_all[df_all['Status'].isin(['Submitted', 'In Review', 'Needs Info', 'External Review'])].shape[0]
                kp4.metric("Active Reviews", in_progress)

                st.divider()

                st.info(f"Displaying {len(df_all)} of {len(df_all)} total requests.")
                display_cols_home = ["ID", "Project Name", "Applicant", "Permit Type", "Status", "Assigned To", "Submission Date", "Last Update"]
                df_display_home = df_all.copy()
                if pd.api.types.is_datetime64_any_dtype(df_display_home['Submission Date']): df_display_home['Submission Date'] = df_display_home['Submission Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(df_display_home[display_cols_home], use_container_width=True, hide_index=True)

                st.divider()
                st.subheader("📊 Overall Reporting Charts")
                rep1, rep2 = st.columns(2)
                with rep1:
                    st.write("**Status Distribution (All)**")
                    status_counts = df_all['Status'].value_counts()
                    st.bar_chart(status_counts)
                with rep2:
                    st.write("**Assignments Distribution (All)**")
                    assignee_counts = df_all['Assigned To'].value_counts()
                    st.bar_chart(assignee_counts)

            else:
                selected_dept = st.session_state.dashboard_detail
                st.markdown(f"Managing requests assigned to **{selected_dept}**.")

                df_dept = df_all[df_all['Assigned To'] == selected_dept].copy()

                if df_dept.empty:
                    st.info(f"📪 No requests currently assigned to {selected_dept}.")
                else:
                    st.subheader("🔍 Filter Department Requests")
                    filter1, filter3 = st.columns(2)
                    with filter1:
                        status_filter = st.multiselect(f"Filter {selected_dept} by Status", options=PERMIT_STATUS_OPTIONS, default=[s for s in PERMIT_STATUS_OPTIONS if s not in ['Approved', 'Rejected', 'Withdrawn']], key=f"dept_status_{selected_dept}")
                    with filter3:
                        date_filter = st.date_input(f"Filter {selected_dept} by Submission Date", value=(), key=f"dept_date_{selected_dept}")

                    filtered_df_dept = df_dept.copy()
                    if status_filter:
                        filtered_df_dept = filtered_df_dept[filtered_df_dept['Status'].isin(status_filter)]
                    if len(date_filter) == 2 and filtered_df_dept['Submission Date'].notna().all(): 
                        filtered_df_dept = filtered_df_dept[(filtered_df_dept['Submission Date'].dt.date >= date_filter[0]) & (filtered_df_dept['Submission Date'].dt.date <= date_filter[1])]

                    st.subheader(f"📋 Submitted Requests for {selected_dept}")
                    if filtered_df_dept.empty and not df_dept.empty:
                        st.warning("No requests match the current filters for this department.")
                    elif filtered_df_dept.empty:
                        st.info(f"No requests match filters for {selected_dept}.")
                    else:
                        st.info(f"Displaying {len(filtered_df_dept)} of {len(df_dept)} total requests for {selected_dept}.")
                        display_cols = ["ID", "Project Name", "Applicant", "Permit Type", "Status", "Submission Date", "Last Update"]
                        filtered_df_display = filtered_df_dept.copy()
                        filtered_df_display['Submission Date'] = filtered_df_display['Submission Date'].dt.strftime('%Y-%m-%d')
                        st.dataframe(filtered_df_display[display_cols], use_container_width=True, hide_index=True)

                        st.subheader("⚙️ Manage Selected Request")
                        request_ids = filtered_df_dept['ID'].tolist()
                        if request_ids:
                            selected_id = st.selectbox(f"Select Request ID to Manage ({selected_dept})", options=request_ids, key=f"dept_select_{selected_dept}")
                            selected_indices = [i for i, req in enumerate(st.session_state.requests) if req['ID'] == selected_id]

                            if selected_indices:
                                selected_idx = selected_indices[0]
                                selected_data = st.session_state.requests[selected_idx]

                                with st.expander("View/Edit Details", expanded=True):
                                    detail1, detail2 = st.columns(2)
                                    with detail1:
                                        st.write(f"**Project:** {selected_data['Project Name']}")
                                        st.write(f"**Applicant:** {selected_data['Applicant']} ({selected_data['Email']})")
                                        st.write(f"**Location:** {selected_data['Location']}")
                                        st.write(f"**Submitted:** {selected_data['Submission Date']}")
                                        st.write(f"**Type:** {selected_data['Permit Type']}")
                                        st.markdown(f"**Description:** \n```\n{selected_data['Description']}\n```")

                                        confidence_score = selected_data.get('Classification Confidence')
                                        classification_approved = selected_data.get('Classification Approved', True)

                                        if confidence_score is not None:
                                            confidence_text = f"AI Confidence: **{confidence_score:.2f}**"
                                            if not classification_approved and confidence_score < CONFIDENCE_THRESHOLD:
                                                st.warning(f"⚠️ {confidence_text} - Requires manual review/approval.")
                                            elif not classification_approved:
                                                st.info(f"{confidence_text} - Awaiting manual approval.")
                                            else:
                                                st.success(f"✅ {confidence_text} - Approved.")
                                        else:
                                            st.caption("Classification confidence N/A.")

                                    with detail2:
                                        st.write("**Current Status & Assignment:**")
                                        current_status = selected_data['Status']
                                        current_assignee = selected_data['Assigned To']

                                        new_status = st.selectbox("Update Status", options=PERMIT_STATUS_OPTIONS, index=PERMIT_STATUS_OPTIONS.index(current_status), key=f"status_{selected_id}")
                                        new_assignee = st.selectbox("Reassign To (Department)", options=DEPARTMENTS, index=DEPARTMENTS.index(current_assignee) if current_assignee in DEPARTMENTS else 0, key=f"assignee_{selected_id}")

                                        st.write("**Classification Approval:**")
                                        classification_approved = selected_data.get('Classification Approved', True)
                                        classification_approved_status = "Approved" if classification_approved else "Needs Approval"
                                        st.write(f"Status: **{classification_approved_status}**")

                                        # --- Rejection Workflow ---
                                        if st.session_state.rejecting_classification_id == selected_id:
                                            st.markdown("**Rejecting AI Classification:**")
                                            # Find index for current type to pre-select
                                            try: current_type_index = PERMIT_TYPES.index(selected_data['Permit Type'])
                                            except ValueError: current_type_index = 0 # Default to first if current not found

                                            new_manual_type = st.selectbox(
                                                "Select Correct Permit Type*",
                                                options=PERMIT_TYPES,
                                                index=current_type_index,
                                                key=f"manual_type_select_{selected_id}"
                                            )
                                            confirm_col, cancel_col = st.columns(2)
                                            with confirm_col:
                                                if st.button("Confirm Rejection & Update Type", type="primary", key=f"confirm_reject_{selected_id}"):
                                                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                                                    old_type = selected_data['Permit Type']
                                                    st.session_state.requests[selected_idx]['Permit Type'] = new_manual_type
                                                    st.session_state.requests[selected_idx]['Classification Approved'] = True # Now manually corrected
                                                    st.session_state.requests[selected_idx]['Review History'].append(f"{now_str}: Classification rejected. Type manually changed from '{old_type}' to '{new_manual_type}'.")
                                                    st.session_state.requests[selected_idx]['Last Update'] = now_str
                                                    st.session_state.rejecting_classification_id = None # Reset state
                                                    st.success(f"Classification rejected. Permit type updated to '{new_manual_type}'.")
                                                    st.rerun()
                                            with cancel_col:
                                                if st.button("Cancel Rejection", key=f"cancel_reject_{selected_id}"):
                                                    st.session_state.rejecting_classification_id = None # Reset state
                                                    st.rerun()
                                        # --- Initial Approve/Reject Buttons ---
                                        elif not classification_approved:
                                            approve_col, reject_col = st.columns(2)
                                            with approve_col:
                                                if st.button("✅ Approve Classification", key=f"approve_class_{selected_id}"):
                                                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                                                    st.session_state.requests[selected_idx]['Classification Approved'] = True
                                                    st.session_state.requests[selected_idx]['Review History'].append(f"{now_str}: AI classification manually approved.")
                                                    st.session_state.requests[selected_idx]['Last Update'] = now_str
                                                    st.success("AI Classification Approved!")
                                                    st.rerun()
                                            with reject_col:
                                                if st.button("❌ Reject Classification", key=f"reject_class_{selected_id}"):
                                                    st.session_state.rejecting_classification_id = selected_id # Set state to start rejection
                                                    st.rerun()
                                        # --- End Classification Approval Section ---

                                        st.write("**Review Tracking:**")
                                        st.write(f"Needs Review By: {', '.join(selected_data.get('Needs Review By', [])) or 'None'}")
                                        st.write(f"External Review: {selected_data['External Review Status']}")

                                        if st.button("🔄 Update Status/Assignment", key=f"update_{selected_id}", type="primary"):
                                            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                                            update_log = []
                                            if new_status != current_status:
                                                st.session_state.requests[selected_idx]['Status'] = new_status
                                                update_log.append(f"{now_str}: Status changed to {new_status}.")
                                            if new_assignee != current_assignee:
                                                st.session_state.requests[selected_idx]['Assigned To'] = new_assignee
                                                update_log.append(f"{now_str}: Reassigned to {new_assignee}.")

                                            if update_log:
                                                st.session_state.requests[selected_idx]['Last Update'] = now_str
                                                st.session_state.requests[selected_idx]['Review History'].extend(update_log)
                                                st.success("Request updated!")
                                                st.rerun() 
                                            else:
                                                st.info("No changes detected.")

                                        st.write("**Simulate Actions:**")
                                        sim1, sim2 = st.columns(2)
                                        with sim1:
                                            if st.button("➡️ Send for External Review", key=f"external_{selected_id}"):
                                                st.session_state.requests[selected_idx]['Status'] = "External Review"
                                                st.session_state.requests[selected_idx]['External Review Status'] = "Pending"
                                                st.session_state.requests[selected_idx]['Last Update'] = datetime.now().strftime("%Y-%m-%d %H:%M")
                                                st.session_state.requests[selected_idx]['Review History'].append(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}: Sent for External Review.")
                                                st.success("Sent for external review.")
                                        with sim2:
                                            if st.button("🚩 Trigger Interdept. Review", key=f"interdept_{selected_id}"):
                                                needed = selected_data.get('Needs Review By', [])
                                                if needed:
                                                    st.info(f"Simulating notification to: {', '.join(needed)}")
                                                else: st.warning("No interdepartmental reviews required for this type.")

                                        st.write("**Review History:**")
                                        st.text_area("History Log", value="\n".join(selected_data['Review History']), height=150, disabled=True, key=f"history_{selected_id}")

                                        st.divider()
                                        st.write("**Generated Documents (Placeholders):**")
                                        doc1, doc2 = st.columns(2)
                                        with doc1:
                                            st.text_area("Permit Checklist", value=generate_mock_checklist(selected_data['Permit Type']), height=150, disabled=True, key=f"checklist_{selected_id}")
                                        with doc2:
                                            if selected_data['Status'] == 'Approved':
                                                st.text_area("Issuance Document", value=generate_mock_issuance_doc(selected_data), height=150, disabled=True, key=f"issuance_{selected_id}")
                                                if st.button("Download Issuance Doc (Mock)", key=f"dl_{selected_id}"):
                                                    st.toast("Download would start (mocked).")
                                            else: st.info("Issuance document generated upon approval.")
                            else:
                                st.warning(f"Selected ID '{selected_id}' seems to have been removed or modified. Please refresh or check filters.")
                        else:
                            st.info("No requests available to manage in this view.")

                    # --- Department-Specific Reporting Charts ---
                    st.divider()
                    st.subheader(f"📊 Reporting Charts for {selected_dept}") 
                    rep1, rep2 = st.columns(2)
                    with rep1:
                        st.write(f"**Status Distribution ({selected_dept})**")
                        status_counts_dept = df_dept['Status'].value_counts()
                        st.bar_chart(status_counts_dept)
                    with rep2:
                        st.write(f"**Permit Type Distribution ({selected_dept})**")
                        type_counts_dept = df_dept['Permit Type'].value_counts() 
                        st.bar_chart(type_counts_dept)
            

        except Exception as e:
            st.error(f"An error occurred while rendering the dashboard: {e}")
            st.exception(e) 
                        