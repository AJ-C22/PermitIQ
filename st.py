import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import joblib
from dotenv import load_dotenv
import os
import google.generativeai as genai
import random 
import time

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
REVIEWERS = ["Alice", "Bob", "Charlie", "Diana"]
DEPARTMENTS_NEEDING_REVIEW = {
    "Building Permit": ["Fire", "Public Works"],
    "ADU Conversion": ["Regional Planning", "Public Works"],
    "Site Plan Review": ["Regional Planning"],
}
PERMIT_STATUS_OPTIONS = ["Submitted", "In Review", "Needs Info", "Approved", "Rejected", "Withdrawn", "External Review"]

MOCK_PERMIT_DATA = [
    {
        "ID": f"P{random.randint(10000, 99999)}", "Project Name": "Downtown Cafe Renovation", "Applicant": "Jane Doe",
        "Email": "jane.d@example.com", "Location": "456 Central Ave", "Department": "Building & Safety", "Permit Type": "Tenant Improvement",
        "Description": "Interior renovation for new cafe layout, including new partition walls and finishes.", "Submission Date": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"), "Status": "In Review",
        "Assigned To": "Alice", "Last Update": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d %H:%M"),
        "Needs Review By": [],
        "Review History": [f"{(datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d %H:%M')}: Submitted, assigned to Alice.", f"{(datetime.now() - timedelta(days=8)).strftime('%Y-%m-%d %H:%M')}: Initial review started."],
        "External Review Status": "N/A"
    },
    {
        "ID": f"P{random.randint(10000, 99999)}", "Project Name": "Smith Residence ADU", "Applicant": "John Smith",
        "Email": "j.smith@sample.net", "Location": "789 Suburb Ln", "Department": "Regional Planning", "Permit Type": "ADU Conversion",
        "Description": "Convert existing garage into accessory dwelling unit (ADU). Adding plumbing and electrical.", "Submission Date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"), "Status": "Submitted",
        "Assigned To": "Bob", "Last Update": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M"),
        "Needs Review By": DEPARTMENTS_NEEDING_REVIEW.get("ADU Conversion", []),
        "Review History": [f"{(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d %H:%M')}: Submitted, assigned to Bob."],
        "External Review Status": "N/A"
    },
    {
        "ID": f"P{random.randint(10000, 99999)}", "Project Name": "Oak Street Road Repair", "Applicant": "City Public Works",
        "Email": "pw@city.gov", "Location": "Oak Street between 1st and 3rd", "Department": "Public Works", "Permit Type": "road",
        "Description": "Asphalt repair and resurfacing for Oak Street section.", "Submission Date": (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d"), "Status": "Approved",
        "Assigned To": "Charlie", "Last Update": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d %H:%M"),
        "Needs Review By": DEPARTMENTS_NEEDING_REVIEW.get("road", []),
        "Review History": [f"{(datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d %H:%M')}: Submitted, assigned to Charlie.", f"{(datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d %H:%M')}: Review Complete.", f"{(datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d %H:%M')}: Approved."],
        "External Review Status": "N/A"
    },
        {
        "ID": f"P{random.randint(10000, 99999)}", "Project Name": "Hillside Electrical Upgrade", "Applicant": "Elec Co.",
        "Email": "contact@elecco.com", "Location": "Rural Route 5", "Department": "Building & Safety", "Permit Type": "unincorporated electrical",
        "Description": "Upgrade main electrical panel and service line for residence in unincorporated area.", "Submission Date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"), "Status": "Submitted",
        "Assigned To": "Diana", "Last Update": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d %H:%M"),
        "Needs Review By": DEPARTMENTS_NEEDING_REVIEW.get("unincorporated electrical", []),
        "Review History": [f"{(datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d %H:%M')}: Submitted, assigned to Diana."],
        "External Review Status": "N/A"
    }
]

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

# Define Logo Colors
LOGO_BLUE = "#17A9CE"
LOGO_GREEN = "#6BC856"

# --- Helper Functions ---
def get_next_reviewer():
    """Simple round-robin assignment."""
    reviewer = REVIEWERS[st.session_state.reviewer_idx % len(REVIEWERS)]
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
        <h1 style='text-align: center; font-size: 2.4em; margin-bottom: 0px; margin-top: 0px;'>
            <span style='color:{LOGO_BLUE};'>PERMIT</span><span style='color:{LOGO_GREEN};'>IQ</span>
        </h1>
        """,
        unsafe_allow_html=True
    )
st.sidebar.divider() 

selected_view = st.sidebar.radio(
    "Navigation",
    ["📝 Submit New Request", "📊 Internal Dashboard"],
    key="nav_radio",
    captions=["Fill out a permit application.", "Track and manage requests."]
)
st.session_state.view = "form" if selected_view == "📝 Submit New Request" else "dashboard"
st.sidebar.divider()
st.sidebar.info(f"👷 Mock Reviewers: {', '.join(REVIEWERS)}")

form_submitted_this_run = False

# ========== FORM PAGE ==========


if st.session_state.view == "form":

    st.subheader("✨ Ask Ethica AI about Procedures or Data")
    query_col, btn_col = st.columns([20, 1])

    with query_col:
        ai_question_input = st.text_input("Ask a question...", key="ai_input", placeholder="e.g., What are the requirements for an ADU conversion?", label_visibility="collapsed")
    with btn_col:
        # Disable button if Gemini isn't configured
        ai_submit_button = st.button("➡️", key="ai_submit", use_container_width=True, disabled=not gemini_configured)

    if ai_submit_button and ai_question_input:
        st.session_state.ai_question = ai_question_input
        st.session_state.ai_response = None # Reset previous response
        prompt = st.session_state.ai_question

        with st.spinner("🧠 Ethica AI is thinking..."):
            try:
                gemini_model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")
                response = gemini_model.generate_content(prompt)

                # --- Store the response text ---
                st.session_state.ai_response = response.text

            except Exception as e:
                st.error(f"🤕 Sorry, couldn't get an answer from the AI: {e}")
                st.session_state.ai_response = None # Ensure response is None on error

    # Display AI response if available
    if st.session_state.ai_response:
        st.markdown("**EthicaAI Response:**")
        # Use an expander for potentially long responses
        with st.expander("View AI Response", expanded=True):
            st.markdown(st.session_state.ai_response) # Display the actual response

    st.caption("Ethica AI can provide answers about general procedures or specific permit data if available.")
    st.divider() # Separator before the form

    st.title("📝 Submit a Permit Request")
    st.markdown("Fill in the details below to submit your permit application.")

    # --- Form definition ---
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
            department = st.selectbox("Primary Department*", ["Building & Safety", "Fire", "Public Works", "Regional Planning"], key="department")
        with col4:
            user_permit_type = st.selectbox("Requested Permit Type*", [
                'unincorporated building residential',
                'road',
                'unincorporated electrical',
                'unincorporated mechanical'
            ], key="user_permit_type")

        description = st.text_area("Detailed Project Description*", height=150, key="description", placeholder="Describe the scope of work...")
        submission_date = datetime.today()

        # Form submit button
        submitted = st.form_submit_button("➡️ Submit Permit Request", use_container_width=True, type="primary")

        # --- Form Submission Logic ---
        if submitted:
            form_submitted_this_run = True # Set the flag
            # Validation
            if not all([applicant_name, project_name, email, project_location, description]):
                st.error("⚠️ Please fill in all required fields marked with *.")
                form_submitted_this_run = False # Reset flag if validation fails
            else:
                # Use st.status for processing feedback
                with st.status("Processing submission...", expanded=True) as status:
                    st.write("⏳ Validating input...")
                    time.sleep(0.5)

                    # Auto-Classification
                    predicted_type = user_permit_type
                    if model and description:
                        try:
                            st.write("🤖 Auto-classifying...")
                            model_prediction = model.predict([description])[0]
                            if hasattr(model, "predict_proba"):
                                proba = model.predict_proba([description])
                                confidence = proba.max()
                                if confidence > 0.6 and model_prediction != user_permit_type:
                                    predicted_type = model_prediction
                                    st.info(f"🤖 System classified as: **{predicted_type}** (Confidence: {confidence:.2f})")
                                else: predicted_type = user_permit_type
                            else: # No predict_proba
                                if model_prediction != user_permit_type: predicted_type = model_prediction
                                else: predicted_type = user_permit_type
                            time.sleep(0.5)
                        except Exception as e: st.warning(f"⚠️ Classification failed: {e}. Using user type.")
                    else: predicted_type = user_permit_type

                    # Routing
                    st.write("👷 Assigning reviewer...")
                    assigned_reviewer = get_next_reviewer()
                    time.sleep(0.5)

                    # Prepare data
                    new_entry = {
                        "ID": f"P{random.randint(10000, 99999)}", "Project Name": project_name, "Applicant": applicant_name,
                        "Email": email, "Location": project_location, "Department": department, "Permit Type": predicted_type,
                        "Description": description, "Submission Date": submission_date.strftime("%Y-%m-%d"), "Status": "Submitted",
                        "Assigned To": assigned_reviewer, "Last Update": submission_date.strftime("%Y-%m-%d %H:%M"),
                        "Needs Review By": DEPARTMENTS_NEEDING_REVIEW.get(predicted_type, []),
                        "Review History": [f"{submission_date.strftime('%Y-%m-%d %H:%M')}: Submitted, assigned to {assigned_reviewer}."],
                        "External Review Status": "N/A"
                    }

                    # Save data (append to session state list)
                    st.write("💾 Saving request...")
                    st.session_state.requests.append(new_entry)
                    time.sleep(0.5)

                    # Update status indicator
                    status.update(label="✅ Submission Complete!", state="complete", expanded=False)

    # --- Display Success Message outside the form ---
    if form_submitted_this_run:
         # This message appears *after* the form block if submission was successful in this run
         st.success(f"✅ Permit request '{st.session_state.requests[-1]['Project Name']}' submitted successfully! Assigned to {st.session_state.requests[-1]['Assigned To']}.")
         st.balloons()
         # We don't need to clear form here because clear_on_submit=True handles it.

# ========== DASHBOARD PAGE ==========
# Ensure this elif is OUTSIDE the "form" block
elif st.session_state.view == "dashboard":
    st.title("📊 Internal Dashboard")
    st.markdown("Track, manage, and review submitted permit requests.")

    # Check if there are requests AFTER potentially adding one in the form view run
    if not st.session_state.requests:
        st.info("📪 No permit requests submitted yet.")
    else:
        # --- Dashboard content (KPIs, Filtering, Table, Details) ---
        try: # Wrap dashboard logic in try/except for better error isolation
            df = pd.DataFrame(st.session_state.requests)
            # Convert dates safely, coercing errors
            df['Submission Date'] = pd.to_datetime(df['Submission Date'], errors='coerce')
            df['Last Update DT'] = pd.to_datetime(df['Last Update'], errors='coerce')

            # --- KPIs & Overview ---
            st.subheader("📈 KPIs & Overview")
            kp1, kp2, kp3, kp4 = st.columns(4)
            kp1.metric("Total Requests", len(df))
            # Calculate average time safely
            valid_times = df.dropna(subset=['Submission Date', 'Last Update DT'])
            avg_time = (valid_times['Last Update DT'] - valid_times['Submission Date']).mean() if not valid_times.empty else None
            kp2.metric("Avg. Time (Days)", f"{avg_time.days + avg_time.seconds/86400:.1f}" if pd.notna(avg_time) else "N/A")

            approved_count = df[df['Status'] == 'Approved'].shape[0]
            kp3.metric("Approved Permits", approved_count)
            in_progress = df[df['Status'].isin(['Submitted', 'In Review', 'Needs Info', 'External Review'])].shape[0]
            kp4.metric("Active Reviews", in_progress)

            # --- Filtering ---
            st.subheader("🔍 Filter Requests")
            filter1, filter2, filter3 = st.columns(3)
            with filter1:
                status_filter = st.multiselect("Filter by Status", options=PERMIT_STATUS_OPTIONS, default=[s for s in PERMIT_STATUS_OPTIONS if s not in ['Approved', 'Rejected', 'Withdrawn']])
            with filter2:
                reviewer_filter = st.multiselect("Filter by Assignee", options=['Unassigned'] + REVIEWERS, default=['Unassigned'] + REVIEWERS)
            with filter3:
                date_filter = st.date_input("Filter by Submission Date (Range)", value=(), key="date_filter") # Empty tuple for range

            # Apply filters
            filtered_df = df.copy()
            if status_filter:
                filtered_df = filtered_df[filtered_df['Status'].isin(status_filter)]
            if reviewer_filter:
                filtered_df = filtered_df[filtered_df['Assigned To'].isin(reviewer_filter)]
            if len(date_filter) == 2 and filtered_df['Submission Date'].notna().all(): # Check for NaT after conversion
                 filtered_df = filtered_df[(filtered_df['Submission Date'].dt.date >= date_filter[0]) & (filtered_df['Submission Date'].dt.date <= date_filter[1])]

            # --- Permit Request Table ---
            st.subheader("📋 Submitted Permit Requests")
            if filtered_df.empty and not df.empty:
                 st.warning("No requests match the current filters.")
            elif filtered_df.empty and df.empty:
                 st.info("No requests available.") # Should be caught earlier, but safe check
            else:
                 st.info(f"Displaying {len(filtered_df)} of {len(df)} total requests.")
                 display_cols = ["ID", "Project Name", "Applicant", "Permit Type", "Status", "Assigned To", "Submission Date", "Last Update"]
                 # Format Submission Date for display if needed
                 filtered_df_display = filtered_df.copy()
                 filtered_df_display['Submission Date'] = filtered_df_display['Submission Date'].dt.strftime('%Y-%m-%d')
                 st.dataframe(filtered_df_display[display_cols], use_container_width=True, hide_index=True)

                 # --- Detailed View & Actions ---
                 st.subheader("⚙️ Manage Selected Request")
                 request_ids = filtered_df['ID'].tolist()
                 selected_id = st.selectbox("Select Request ID to Manage", options=request_ids)
                 # Find original index safely
                 selected_indices = df.index[df['ID'] == selected_id].tolist()

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

                         with detail2:
                             st.write("**Current Status & Assignment:**")
                             current_status = selected_data['Status']
                             current_assignee = selected_data['Assigned To']

                             new_status = st.selectbox("Update Status", options=PERMIT_STATUS_OPTIONS, index=PERMIT_STATUS_OPTIONS.index(current_status), key=f"status_{selected_id}")
                             new_assignee = st.selectbox("Reassign To", options=REVIEWERS, index=REVIEWERS.index(current_assignee) if current_assignee in REVIEWERS else 0, key=f"assignee_{selected_id}")

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
                                      st.rerun() # Rerun to reflect changes
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
                                      st.rerun() # Corrected indentation
                             with sim2:
                                  if st.button("🚩 Trigger Interdept. Review", key=f"interdept_{selected_id}"):
                                       needed = st.session_state.requests[selected_idx]['Needs Review By']
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
                      st.warning(f"Selected ID '{selected_id}' not found in the current data. It might have been filtered out or there's an issue.")


                 # --- Reporting Charts ---
                 st.divider()
                 st.subheader("📊 Reporting Charts")
                 rep1, rep2 = st.columns(2)
                 with rep1:
                     st.write("**Status Distribution**")
                     # Use original df for overall counts
                     status_counts = pd.DataFrame(st.session_state.requests)['Status'].value_counts()
                     st.bar_chart(status_counts)
                 with rep2:
                     st.write("**Assignments Distribution**")
                     assignee_counts = pd.DataFrame(st.session_state.requests)['Assigned To'].value_counts()
                     st.bar_chart(assignee_counts)

        except Exception as e:
             st.error(f"An error occurred while rendering the dashboard: {e}")
             st.exception(e) # Show traceback for debugging
                        