import random
from datetime import datetime, timedelta
PERMIT_TYPES = ['unincorporated building residential',
    'road',
    'unincorporated electrical',
    'unincorporated mechanical',
    'fire',
    'unincorporated sewer']
PERMIT_STATUS_OPTIONS = ["Submitted", "In Review", "Needs Info", "Approved", "Rejected", "Withdrawn", "External Review"]
DEPARTMENTS_NEEDING_REVIEW = {
    "Building Permit": ["Fire", "Public Works"],
    "ADU Conversion": ["Regional Planning", "Public Works"],
    "Site Plan Review": ["Regional Planning"],
}
DEPARTMENTS = ["Department of Public Works", 
    "Fire Department",
    "Regional Planning",
    "Department of Environmental Health",
    "Department of Public Health",
    "Department of Parks and Recreation",]
MOCK_PERMIT_DATA = [
    {
        "ID": f"P{random.randint(10000, 99999)}", "Project Name": "Downtown Cafe Renovation", "Applicant": "Jane Doe",
        "Email": "jane.d@example.com", "Location": "456 Central Ave", "Department": "Building & Safety", "Permit Type": "Tenant Improvement",
        "Description": "Interior renovation for new cafe layout, including new partition walls and finishes.", "Submission Date": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"), "Status": "In Review",
        "Assigned To": "Department of Environmental Health", "Last Update": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d %H:%M"),
        "Needs Review By": [],
        "Review History": [f"{(datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d %H:%M')}: Submitted, assigned to the Department of Environmental Health.", f"{(datetime.now() - timedelta(days=8)).strftime('%Y-%m-%d %H:%M')}: Initial review started."],
        "External Review Status": "N/A",
        "Classification Confidence": 0.98,
        "Classification Approved": True
    },
    {
        "ID": f"P{random.randint(10000, 99999)}", "Project Name": "Smith Residence ADU", "Applicant": "John Smith",
        "Email": "j.smith@sample.net", "Location": "789 Suburb Ln", "Department": "Regional Planning", "Permit Type": "ADU Conversion",
        "Description": "Convert existing garage into accessory dwelling unit (ADU). Adding plumbing and electrical.", "Submission Date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"), "Status": "Submitted",
        "Assigned To": "Fire Department", "Last Update": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M"),
        "Needs Review By": DEPARTMENTS_NEEDING_REVIEW.get("ADU Conversion", []),
        "Review History": [f"{(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d %H:%M')}: Submitted, assigned to the Fire Department."],
        "External Review Status": "N/A",
        "Classification Confidence": 0.75,
        "Classification Approved": False
    },
    {
        "ID": f"P{random.randint(10000, 99999)}", "Project Name": "Oak Street Road Repair", "Applicant": "City Public Works",
        "Email": "pw@city.gov", "Location": "Oak Street between 1st and 3rd", "Department": "Public Works", "Permit Type": "road",
        "Description": "Asphalt repair and resurfacing for Oak Street section.", "Submission Date": (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d"), "Status": "Approved",
        "Assigned To": "Department of Environmental Health", "Last Update": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d %H:%M"),
        "Needs Review By": DEPARTMENTS_NEEDING_REVIEW.get("road", []),
        "Review History": [f"{(datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d %H:%M')}: Submitted, assigned to the Department of Environmental Health.", f"{(datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d %H:%M')}: Review Complete.", f"{(datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d %H:%M')}: Approved."],
        "External Review Status": "N/A",
        "Classification Confidence": None,
        "Classification Approved": True
    },
    {
        "ID": f"P{random.randint(10000, 99999)}", "Project Name": "Hillside Electrical Upgrade", "Applicant": "Elec Co.",
        "Email": "contact@elecco.com", "Location": "Rural Route 5", "Department": "Building & Safety", "Permit Type": "unincorporated electrical",
        "Description": "Upgrade main electrical panel and service line for residence in unincorporated area.", "Submission Date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"), "Status": "Submitted",
        "Assigned To": "Department of Parks and Recreation", "Last Update": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d %H:%M"),
        "Needs Review By": DEPARTMENTS_NEEDING_REVIEW.get("unincorporated electrical", []),
        "Review History": [f"{(datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d %H:%M')}: Submitted, assigned to the Department of Parks and Recreation."],
        "External Review Status": "N/A",
        "Classification Confidence": 0.95,
        "Classification Approved": True
    }
]

for _ in range(20):
    permit_type = random.choice(PERMIT_TYPES)
    assigned_dept = random.choice(DEPARTMENTS)
    submission_date = datetime.now() - timedelta(days=random.randint(1, 30))
    last_update = submission_date + timedelta(days=random.randint(1, 5))

    MOCK_PERMIT_DATA.append({
        "ID": f"P{random.randint(10000, 99999)}",
        "Project Name": f"{random.choice(['Green', 'Sunset', 'Maple', 'Ocean'])} {random.choice(['Plaza', 'Residences', 'Renovation', 'Expansion'])}",
        "Applicant": f"{random.choice(['Jane Doe', 'John Smith', 'Alex Kim', 'Maria Garcia'])}",
        "Email": f"{random.choice(['jane', 'john', 'alex', 'maria'])}@example.com",
        "Location": f"{random.randint(100, 999)} {random.choice(['Main St', 'Broadway', 'Hill Rd', 'Sunset Blvd'])}",
        "Department": assigned_dept.split("–")[0].strip(),
        "Permit Type": permit_type,
        "Description": f"{permit_type.title()} permit for {random.choice(['new construction', 'remodel', 'system upgrade'])}.",
        "Submission Date": submission_date.strftime("%Y-%m-%d"),
        "Status": random.choice(PERMIT_STATUS_OPTIONS),
        "Assigned To": assigned_dept,
        "Last Update": last_update.strftime("%Y-%m-%d %H:%M"),
        "Needs Review By": [],
        "Review History": [
            f"{submission_date.strftime('%Y-%m-%d %H:%M')}: Submitted to {assigned_dept}.",
            f"{last_update.strftime('%Y-%m-%d %H:%M')}: Review update."
        ],
        "External Review Status": "N/A",
        "Classification Confidence": round(random.uniform(0.7, 0.99), 2),
        "Classification Approved": random.choice([True, False])
    })