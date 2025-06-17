import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# ---------------------------
# Load model, pipeline, and columns
# ---------------------------

model = xgb.XGBRegressor()
model.load_model("salary_predictor_final.json")

pipeline = joblib.load("preprocessing_pipeline_final.pkl")
model_columns = joblib.load("model_columns.pkl")

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("Glassdoor Salary Prediction 🚀")

# Preset dropdown options
job_title_options = ["Data Scientist", "Software Engineer", "Machine Learning Engineer", "Business Analyst"]
company_name_options = ["Google", "Amazon", "Microsoft", "Facebook"]
location_options = ["San Francisco, CA", "Seattle, WA", "New York, NY", "Austin, TX"]
headquarters_options = ["Mountain View, CA", "Redmond, WA", "Seattle, WA", "Menlo Park, CA"]
ownership_options = ["Company - Public", "Company - Private", "Subsidiary or Business Segment", "Nonprofit Organization"]
industry_options = ["Internet", "Software Development", "Healthcare", "Finance"]
sector_options = ["Information Technology", "Financial Services", "Healthcare", "Retail & Wholesale"]
revenue_options = [
    "Less than $1 million", "$1 to $5 million", "$5 to $10 million", "$10 to $50 million",
    "$50 to $100 million", "$100 million to $500 million", "$500 million to $1 billion",
    "$1 to $5 billion", "$5 to $10 billion", "$10+ billion (USD)"
]

# ---------------------------
# User Inputs

job_title = st.selectbox("Job Title", job_title_options)
company_name = st.selectbox("Company Name", company_name_options)
location = st.selectbox("Location", location_options)
headquarters = st.selectbox("Headquarters", headquarters_options)
ownership = st.selectbox("Type of Ownership", ownership_options)
industry = st.selectbox("Industry", industry_options)
sector = st.selectbox("Sector", sector_options)
revenue = st.selectbox("Revenue", revenue_options)

rating = st.number_input("Company Rating", min_value=0.0, max_value=5.0, value=4.0)
founded = st.number_input("Founded Year", min_value=1800, max_value=2024, value=2000)
competitor_count = st.number_input("Competitor Count", min_value=0, value=0)
size_num = st.number_input("Company Size (approx. employees)", min_value=1, value=1000)

min_salary = st.slider("Minimum Salary (K)", 0, 500, 80)
max_salary = st.slider("Maximum Salary (K)", min_salary, 1000, 120)

# Skills selection
skill_options = [
    "Python", "SQL", "AWS", "Azure", "Spark", "Hadoop",
    "Excel", "Tableau", "SAS", "Docker", "Kubernetes", "Security Clearance", "R"
]
selected_skills = st.multiselect("Select Relevant Skills:", skill_options)

# Education
phd_yn = st.checkbox("PhD Degree", value=False)
masters_yn = st.checkbox("Masters Degree", value=False)
bachelors_yn = st.checkbox("Bachelors Degree", value=True)

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------

if st.button("Predict Salary"):

    company_age = 2024 - founded

    try:
        job_state = location.split(",")[1].strip()
    except:
        job_state = "Unknown"

    unnamed_val = 0

    input_data = {
        'Unnamed: 0': 0,
        'Job Title': job_title,
        'Rating': rating,
        'Company Name': company_name,
        'Location': location,
        'Headquarters': headquarters,
        'Founded': founded,
        'Type of ownership': ownership,
        'Industry': industry,
        'Sector': sector,
        'Revenue': revenue,
        'competitor_count': competitor_count,
        'company_age': company_age,
        'job_state': job_state,
        'min_salary': min_salary,
        'max_salary': max_salary,
        'python_yn': 1 if "Python" in selected_skills else 0,
        'r_yn': 1 if "R" in selected_skills else 0,
        'sql_yn': 1 if "SQL" in selected_skills else 0,
        'aws_yn': 1 if "AWS" in selected_skills else 0,
        'azure_yn': 1 if "Azure" in selected_skills else 0,
        'spark_yn': 1 if "Spark" in selected_skills else 0,
        'hadoop_yn': 1 if "Hadoop" in selected_skills else 0,
        'excel_yn': 1 if "Excel" in selected_skills else 0,
        'tableau_yn': 1 if "Tableau" in selected_skills else 0,
        'sas_yn': 1 if "SAS" in selected_skills else 0,
        'docker_yn': 1 if "Docker" in selected_skills else 0,
        'kubernetes_yn': 1 if "Kubernetes" in selected_skills else 0,
        'security_clearance': 1 if "Security Clearance" in selected_skills else 0,
        'phd_yn': int(phd_yn),
        'masters_yn': int(masters_yn),
        'bachelors_yn': int(bachelors_yn),
        'size_num': size_num
    }

    input_df = pd.DataFrame([input_data])

    # Ensure column order matches model training
    input_df = input_df[model_columns]

    # Apply preprocessing pipeline
    processed_input = pipeline.transform(input_df)

    # Convert to dense if pipeline returned sparse matrix
    if hasattr(processed_input, 'toarray'):
        processed_input = processed_input.toarray()

    predicted_salary = model.predict(processed_input)[0]

    st.success(f"Predicted Salary: ${predicted_salary:.2f}K")
