# 🚀 Glassdoor Salary Prediction 

## 📊 Project Overview

This project aims to build a production-ready salary prediction system using real-world Glassdoor job data. The machine learning model predicts the average salary based on job description, company information, location, skills, and other job-related features.

The complete ML pipeline includes:

- End-to-end data preprocessing
- OneHotEncoding for stable categorical feature handling
- Feature engineering
- Model training using XGBoost
- Fully synchronized pipeline and model deployment
- Streamlit-powered web application

## ⚙️ Tech Stack

**ML Algorithm**: XGBoost  
**Preprocessing**: Scikit-Learn Pipeline  
**Encoding**: OneHotEncoder (handle_unknown='ignore')  
**Model Deployment**: Streamlit  
**Language**: Python 3.12  
**Model Serialization**: joblib, JSON  
**Frontend**: Streamlit UI  
**Backend**: Python  
**Deployment Platform**: Streamlit Cloud / GitHub

## 🏗 Project Structure

- app.py - Streamlit app code (frontend + inference)
- preprocessing_pipeline_final.pkl - Final preprocessing pipeline (OneHotEncoder)
- salary_predictor_final.json - Final trained XGBoost model
- model_columns.pkl - Column order for consistent pipeline transformation
- requirements.txt - Full dependencies for deployment
- README.md - This file

## 📦 Dataset Overview

We used a cleaned Glassdoor job dataset (`final_X_for_pipeline2.csv`) which includes:

- Company Information (name, ownership, sector, revenue, etc.)
- Location and Headquarters
- Salary Estimates (`min_salary`, `max_salary`)
- Job Title
- Job Description Features (skills, education, certifications)
- Engineered features: `company_age`, `competitor_count`, `job_state`
- Target: `avg_salary`

## 🚀 Model Development Workflow

### 1️⃣ Preprocessing

- OneHotEncoder applied to all categorical columns.
- StandardScaler applied to numerical features.
- Binary feature extraction (skills and education).
- Pipeline was fitted once and saved (`preprocessing_pipeline_final.pkl`).

### 2️⃣ Model Training

- XGBoost Regressor trained on pipeline-transformed data.
- Final model saved as `salary_predictor_final.json`.

### 3️⃣ Deployment

- Streamlit handles:
  - User input
  - Automatic pipeline transformation
  - Real-time salary prediction

## 📊 Features Used

### Categorical Columns:

- Job Title
- Company Name
- Location
- Headquarters
- Type of ownership
- Industry
- Sector
- Revenue
- job_state

### Numerical Columns:

- Unnamed: 0
- Rating
- Founded
- competitor_count
- company_age
- min_salary
- max_salary
- size_num

### Binary Skill Columns:

- python_yn
- r_yn
- sql_yn
- aws_yn
- azure_yn
- spark_yn
- hadoop_yn
- excel_yn
- tableau_yn
- sas_yn
- docker_yn
- kubernetes_yn
- security_clearance

### Binary Education Columns:

- phd_yn
- masters_yn
- bachelors_yn

## 📂 Deployment Instructions

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App Locally

```bash
streamlit run app.py
```
### Deploy to Streamlit Cloud
- Push your repository (including model & pipeline files) to GitHub.
- Connect GitHub repo to Streamlit Cloud.
- Set app.py as the main entry file.
- Done
---
## 🏆 Key Takeaways
- Fully production-grade ML pipeline.
- Completely avoids feature shape mismatches.
- Stable categorical encoding using OneHotEncoder.
- Seamless integration between model training and live prediction.
---
## Author
### **SIDDHESH BHURKE**
#### https://www.linkedin.com/in/siddheshbhurke/["https://www.linkedin.com/in/siddheshbhurke/"]
