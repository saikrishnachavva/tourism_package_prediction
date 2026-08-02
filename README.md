# ✈️ Wellness Tourism Package — Purchase Prediction

An end-to-end **MLOps pipeline** that predicts whether a customer will purchase the newly introduced **Wellness Tourism Package** offered by *Visit with Us*, a leading travel company.

The project automates data validation, preprocessing, model training with hyperparameter tuning, and deployment using **GitHub Actions CI/CD** and **Streamlit Community Cloud**.

---

## 📌 Business Context

*Visit with Us* is leveraging data-driven strategies to optimise marketing for its new Wellness Tourism Package. Manually identifying potential buyers is inconsistent, time-consuming, and error-prone. This MLOps pipeline replaces that manual process with a scalable, automated system that:

- Ingests and validates customer data
- Preprocesses and splits the dataset
- Trains and tunes multiple classification models
- Deploys the best model behind a live web app

---

## 🎯 Objective

Build a **classification model** that predicts whether a customer will purchase the Wellness Tourism Package (`ProdTaken = 1`) *before* the sales team contacts them, and wrap it in a fully automated CI/CD pipeline on GitHub.

---

## 📂 Repository Structure

```
tourism_package_prediction/
├── data/
│   └── tourism.csv                  # Raw dataset
├── model_building/
│   ├── data_register.py             # Dataset validation & summary
│   ├── prep.py                      # Data cleaning, encoding & train-test split
│   └── train.py                     # Model tuning, evaluation & saving
├── deployment/
│   ├── app.py                       # Streamlit web application
│   └── requirements.txt             # App dependencies (Streamlit Cloud)
├── model/
│   └── best_tourism_model.joblib    # Trained model (committed by the pipeline)
└── requirements.txt                 # Pipeline dependencies (GitHub Actions)

.github/
└── workflows/
    └── pipeline.yml                 # GitHub Actions CI/CD workflow
```

---

## 📊 Data Dictionary

### Customer Details

| Feature | Description |
|---|---|
| **CustomerID** | Unique identifier for each customer |
| **ProdTaken** | Target variable — purchased a package (0: No, 1: Yes) |
| **Age** | Age of the customer |
| **TypeofContact** | Contact method (Company Invited / Self Enquiry) |
| **CityTier** | City category (Tier 1 > Tier 2 > Tier 3) |
| **Occupation** | e.g. Salaried, Freelancer, Small Business, Large Business |
| **Gender** | Male / Female |
| **NumberOfPersonVisiting** | People accompanying the customer |
| **PreferredPropertyStar** | Preferred hotel star rating |
| **MaritalStatus** | Single, Married, Divorced |
| **NumberOfTrips** | Average annual trips |
| **Passport** | Holds a valid passport (0: No, 1: Yes) |
| **OwnCar** | Owns a car (0: No, 1: Yes) |
| **NumberOfChildrenVisiting** | Children below age 5 accompanying |
| **Designation** | Designation in current organisation |
| **MonthlyIncome** | Gross monthly income |

### Customer Interaction Data

| Feature | Description |
|---|---|
| **PitchSatisfactionScore** | Satisfaction with the sales pitch (1–5) |
| **ProductPitched** | Product type pitched (Basic, Standard, Deluxe, Super Deluxe, King) |
| **NumberOfFollowups** | Follow-ups after the pitch |
| **DurationOfPitch** | Duration of the sales pitch (minutes) |

---

## ⚙️ CI/CD Pipeline (GitHub Actions)

The workflow (`.github/workflows/pipeline.yml`) triggers on every push to `main` and runs three sequential jobs:

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│   register-dataset   │───▶ |      data-prep       │────▶│   model-training     │
│   Validate CSV       │     │   Clean & Split      │     │   Tune & Evaluate    │
│   Print summary      │     │   Upload artifacts   │     │   Commit best model  │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

| Job | What it does |
|---|---|
| **register-dataset** | Checks all expected columns are present; prints a dataset summary |
| **data-prep** | Drops unnecessary columns, handles missing values, encodes categoricals, performs stratified 80/20 train-test split, uploads CSV artifacts |
| **model-training** | Downloads split artifacts, tunes Decision Tree, Bagging, Random Forest, AdaBoost & Gradient Boosting via GridSearchCV, evaluates the best model, commits the `.joblib` file back to the repo |

---

## 🧠 Models Evaluated

| Model | Tuning Method |
|---|---|
| Decision Tree | GridSearchCV (max_depth, min_samples_split, criterion) |
| Bagging | GridSearchCV (n_estimators, max_samples) |
| Random Forest | GridSearchCV (n_estimators, max_depth, min_samples_split) |
| AdaBoost | GridSearchCV (n_estimators, learning_rate) |
| Gradient Boosting | GridSearchCV (n_estimators, max_depth, learning_rate, subsample) |

The model with the **highest cross-validated accuracy** is selected, evaluated on the test set, and saved.

### Evaluation Metrics

- Accuracy (Train & Test)
- Precision, Recall, F1-Score
- AUC-ROC
- Full Classification Report

---

## 🚀 Streamlit App

The live app is deployed on **Streamlit Community Cloud** and lets users enter customer details to get a real-time purchase prediction.

### Features

- 👤 **Customer Details** — Age, Gender, Occupation, Marital Status, Designation, Monthly Income
- 🌍 **Travel Details** — Contact Type, City Tier, Persons Visiting, Children, Trips, Passport, Car
- 📞 **Interaction Data** — Product Pitched, Property Star, Pitch Duration, Follow-ups, Satisfaction Score
- 🔮 **Prediction** — Purchase likelihood with probability score

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9 (pipeline) / 3.11 (Streamlit) |
| ML Libraries | scikit-learn, joblib |
| Data | pandas |
| CI/CD | GitHub Actions |
| Deployment | Streamlit Community Cloud |
| Web App | Streamlit |

---

## 📦 Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/tourism-package-prediction.git
cd tourism-package-prediction
```

### 2. Install dependencies

```bash
pip install -r tourism_package_prediction/requirements.txt
```

### 3. Run the pipeline locally (optional)

```bash
# Step 1: Validate data
python tourism_package_prediction/model_building/data_register.py

# Step 2: Prepare data
python tourism_package_prediction/model_building/prep.py

# Step 3: Train model
python tourism_package_prediction/model_building/train.py
```

### 4. Run the Streamlit app locally

```bash
pip install -r tourism_package_prediction/deployment/requirements.txt
streamlit run tourism_package_prediction/deployment/app.py
```
