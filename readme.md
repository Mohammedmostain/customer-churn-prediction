
# 📡 Telco Customer Retention Hub

### End-to-End Churn Prediction System

**Capstone Project | Computer Science**

**Technologies:** 🐍 Python 3.9+ | 🔴 Streamlit | 🧠 Scikit-Learn | 📊 Plotly

---

## 📋 Project Overview

This project addresses the critical issue of **Customer Churn** in the telecommunications sector. By leveraging historical customer data, I have built a machine learning pipeline that predicts the likelihood of a customer canceling their service.

The solution moves beyond static analysis to provide a deployed **"Churn Command Center"**—an interactive dashboard that empowers business stakeholders to identify high-risk customers and implement proactive retention strategies.

## 🚀 Key Features

* **Production-Ready Pipeline:** Automated data preprocessing, including handling missing values, One-Hot encoding, and robust feature scaling.
* **Champion Model:** A fine-tuned **Random Forest Classifier** selected for its balance of accuracy and ability to capture non-linear customer behaviors.
* **Interactive Dashboard:** A Streamlit-based web application featuring:
* **Single Prediction:** Instant risk scoring for individual customers.
* **Batch Processing:** Support for uploading CSV files to score thousands of customers simultaneously.
* **Risk Profiling:** Visual breakdown of risk factors (e.g., "High Risk due to Month-to-month contract").


* **Business Intelligence:** Comprehensive EDA identifying key churn drivers such as fiber optic service issues and payment method friction.

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest, Logistic Regression, Decision Trees)
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Web Framework:** Streamlit
* **Model Serialization:** Joblib

## 📂 Project Structure

```text
├── data/                      # Raw and processed datasets
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/                    # Serialized trained models
│   └── final_champion_tuned.joblib
├── notebooks/                 # Experimental notebooks for EDA & model selection
│   └── EDA_and_Model_Selection.ipynb
├── src/                       # Core source code
│   ├── __init__.py
│   ├── data_preprocessing.py  # Cleaning and feature engineering pipelines
│   ├── model_training.py      # Training logic and hyperparameter tuning
│   └── utils.py               # Utility functions (metrics, plotting)
├── app.py                     # Entry point for the Streamlit Dashboard
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

```

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Mohammedmostain/customer-churn-prediction.git
cd telco-churn-capstone

```


2. **Create a virtual environment:**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```



## 💻 Usage

### 1. Train the Model

To retrain the model on new data:

```bash
python src/model_training.py

```

*This script will process the data, train the Random Forest model, output performance metrics to the console, and save the serialized model to the `models/` directory.*

### 2. Launch the Dashboard

To start the web application:

```bash
streamlit run app.py

```

*The application will open automatically in your browser at `http://localhost:8501`.*

## 📊 Model Results

The **Random Forest Classifier** was selected as the champion model after rigorous comparison with Logistic Regression and Decision Tree baselines.

| Metric | Score | Significance |
| --- | --- | --- |
| **Accuracy** | **80.4%** | Correctly classifies 4 out of 5 customers. |
| **ROC-AUC** | **0.85** | Strong ability to rank customers by risk probability. |
| **Recall** | **0.56** | Effectiveness in identifying actual churners. |

**Top Predictors of Churn:**

1. **Contract Type:** "Month-to-month" contracts are the highest risk factor.
2. **Tenure:** New customers (0-12 months) are significantly more volatile.
3. **Internet Service:** "Fiber Optic" users show higher churn rates, indicating potential price/service dissatisfaction.
4. **Payment Method:** Electronic Check users churn more frequently than those on automatic payments.

---

*Data Source: [Telco Customer Churn - Kaggle*](https://www.kaggle.com/blastchar/telco-customer-churn)
