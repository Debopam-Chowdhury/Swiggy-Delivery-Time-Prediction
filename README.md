# 🛵 Swiggy Delivery Time Prediction ⏱️

Predicting the delivery time from a restaurant to the customer using real-world features like weather, traffic, vehicle condition, order type, etc. This project combines **data science**, **machine learning**, and **MLOps** to solve a practical last-mile delivery problem.

![ML Pipeline](https://img.shields.io/badge/MLOps-DVC%20%7C%20MLflow%20%7C%20Dagshub-blue)
![Deployment](https://img.shields.io/badge/Deployed-Render.com-brightgreen)
![Model](https://img.shields.io/badge/Model-Stacking%20Regressor-yellow)
![Status](https://img.shields.io/badge/Project%20Status-Deployed%20%26%20Functional-success)

🌐 **Live Demo**: [https://swiggy-delivery-time-prediction.onrender.com](https://swiggy-delivery-time-prediction.onrender.com)

---

## 📌 Table of Contents

- [📊 Problem Statement](#-problem-statement)
- [📦 Dataset Overview](#-dataset-overview)
- [🧼 Data Cleaning & Feature Engineering](#-data-cleaning--feature-engineering)
- [📈 EDA & Statistical Analysis](#-eda--statistical-analysis)
- [⚙️ Preprocessing Pipeline](#️-preprocessing-pipeline)
- [🧪 Experimentation](#-experimentation)
- [🚀 Model Building & Evaluation](#-model-building--evaluation)
- [🛠 MLOps: MLflow + DVC + Dagshub](#-mlops-mlflow--dvc--dagshub)
- [🌐 Web App Deployment](#-web-app-deployment)
- [📁 Project Structure](#-project-structure)

---

## 📊 Problem Statement

Predict the **delivery time** (in minutes) taken by a Swiggy delivery person for each order using historical order and delivery data. This helps Swiggy:
- Improve delivery estimates.
- Optimize route and logistics planning.
- Enhance customer satisfaction.

---

## 📦 Dataset Overview

The dataset includes delivery metadata like rider profile, time of order, traffic/weather conditions, and more.

| Column                       | Description                                  |
|-----------------------------|----------------------------------------------|
| Delivery_person_ID          | Unique ID of delivery agent                  |
| Delivery_person_Age         | Age of the delivery person                   |
| Delivery_person_Ratings     | Average ratings given to the delivery agent  |
| Restaurant_latitude/longitude | Location of the restaurant               |
| Delivery_location_lat/long  | Delivery location                            |
| Order_Date, Time_Ordered    | Date & time order was placed                 |
| Time_Order_picked           | When order was picked up                     |
| Weatherconditions           | Weather at the time of delivery              |
| Road_traffic_density        | Traffic level during delivery                |
| Vehicle_condition           | Rating of vehicle                            |
| Type_of_order/vehicle       | Type of food and vehicle used                |
| multiple_deliveries         | Indicates batch deliveries                   |
| Festival, City              | External effects on delivery                 |
| Time_taken(min)             | Target column                                |

🧾 **Note:** Many of the columns were raw or incomplete and needed cleaning or transformation.

---

## 🧼 Data Cleaning & Feature Engineering

Performed robust cleaning steps:
- Converted all time columns to datetime.
- Extracted useful components like `Order Hour`, `Order Weekday`, `Delivery Duration`.
- Cleaned `multiple_deliveries`, `ratings`, and encoded festivals.
- Removed outliers, imputed/dropped missing values after experimentation.

✅ Created new features such as:
- `Delivery Distance` using haversine formula.
- `Delivery Duration` (from pickup to delivery).
- `Order-Pickup Gap` (time from placing to pickup).

---

## 📈 EDA & Statistical Analysis

Visual and statistical exploration of key insights:
- Univariate, bivariate, multivariate analysis using **Seaborn**, **Plotly**.
- Outlier detection.
- Correlation heatmaps.
- Hypothesis testing:
  - ANOVA for categorical impact.
  - Chi-Square for independence checks.
  - Jarque-Bera for normality testing.

---

## ⚙️ Preprocessing Pipeline

Used `Pipeline` from `sklearn` to ensure reproducibility and modularity.

| Step               | Technique            |
|--------------------|----------------------|
| Scaling            | StandardScaler       |
| Encoding           | OneHotEncoder        |
| Imputation         | Experimented → Dropped |
| Feature Selection  | Correlation + Feature importance |

---

## 🧪 Experimentation

### ➕ Missing Values Strategy:
- Imputation (mean/median, model-based).
- **Dropped** rows → best MAE/R² balance.

### 🧪 Models Tested (via Optuna):
| Model                     | Notes                                  |
|---------------------------|----------------------------------------|
| Linear Regression         | Baseline                              |
| Random Forest Regressor   | Top performer                          |
| LightGBM                  | High performance, low inference time   |
| XGBoost                   | Decent, but overfit                    |
| Ridge/Lasso/ElasticNet    | Weak performance                       |
| Stacking Regressor        | **Final model** (LGBM + RF + LR)       |

---

## 🚀 Model Building & Evaluation

**Final Model: Stacking Regressor**
- Base Estimators: `RandomForestRegressor`, `LGBMRegressor`
- Final Estimator: `LinearRegression`
- Hyperparameter tuning via **Optuna**
- Tracking all models & metrics via **MLflow**

🧪 **Evaluation Metrics**:
- Mean Absolute Error (MAE)
- R² Score

| Metric      | Value |
|-------------|-------|
| Train MAE   | 2.48  |
| Train R²    | 0.89  |
| Test MAE    | 3.01  |
| Test R²     | 0.83  |
| CV MAE      | 3.07  |

---

## 🛠 MLOps: MLflow + DVC + Dagshub

### 🔁 Versioning:
- **DVC** used to track `data`, `model.pkl`, and pipeline artifacts.
- Registered and version-controlled models.

### 🧪 Tracking:
- **MLflow** integrated with **Dagshub** to:
  - Log hyperparameters.
  - Track metrics across experiments.
  - Compare models visually.


## 🌐 Web App Deployment

**Built with:** `Flask`, `HTML`, `CSS`, `Render.com`

- Inputs taken from user via an interactive form.
- Predicts delivery time instantly based on user input.
- Mobile-responsive and clean UI.
- Shows prediction result clearly with minimal latency.

## 📁 Project Structure
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>