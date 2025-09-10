# Credit Worthiness
## Run on Google Colab

To explore the notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QxD91FJupnng5tzbLonmusfUyRMf-hpe?usp=sharing)

---

# Creditworthiness Prediction

**Objective:**
Develop a machine learning model to **estimate a customer’s creditworthiness**, supporting the bank’s internal decision-making process on whether to **approve or reject a credit card request**.

---

## Business Context

This project uses anonymized customer data from individuals who already own a credit card and are **repaying their installments regularly**. The goal is to predict if **new applicants** are also likely to be trustworthy.

While the dataset does not include country-specific information, we base our approach on **common industry practices** and the structure of the data.

---

## Business Trade-Offs

Depending on the company’s priorities, the modeling objective can change:

* **Minimizing risk** → Focus on **precision**
  Avoid giving credit to unreliable applicants.

* **Maximizing opportunity** → Focus on **recall**
  Don’t miss out on potentially good customers.

* **Balanced approach** → Focus on **F1-score**
  Ensure a fair trade-off between false positives and false negatives.

This project primarily uses **F1-score** as the guiding metric, but alternatives are explored depending on model behavior.

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

* Statistical and visual exploration of numerical and categorical features
* Distributional comparisons using **Kolmogorov–Smirnov test**
* Handling class imbalances and assessing missing data patterns

### 2. Preprocessing Pipeline

* Missing value imputation with `SimpleImputer`
* Encoding:

  * `OneHotEncoder` for nominal variables
  * `OrdinalEncoder` for ordered categories
* Feature scaling using `StandardScaler`
* Use of `ColumnTransformer` for column-wise transformations
* Full pipeline construction via `Pipeline` API

### 3. Model Training & Optimization

Multiple models were trained and compared:

* **Logistic Regression**
* **Decision Tree**
* **Random Forest**
* **XGBoost (XGBClassifier)**

Hyperparameter tuning was conducted using:

* `RandomizedSearchCV` with distributions like `uniform` and `rand`

### 4. Model Evaluation

Performance assessed using:

* **F1-score** (primary)
* **Precision**, **Recall**
* **ROC AUC**
* **Accuracy**
* Visual tools:

  * `RocCurveDisplay`
  * `PrecisionRecallDisplay`
  * `ConfusionMatrixDisplay`
  * `classification_report`

### 5. Model Interpretation

* Feature importance visualization (e.g., decision trees)
* Tree plotting via `plot_tree`
* (Optional LIME support – can be added later)

---


