# Cardiovascular Disease Prediction Using Machine Learning and XAI

## Overview

This project focuses on predicting cardiovascular diseases (CVD), specifically heart attacks, using machine learning (ML) models and explainable AI (XAI) techniques. Cardiovascular diseases remain the leading cause of death globally, emphasizing the need for accurate, reliable, and efficient prediction models. 

Using advanced data mining and ML techniques, we aim to predict the likelihood of heart disease based on a patient's medical features. Additionally, we incorporate XAI methods to interpret model predictions, providing insights into critical features influencing the outcomes.

---

## Key Features
- **Machine Learning Models:** Evaluation of several algorithms such as Logistic Regression, Decision Tree, Random Forest, Support Vector Classifier, Naive Bayes, XGBoost, and K-Nearest Neighbors (KNN).
- **Performance Metrics:** Achieved a **95.60% accuracy** using the KNN algorithm.
- **Explainable AI (XAI):** Used **LIME (Local Interpretable Model-agnostic Explanations)** to identify and explain the importance of features contributing to predictions.
- **Feature Importance Analysis:** Highlighted the most impactful medical features, such as chest pain (`cp`), old peak (`oldpeak`), and others, for predicting heart disease.

---

## Data and Methodology

### Dataset
The dataset used includes critical medical parameters such as:
- **cp (chest pain type)**: Strongly linked to heart attack prediction.
- **oldpeak**: Reflects ST Depression in ECG, indicating myocardial ischemia.
- **ca (number of major vessels)**: Related to blood flow conditions.
- Other parameters include age, cholesterol levels, blood pressure, and more.

### Machine Learning Models
The following models were trained and evaluated:
1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Support Vector Classifier**
5. **Naive Bayes**
6. **XGBoost**
7. **K-Nearest Neighbors (KNN)**

#### Model Performance
| Model                  | Accuracy (%) |
|------------------------|--------------|
| Logistic Regression    | 89.45        |
| Decision Tree          | 92.30        |
| Random Forest          | 93.20        |
| Support Vector Classifier | 90.10      |
| Naive Bayes            | 88.50        |
| XGBoost                | 94.80        |
| **KNN (Best)**         | **95.60**    |

### Explainable AI (XAI)
We used **LIME** to:
- Visualize feature importance for individual predictions.
- Provide interpretable insights for both positive and negative predictions.

---

## Results

### Performance Metrics (KNN)
- **Accuracy:** 95.60%
- **Precision (Positive):** 0.97
- **Recall (Positive):** 0.94
- **F1-Score (Positive):** 0.95

### Feature Importance
Using LIME, we identified the most impactful features for heart disease prediction:
- **Orange Features:** Indicate higher importance and likelihood of heart disease.
- **Blue Features:** Indicate lower importance and likelihood of heart disease.

Figures illustrate the following:
1. **Important Features**: `cp`, `oldpeak`, `ca`, etc.
2. **Value Counts**: Highlighting the proportion of patients at risk.

---

## Visualizations

### Model Accuracy Comparison
![Model Accuracy Comparison](https://github.com/UtshoData/heart-disease-prediction-using-ML-and-XAI/assets/157609050/7218f3a4-a23d-4866-aa6d-3acb8d1def80)

### Feature Importance with LIME
**Predicted Disease:**  
![Feature Importance (Disease)](https://github.com/UtshoData/heart-disease-prediction-using-ML-and-XAI/assets/157609050/6a6261d8-6d3c-4ee8-ba33-a4ae1435847a)

**Predicted No Disease:**  
![Feature Importance (No Disease)](https://github.com/UtshoData/heart-disease-prediction-using-ML-and-XAI/assets/157609050/106cb6f8-4a5d-47a0-ae53-16f817f92fa7)

---

## Conclusion
- The **KNN algorithm** outperformed other models in terms of accuracy and reliability for heart disease prediction.
- The **XAI technique (LIME)** provided transparency in model decisions, aiding medical professionals in understanding predictions.
- This approach underscores the potential of ML and XAI in improving early detection and management of cardiovascular diseases.

---

## Getting Started

### Prerequisites
- Python 3.8 or later
- Libraries: `numpy`, `pandas`, `scikit-learn`, `lime`, `matplotlib`, `seaborn`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/UtshoData/heart-disease-prediction-using-ML-and-XAI.git
pip install -r requirements.txt
python main.py

Copy this code and save it as `README.md` in the root directory of your GitHub repository. Adjust the dataset link and any other specific project details as needed.

