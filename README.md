# Customer Churn Prediction Pipeline 📊

## Project Overview

The goal of this project is to analyze customer behavior data and predict, using machine learning, which customers are at risk of churning (i.e., leaving the company).

The project implements an end-to-end machine learning pipeline covering data preprocessing, missing value handling, outlier detection, feature engineering, and feature scaling. Two classification algorithms — **Logistic Regression** and **K-Nearest Neighbors (KNN)** — were trained and their performances compared.

## Repository Contents

- `churn_prediction.py`: Main Python script containing data processing, model training, and evaluation steps.
- `musteri_veri_seti.csv`: Customer dataset used for training and testing the models.
- `requirements.txt`: File listing the project's dependencies and library versions.
- `README.md`: Project documentation.

## How to Run

To run this project in your local environment, follow the steps below:

1. **Clone or Download the Repository:**
   Download the project files to your machine and navigate to the project directory via terminal.

2. **Install Required Libraries:**
   Install all dependencies at once by running:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Project**
   Once the environment is set up, run the main script to execute the full pipeline:
   ```bash
   python churn_prediction.py
   ```

## Results and Evaluation Summary

* **Model Selection:** Based on validation results, the Logistic Regression model (90.4% accuracy) outperformed the KNN model (84.9% accuracy) in terms of generalization performance and was therefore selected as the final model.
* **Test Performance:** The selected Logistic Regression model achieved 80.8% accuracy on the unseen test set.
* **Class Imbalance Effect:** When examining the test set metrics of the selected Logistic Regression model, it performed excellently at identifying customers who remain with the company (Churn = 0).
* **Key Finding:** In contrast, the recall for churned customers (Churn = 1) was 52%. This is attributed not to a weakness in the algorithm itself, but to the notable class imbalance in the dataset (332 retained vs. 154 churned).
* **Future Work:** In future iterations, techniques such as synthetic data generation (SMOTE) or weighted classification (class weights) could be applied to improve the recall for the minority (churned) class.

---
**Developer:** Samet Ozan Topcu
