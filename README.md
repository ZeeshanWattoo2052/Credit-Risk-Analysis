# Credit Risk Analysis Using Random Forest

This project uses a machine learning approach to predict credit risk, specifically identifying the likelihood of a loan being charged off. A Random Forest model was used and compared against baseline models like Logistic Regression and K-Nearest Neighbors.

## Dataset

The dataset contains information about loan applicants, including:
- Age, Income, Employment Length
- Loan Purpose (Intent), Loan Amount, Interest Rate
- Home Ownership, Credit History Length, Previous Default Status
- Loan Approval Status (Target Variable)

Dataset source: [credit_risk.csv](./credit_risk.csv)

## Workflow

1. Data Cleaning and Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training using Random Forest
5. Model Evaluation (Accuracy, Confusion Matrix, Classification Report)
6. Comparison with Logistic Regression and KNN

## Results

The Random Forest model achieved **91.77% accuracy**, outperforming other models.

## Requirements

Install required packages with:

```bash
pip install -r requirements.txt
```

## Run

You can open the notebook to explore the analysis:

```bash
jupyter notebook Credit_Risk_Analysis_Using_RF.ipynb
```

---

*Created By: Zeeshan Ahmad Wattoo*  
*Project: Credit Risk Prediction*
