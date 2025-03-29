# Customer Churn Classification Project

## Task Objective
Predict customer churn for the period between January 1, 2024, and February 28, 2024, using time-series data with a double customer_id and date index.

## Project Motivation
Develop a robust machine learning solution to identify customers at high risk of churning, enabling proactive retention strategies.

## Project Structure
```
├── data/
│   └── churn_data.csv          # Raw customer transaction data
├── src/
│   ├── processor.py            # Data preprocessing and feature engineering
│   └── trainer.py              # Model training and evaluation
├── main.py                     # Main script to run the entire pipeline
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Technical Approach

### 1. Data Preprocessing
- Handled missing values using mean imputation
- Encoded categorical features using LabelEncoder
- Transformed dates and performed time-based feature engineering

### 2. Feature Engineering Highlights
- Generated 30+ time-dependent features
- Calculated transaction trends
- Derived customer tenure metrics
- Analyzed plan type transitions
- Created churn risk indicators

### 3. Modeling Strategy
- Algorithm: XGBoost Classifier
- Train/Test Split: 80/20
- Validation: Accuracy and F1 Score metrics
- Mitigation of Data Leakage:
  - Careful feature engineering avoiding look-ahead bias
  - Using time-based feature calculations
  - Ensuring features are constructed from past data

### 4. Model Explanation
- Method: LIME (Local Interpretable Model-agnostic Explanations)
- Technical Rationale: 
  - Provides local, instance-level interpretability
  - Helps understand model predictions for individual customers
  - Breaks down feature contributions to a specific prediction
  - Especially useful in churn prediction for understanding key drivers

## Prerequisites
- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`

## Usage
1. Prepare dataset in `data/churn_data.csv`
2. Run pipeline: `python main.py`

## Outputs
- Processed feature dataset
- Trained XGBoost model
- Evaluation metrics JSON
- Confusion matrix visualization
- LIME model explanation plot
