# Customer-Lifetime-Value-Prediction mod

## 📌 Project Overview

This project predicts the future value of customers based on their historical purchasing behavior. The analysis uses the Online Retail dataset and applies customer-level feature engineering and a Random Forest Regression model.

The project helps identify customer purchasing patterns and estimate future customer value.

## 🎯 Objectives

- Analyze customer purchase history
- Calculate Recency, Frequency, Total Spend, and Average Order Value (AOV)
- Predict future customer value
- Evaluate model performance using MAE and RMSE
- Identify the features that contribute most to customer value
- Generate customer-level LTV predictions

## 📊 Dataset

The project uses the Online Retail dataset containing transactional retail purchase records.

Original dataset:
- 541,909 transactions
- 8 columns
- Customer information
- Invoice details
- Product information
- Quantity and Unit Price
- Purchase date
- Country

After data cleaning:
- 397,884 valid transactions
- 4,338 customers identified

## 🛠️ Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Random Forest Regression
- Excel
- VS Code

## 🔄 Project Workflow

1. Loaded the Online Retail dataset.
2. Checked missing values and data types.
3. Removed transactions without Customer IDs.
4. Removed cancelled orders.
5. Removed invalid quantities and prices.
6. Calculated transaction-level Total Amount.
7. Created customer-level features:
   - Recency
   - Frequency
   - Total Spend
   - AOV
8. Created a future customer value target using a historical cutoff date.
9. Split the data into training and testing sets.
10. Trained a Random Forest Regression model.
11. Evaluated the model using MAE and RMSE.
12. Generated customer LTV predictions.
13. Saved the trained model and prediction dataset.
14. Created visualizations for model analysis.

## 🤖 Model Performance

| Metric | Result |
|---|---:|
| MAE | 944.64 |
| RMSE | 4651.88 |

## 📈 Visualizations

### Customer Future Value Distribution

Shows the distribution of predicted future customer value.

### Actual vs Predicted Customer LTV

Compares actual future customer value with the values predicted by the model.

### Feature Importance

Shows the relative importance of the features used by the Random Forest model.

## 📁 Project Files

- `Online Retail.xlsx` — Original dataset
- `customer_lifetime_value.py` — Complete Python analysis and ML code
- `Customer_LTV_Predictions.csv` — Customer-level LTV predictions
- `customer_ltv_model.pkl` — Trained Random Forest model
- `ltv_distribution.png` — Future value distribution
- `actual_vs_predicted_ltv.png` — Actual vs predicted values
- `feature_importance.png` — Model feature importance

## 💡 Key Insight

Total customer spending was the most important feature in the trained model, followed by Average Order Value (AOV), Recency, and Frequency.

## ✅ Conclusion

The project demonstrates how customer transaction history can be transformed into behavioral features and used with machine learning to estimate future customer value. The resulting predictions can support customer segmentation and targeted retention or marketing strategies.

## 👤 Author

**Navaneeth Sheri**

Aspiring Data Analyst  
Skills: Python | SQL | Excel | Power BI | Data Analysis
