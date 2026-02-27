# Customer Churn Prediction

## Project Overview
This project aims to predict customer churn for a telecom company using machine learning.  
By identifying customers likely to leave, the company can take proactive measures to retain them.

## Dataset
- **Source:** Telco Customer Churn dataset  
- **Size:** ~7,000 customers  
- **Features:** Demographics, account info, services subscribed, monthly charges, tenure, etc.

## Data Cleaning & Preprocessing
- Handled missing values (e.g., `TotalCharges`)  
- Encoded categorical variables (binary mapping & one-hot encoding)  
- Checked correlations and cleaned irrelevant columns (e.g., `customerID`)  

## Exploratory Data Analysis (EDA)
- Distribution of churned vs. retained customers  
- Tenure analysis, monthly charges, contract types  
- Visualizations (histograms, countplots, heatmaps)  

## Modeling
- **Models used:** Logistic Regression, Random Forest  
- **Train/Test Split:** 80% / 20%  
- Standardized features when needed  

## Evaluation
- Logistic Regression: **Accuracy = 0.738**, **Recall (Churn=1) = 0.68**  
- Random Forest: **Accuracy = 0.712**  
- Optimized threshold for Logistic Regression → **Recall = 0.90** at **threshold = 0.53**  
- ROC Curve: **AUC = 0.82**  

## Feature Importance & Insights
### Top factors increasing churn:
- Month-to-month contract  
- High monthly charges  
- Short tenure  
- Paperless billing  

### Top factors reducing churn:
- Long-term contracts (1 or 2 years)  
- Additional services (Tech Support, Online Security)  
- Longer tenure  

**Business Insights:**
- Focus retention campaigns on month-to-month customers  
- Offer incentives for early subscription to long-term contracts  
- Promote additional services to reduce churn  

## Conclusion
A Logistic Regression model was developed to predict customer churn.  
The model achieved an AUC score of 0.82, indicating strong classification performance.  
By optimizing the decision threshold to 0.53, recall was increased to 90%, allowing the company to identify the majority of customers likely to churn.  

This model can help the company proactively target high-risk customers with retention strategies.

## Next Steps (Optional)
- Collect additional customer behavior data for better prediction  
- Experiment with ensemble models (XGBoost, LightGBM)  
- Deploy the model in a dashboard for real-time churn monitoring