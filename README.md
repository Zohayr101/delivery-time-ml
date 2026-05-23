# Semi-Truck Delivery Time Prediction Using Machine Learning

## Project Overview
This machine learning project predicts semi-truck delivery times using supervised regression techniques.

The goal is to analyze logistics variables such as:
- Distance traveled
- Cargo weight
- Shipping priority
- Traffic conditions
- Weather conditions

and estimate delivery duration in hours.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Dataset Features

| Feature | Description |
|---|---|
| Distance_Miles | Delivery distance in miles |
| Cargo_Weight_Lbs | Cargo weight in pounds |
| Shipping_Type | Freight priority level |
| Traffic_Level | Road traffic conditions |
| Weather_Condition | Weather during transport |
| Delivery_Time_Hours | Target variable |

---

## Weekly Progress

### Week 1 - Dataset Setup
- Defined ML regression objective
- Created logistics dataset
- Loaded and explored data structure
- Calculated summary statistics

### Week 2 - Data Cleaning & Preprocessing
- Removed duplicates
- Encoded categorical features
- Engineered weight categories
- Prepared data for modeling

### Week 3 - Exploratory Data Analysis
- Visualized delivery trends
- Analyzed feature correlations
- Identified major delivery-time factors
- Generated scatterplots and boxplots

---

## Future Work
- Train Linear Regression model
- Train Decision Tree Regressor
- Compare MAE, RMSE, and R² scores
- Apply hyperparameter tuning
