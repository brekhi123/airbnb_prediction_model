# Airbnb Review Prediction Model

## Overview
This repository contains the implementation of the Airbnb Review Prediction Model, an academic project completed for the CPSC 330: Applied Machine Learning course. The project focuses on predicting the `reviews_per_month` for New York City Airbnb listings from 2019, using the [New York City Airbnb Open Data](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) dataset. The goal is to estimate listing popularity to help hosts optimize their listings.

The project is implemented in a Jupyter Notebook (`hw5final.ipynb`) and follows a machine learning pipeline, including data splitting, exploratory data analysis (EDA), feature engineering, preprocessing, model selection, hyperparameter tuning, feature selection, and result interpretation using SHAP values.

## Dataset
The dataset is the [New York City Airbnb Open Data](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) from 2019, with 48,895 listings and 16 features:
- **Features**: `id`, `name`, `host_id`, `host_name`, `neighbourhood_group`, `neighbourhood`, `latitude`, `longitude`, `room_type`, `price`, `minimum_nights`, `number_of_reviews`, `last_review`, `calculated_host_listings_count`, `availability_365`.
- **Target**: `reviews_per_month`, a proxy for listing popularity.
- **Key Characteristics**:
  - Missing values in `reviews_per_month`, `last_review`, `host_name`, and `name`.
  - Numerical and categorical features with varying scales and class imbalances.
  - Outliers in `reviews_per_month` (

System: e.g., max value of 58.5).

## Project Structure
The notebook is organized into 14 sections, each addressing a specific task in the machine learning workflow:

1. **Problem Definition**: Selected regression to predict `reviews_per_month` based on features like location, price, and availability.
2. **Data Splitting**: Split data into 70% training and 30% test sets (`random_state=123`). Used a 2000-example training subset due to computational limits.
3. **Exploratory Data Analysis (EDA)**:
   - **Summary Statistics**:
     - Quantiles of `reviews_per_month` show most values near 0, with outliers up to 58.5, justifying R² as the metric.
     - Class imbalances in `room_type` and `neighbourhood_group` may bias predictions.
   - **Visualizations**:
     - Boxplot of `reviews_per_month` by `room_type` shows higher popularity for entire homes/apartments.
     - Correlation heatmap highlights a strong link between `number_of_reviews` and `reviews_per_month`.
   - **Observations**: Missing data, scale inconsistencies, and imbalances require preprocessing.
   - **Metric**: R² chosen due to outliers and small target values.
4. **Feature Engineering**: Added `minimum_nights_binned` and `has_name` features.
5. **Preprocessing**: Used `ColumnTransformer` for numerical (scaling, imputation), categorical (one-hot encoding), and text (`name` via `CountVectorizer`) features.
6. **Baseline Model**: DummyRegressor scored R² = 0.0.
7. **Linear Models**: Ridge regression scored a mean cross-validation R² of 0.37.
8. **Different Models**: Evaluated KNeighborsRegressor (0.32), DecisionTreeRegressor (0.08), and RandomForestRegressor (0.51).
9. **Feature Selection**: RFECV improved KNN (0.35) and DecisionTreeRegressor (0.14), with RandomForestRegressor stable at 0.50.
10. **Hyperparameter Tuning**: Tuned parameters (e.g., Ridge: `alpha=74.25`, RandomForest: `n_estimators=105`, `max_depth=33`), improving Ridge to 0.41 and RandomForest to 0.51.
11. **Feature Importances**: RandomForest identified `number_of_reviews` (0.38) and `availability_365` (0.12) as key predictors. SHAP analysis showed varying feature impacts.
12. **Test Results**: RandomForestRegressor achieved a test R² score of **0.4625**.
13. **Summary**: Compiled model performance and insights, suggesting more folds and full dataset training for improvement.
14. **Takeaway**: Emphasized generalization to balance bias and variance for robust models.

## Key Findings
- **Best Model**: RandomForestRegressor achieved a mean cross-validation R² of 0.51 and a test R² of 0.4625.
- **Key Features**: `number_of_reviews` and `availability_365` were most influential, with varying impacts (SHAP analysis).
- **Challenges**: Class imbalances, missing data, and a reduced training set (2000 examples) limited performance.
- **Metric**: R² was suitable due to outliers and small target values.

## Potential Improvements
- **More Cross-Validation Folds**: Used two folds due to time constraints; more folds could improve robustness.
- **Full Training Set**: Training on all data could enhance performance.
- **Hyperparameter Tuning**: Tuning additional RandomForest parameters (e.g., `min_samples_split`) could improve results.
- **Feature Engineering**: Adding external data (e.g., `reviews.csv.gz`) or interaction terms could capture more patterns.
- **Ensembling**: Stacking models like Ridge and RandomForest could boost accuracy.

## Dependencies
The notebook requires:
- `pandas`
- `matplotlib`
- `numpy`
- `scikit-learn`
- `shap`
- `otter`

Install with:
```bash
pip install pandas matplotlib numpy scikit-learn shap otter-grader
```

## How to Run
1. **Download Dataset**: Get `AB_NYC_2019.csv` from [Kaggle](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) and place it in the notebook directory.
2. **Set Up Environment**: Install required libraries.
3. **Run Notebook**:
   - Open `hw5final.ipynb` in Jupyter Notebook or JupyterLab.
   - Run all cells (`Kernel -> Restart Kernel and Clear All Outputs`, then `Run -> Run All Cells`).
4. **View Results**: Outputs include tables, visualizations, and the test R² score (0.4625).

## Notes
- Tested with Python 3.12.0 in the `cpsc330` conda environment.
- Used a 2000-example training subset and two-fold cross-validation due to computational constraints.
- Ensure plots and outputs are rendered for review.
- Export a PDF or HTML version if the notebook is too large for upload.

## Results
The final test score is **R² = 0.4625** using the RandomForestRegressor.

## Acknowledgments
- SHAP analysis code adapted from the CPSC 330 lecture notes.
