Healthcare Data Analysis and Predictive Modeling – Project Report
1. Project Overview
This project presents an interactive web-based application built using Streamlit for performing predictive analytics on healthcare datasets. Users can upload a healthcare-related CSV file, select a prediction target, choose between a classification or regression model, and receive model performance metrics and visual insights.

2. Objectives
Provide a user-friendly interface to analyze healthcare data.

Allow users to choose between classification and regression based on the nature of the target variable.

Preprocess data including categorical encoding and handling missing values.

Train a Random Forest model and evaluate its performance.

Visualize prediction outcomes for interpretability.

3. Key Features
File Upload: Accepts .csv healthcare datasets via drag-and-drop or manual selection.

Dynamic Target Selection: Automatically lists all columns for users to select as the prediction target.

Model Type Selection: Supports both classification and regression models.

Automatic Data Preprocessing:

Differentiates and encodes low vs. high cardinality categorical features.

Handles missing values by filling with zeros.

Model Training:

Uses RandomForestClassifier for classification tasks.

Uses RandomForestRegressor for regression tasks.

Applies train_test_split for unbiased training/testing.

Performance Metrics:

Accuracy for classification.

Mean Squared Error (MSE) for regression.

Visualization:

For classification: Scatter plot of the first two features with predicted labels.

For regression: Scatter plot comparing true vs. predicted values.

4. Technologies Used
Python Libraries:

streamlit – for building the web application UI.

pandas, numpy – for data manipulation and numerical computations.

sklearn – for model training and performance evaluation.

matplotlib, seaborn – for data visualization.

5. Workflow
Data Input:

User uploads healthcare dataset.

Exploration:

Preview of the uploaded data is displayed.

Model Configuration:

User selects target column and model type.

Preprocessing:

Automatic encoding of categorical features.

Feature-target split.

Model Training:

Random Forest model is trained using training data.

Prediction & Evaluation:

Predictions made on test data.

Accuracy or MSE computed and displayed.

Visualization:

Classification: 2D scatter plot of predicted labels.

Regression: Plot of actual vs. predicted values.

6. Example Use Cases
Predicting disease occurrence (classification).

Estimating medical expenses or hospital stay duration (regression).

Analyzing patient demographics and outcomes.

7. Limitations and Future Improvements
Current Limitations:

No hyperparameter tuning support.

Limited visualization options (2D plots only).

Handling of only basic missing data strategies (fill with 0).

No support for model interpretability like SHAP values.

Potential Enhancements:

Integration of model explainability tools.

Addition of more models (e.g., XGBoost, Logistic Regression).

Better missing value treatment (mean/mode imputation).

Download option for predictions and model.

8. Conclusion
This project demonstrates how an interactive ML-powered healthcare analysis tool can empower users to upload their own datasets, explore relationships, and gain predictive insights quickly and effectively. With further enhancements, it can serve as a valuable prototype for healthcare data scientists and analysts.
