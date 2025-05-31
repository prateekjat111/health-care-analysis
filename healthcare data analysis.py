import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Healthcare Data Analysis and Predictive Modeling")

uploaded_file = st.file_uploader("Upload your healthcare CSV data file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    all_columns = data.columns.tolist()
    target = st.selectbox("Select the target variable (label) for prediction", all_columns)

    model_type = st.selectbox("Select the type of predictive model", ["Classification", "Regression"])

    if st.button("Train Model and Visualize"):
        X = data.drop(columns=[target])
        y = data[target]

        from sklearn.preprocessing import LabelEncoder

        # For regression, ensure target is numeric
        if model_type == "Regression":
            if not pd.api.types.is_numeric_dtype(y):
                try:
                    y = pd.to_numeric(y)
                except Exception:
                    st.error("Target variable must be numeric for regression.")
                    st.stop()

        # Identify categorical columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Separate low and high cardinality categorical columns
        low_card_cols = [col for col in cat_cols if X[col].nunique() < 20]
        high_card_cols = [col for col in cat_cols if X[col].nunique() >= 20]

        # One-hot encode low cardinality categorical columns
        X_low_card = pd.get_dummies(X[low_card_cols])

        # Label encode high cardinality categorical columns
        X_high_card = X[high_card_cols].copy()
        le = LabelEncoder()
        for col in high_card_cols:
            X_high_card[col] = le.fit_transform(X_high_card[col].astype(str))

        # Numeric columns
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Combine all
        X = pd.concat([X_low_card, X_high_card, X[num_cols]], axis=1).fillna(0)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_type == "Classification":
            model = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy on test set: {acc:.2f}")

            # Visualization: plot first two features with predicted labels
            if X.shape[1] >= 2:
                # Sample data for visualization to improve performance
                sample_size = min(200, len(X_test))
                X_vis = X_test.iloc[:sample_size]
                y_pred_vis = y_pred[:sample_size]

                fig, ax = plt.subplots()
                sns.scatterplot(x=X_vis.iloc[:,0], y=X_vis.iloc[:,1], hue=y_pred_vis, palette="Set1", ax=ax)

                # Limit number of text labels to 50 for readability and performance
                label_sample_size = min(50, sample_size)
                for i in range(label_sample_size):
                    ax.text(X_vis.iloc[i,0], X_vis.iloc[i,1], str(y_pred_vis[i]), fontsize=8)

                ax.set_xlabel(X.columns[0])
                ax.set_ylabel(X.columns[1])
                ax.set_title("Predicted Labels Visualization")
                st.pyplot(fig)
            else:
                st.write("Not enough features for 2D visualization.")

        else:  # Regression
            model = RandomForestRegressor(random_state=42, n_estimators=50, max_depth=10)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Mean Squared Error on test set: {mse:.2f}")

            # Visualization: plot true vs predicted values
            sample_size = min(200, len(y_test))
            y_test_vis = y_test.iloc[:sample_size]
            y_pred_vis = y_pred[:sample_size]

            fig, ax = plt.subplots()
            ax.scatter(y_test_vis, y_pred_vis)

            label_sample_size = min(50, sample_size)
            for i in range(label_sample_size):
                ax.text(y_test_vis.iloc[i], y_pred_vis[i], f"{y_pred_vis[i]:.2f}", fontsize=8)

            ax.set_xlabel("True Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("True vs Predicted Values")
            st.pyplot(fig)
