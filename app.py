import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="HCV Classification Dashboard", layout="wide")

st.title("Hepatitis C Virus (HCV) - Classification & Model Evaluation")
st.markdown(
    "This application evaluates pre-trained classification models on an uploaded test dataset. "
    "Upload `test_data.csv` using the sidebar to view dynamic performance metrics and diagnostics."
)

# 1. Sidebar Controls
st.sidebar.header("Evaluation Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)", 
    type=["csv"],
    help="Upload test_data.csv containing all 12 input features and the true 'target' column."
)

model_paths = {
    "Logistic Regression": "model/Logistic_Regression.joblib",
    "Decision Tree": "model/Decision_Tree.joblib",
    "KNN": "model/KNN.joblib",
    "Naive Bayes": "model/Naive_Bayes.joblib",
    "Support Vector Machine": "model/Support_Vector_Machine.joblib"
}

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model", 
    list(model_paths.keys())
)

TARGET_COLUMN = "target"

# 2. Evaluation Logic
if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
        
        if TARGET_COLUMN not in test_df.columns:
            st.error(f"Error: Target column '{TARGET_COLUMN}' not found in the uploaded file.")
        else:
            X_eval = test_df.drop(columns=[TARGET_COLUMN])
            y_eval = test_df[TARGET_COLUMN]

            chosen_artifact_path = model_paths[selected_model_name]

            if not os.path.exists(chosen_artifact_path):
                st.error(f"Model file not found at: {chosen_artifact_path}. Please check your 'model/' directory.")
            else:
                pipeline = joblib.load(chosen_artifact_path)

                y_pred = pipeline.predict(X_eval)
                y_proba = pipeline.predict_proba(X_eval)[:, 1] if hasattr(pipeline, "predict_proba") else None

                acc = accuracy_score(y_eval, y_pred)
                auc = roc_auc_score(y_eval, y_proba) if y_proba is not None else np.nan
                prec = precision_score(y_eval, y_pred, zero_division=0)
                rec = recall_score(y_eval, y_pred, zero_division=0)
                f1 = f1_score(y_eval, y_pred, zero_division=0)
                mcc = matthews_corrcoef(y_eval, y_pred)

                st.subheader(f"Evaluation Metrics: {selected_model_name}")
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("AUC Score", f"{auc:.4f}" if not np.isnan(auc) else "N/A")
                col3.metric("Precision", f"{prec:.4f}")
                col4.metric("Recall", f"{rec:.4f}")
                col5.metric("F1 Score", f"{f1:.4f}")
                col6.metric("MCC Score", f"{mcc:.4f}")

                st.divider()

                col_left, col_right = st.columns(2)

                with col_left:
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_eval, y_pred)
                    fig, ax = plt.subplots(figsize=(4.5, 3.5))
                    sns.heatmap(
                        cm, 
                        annot=True, 
                        fmt="d", 
                        cmap="Blues", 
                        xticklabels=["Healthy (0)", "Disease (1)"],
                        yticklabels=["Healthy (0)", "Disease (1)"],
                        ax=ax
                    )
                    ax.set_xlabel("Predicted Label")
                    ax.set_ylabel("True Label")
                    st.pyplot(fig)
                    plt.close(fig)

                with col_right:
                    st.subheader("Classification Report")
                    report_dict = classification_report(
                        y_eval, 
                        y_pred, 
                        target_names=["Healthy (0)", "Disease (1)"], 
                        output_dict=True
                    )
                    report_df = pd.DataFrame(report_dict).transpose()
                    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while evaluating the dataset: {e}")

else:
    st.info("Please upload `test_data.csv` using the sidebar on the left to view model evaluation results.")
