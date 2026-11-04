# Hepatitis C Virus (HCV) Patient Classification

## a. Problem statement
The objective of this project is to develop and evaluate five supervised machine learning classification models to predict whether an individual has liver disease (Hepatitis, Fibrosis, or Cirrhosis) or is a healthy control based on standard laboratory blood biomarkers and demographic factors. 

Early, non-invasive detection of hepatic pathology using routine biochemical measurements is critical in clinical settings to identify patients at risk before irreversible liver fibrosis, end-stage cirrhosis, or liver failure occurs.

---

## b. Dataset description
* **Source:** UCI Machine Learning Repository (HCV Data, Dataset ID: 571)
* **Instance Count:** 615 samples
* **Predictive Features:** 12 total features
  * **Demographics (2 features):**
    * `Age`: Age of the subject in years (continuous numeric)
    * `Sex`: Biological sex (categorical: 'm', 'f')
  * **Laboratory Blood Biomarkers (10 continuous numeric features):**
    * `ALB`: Albumin (g/L)
    * `ALP`: Alkaline Phosphatase (IU/L)
    * `ALT`: Alanine Transaminase (U/L)
    * `AST`: Aspartate Transaminase (U/L)
    * `BIL`: Bilirubin (μmol/L)
    * `CHE`: Cholinesterase (kU/L)
    * `CHOL`: Cholesterol (mmol/L)
    * `CREA`: Creatinine (μmol/L)
    * `GGT`: Gamma-Glutamyl Transferase (U/L)
    * `PROT`: Total Protein (g/L)
* **Target Formulation (`target`):**
  * `0` (Healthy Control): Derived from original categories `0=Blood Donor` and `0s=suspect Blood Donor` (540 total samples)
  * `1` (Liver Disease): Derived from categories `1=Hepatitis`, `2=Fibrosis`, and `3=Cirrhosis` (75 total samples)
* **Preprocessing Pipeline:**
  * Dropped the uninformative patient identifier column (`Unnamed: 0`).
  * Imputed missing values in continuous blood biomarkers using the feature median to resist skew from clinical outlier spikes.
  * Encoded categorical `Sex` via One-Hot Encoding (`drop='first'`).
  * Scaled continuous numeric biomarkers using `StandardScaler` (z-score normalization).
  * Applied an 80/20 stratified train-test split (492 train instances, 123 test instances) to maintain the exact ~88% to ~12% class balance on unseen evaluation data.

---

## c. Github Repository Link
* **GitHub Repository URL:** [INSERT_YOUR_GITHUB_REPO_URL_HERE]
* **Streamlit Live Deployment URL:** [INSERT_YOUR_STREAMLIT_APP_URL_HERE]

---

## d. Models used:

All five classifiers were fitted on the training split (492 instances) and evaluated on the exact same unseen test split (`test_data.csv`, 123 instances).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9593 | 0.9228 | 1.0000 | 0.6667 | 0.8000 | 0.7982 |
| Decision Tree | 0.9512 | 0.7290 | 0.8462 | 0.7333 | 0.7857 | 0.7608 |
| KNN | 0.9431 | 0.9278 | 1.0000 | 0.5333 | 0.6957 | 0.7077 |
| Naive Bayes | 0.9106 | 0.9222 | 0.6429 | 0.6000 | 0.6207 | 0.5705 |
| Support Vector Machine | 0.9350 | 0.9802 | 0.7692 | 0.6667 | 0.7143 | 0.6800 |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest overall Accuracy (0.9593), F1 score (0.8000), and MCC (0.7982) with zero False Positives (Precision = 1.0000). The standardized biomarker boundaries allow the linear hyperplane to cleanly isolate unambiguous healthy controls, though it missed one-third of disease cases (Recall = 0.6667). |
| **Decision Tree** | Produced the highest Recall among all tested models (0.7333), capturing non-linear threshold triggers (e.g., elevated AST paired with high BIL). However, its AUC (0.7290) was the lowest because unpruned probability steps yield crude, step-like discrimination thresholds rather than smooth probabilistic curves. |
| **KNN** | Demonstrated perfect Precision (1.0000) and a competitive AUC (0.9278), but achieved the lowest Recall (0.5333). In a 12-dimensional feature space with severe class imbalance (88% healthy), majority-class nearest neighbors dominate localized voting spheres around peripheral disease instances. |
| **Naive Bayes** | Showed the weakest overall performance across Accuracy (0.9106), F1 (0.6207), and MCC (0.5705). The algorithm's fundamental assumption of conditional feature independence is heavily violated by clinical biomarkers; liver enzymes like ALT, AST, and GGT exhibit strong mutual collinearity during hepatic cell damage. |
| **Support Vector Machine** | Achieved the highest discrimination capacity with an outstanding AUC of 0.9802 using the RBF kernel. It successfully maps non-linear enzyme interactions into higher dimensions, maintaining stable trade-offs across precision (0.7692) and recall (0.6667). |
| **Overall, Winner for your dataset?** | **Logistic Regression** is the overall top performer based on the highest balanced metric scores (F1: 0.8000, MCC: 0.7982, Accuracy: 0.9593). However, for practical clinical diagnosis where detecting sick patients is paramount, **Support Vector Machine (SVM)** is a strong co-winner due to its superior AUC (0.9802), allowing clinicians to adjust the probability threshold to prioritize Recall. |
