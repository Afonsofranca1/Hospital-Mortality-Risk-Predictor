# Hospital Mortality Risk Prediction

This project uses data science and machine learning to predict **in-hospital mortality** for patients admitted with cardiovascular conditions. It leverages real clinical data and deploys a predictive dashboard using Plotly Dash.

---

## Objective

To build a predictive model that estimates a patient’s risk of death during hospitalization based on clinical, demographic, and biochemical features — and to deploy this model as an interactive, user-friendly dashboard for hospital or triage use.

---

## Technologies Used

| Component | Tools/Libraries |
|----------:|----------------|
| Data Storage | CSV |
| EDA | Pandas, Matplotlib |
| ML Models | Logistic Regression, Random Forest, XGBoost, SVM |
| Optimization | GridSearchCV, Stratified K-Fold |
| Imbalance Handling | Class weighting |
| Deployment | Plotly Dash, Dash Bootstrap Components |
| Model Persistence | Joblib |
| Notebook | Jupyter |

---

## Workflow

1. **Data Extraction & Cleaning**:
   - Merged multiple hospital data tables (admissions, outcomes).
   - Cleaned invalid/missing entries (`EMPTY`, `'\'`, etc.).
   - Removed rows with missing target or features required for prediction.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized age, comorbidities, and admission types.
   - Found higher mortality in older patients and those with renal/cardiac history.

3. **Feature Engineering & Preprocessing**:
   - Encoded categorical features.
   - Imputed missing values.
   - Balanced the dataset using class weighting.

4. **Model Building**:
   - Trained and tuned Logistic Regression, Random Forest, XGBoost, and SVM.
   - Evaluated with ROC-AUC, precision, recall, and F1 score.
   - Best model: **XGBoost** (highest AUC, good calibration).

5. **Deployment**:
   - Built interactive dashboard in Dash.
   - User inputs Age, Gender, Rural status → system returns predicted mortality risk.

---

## Results

After hyperparameter optimization with `GridSearchCV` and 5-fold stratified cross-validation, the top models achieved the following AUC scores:

| Model | AUC | Best Parameters |
|-------|-----|-----------------|
| **XGBoost** (`xgb`) | **0.953** | `{'clf__learning_rate': 0.1, 'clf__max_depth': 6, 'clf__n_estimators': 300}` |
| Random Forest (`rf`) | 0.953 | `{'clf__max_depth': 20, 'clf__n_estimators': 400}` |
| Support Vector Machine (`svm`) | 0.918 | `{'clf__C': 0.5, 'clf__kernel': 'rbf'}` |
| Logistic Regression (`logreg`) | 0.911 | `{'clf__C': 0.1}` |

**Final model selected**: **XGBoost**  
It delivered the best trade-off between performance and interpretability, with an AUC of 0.953 and well-calibrated probabilities.

---

## Interpreting Predictions

**Note:** Variables like `GENDER` and `RURAL` have a **small weight** in the model and may **not significantly affect** predictions in isolation.

This is expected:
- Most of the model’s predictive power comes from medical features (e.g., BNP, Creatinine, ACS, CKD).
- `GENDER` and `RURAL` are included for completeness but are not strong mortality indicators on their own.

Solution: You can extend the dashboard to allow modifying more clinical features to see bigger differences in predicted risk.

---

## 🖥Dashboard Preview

After installing dependencies, run:

```bash
pip install dash plotly dash-bootstrap-components pandas joblib
python mortality_dashboard_v2.py
```

Visit [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser to access the app.

---

## Files Overview

| File | Description |
|------|-------------|
| `mortality_dashboard_v2.py` | Final Dash app with working pipeline |
| `best_model_xgb.pkl`        | Saved XGBoost model pipeline |
| `notebooks/eda_modeling.ipynb` | EDA + model training notebook |
| `README.md`                 | Project documentation |

---

## Future Improvements

- Add lab values (e.g., BNP, Creatinine) to dashboard inputs.
- Include SHAP plots for explainable AI insights.
- Deploy as a web app via Heroku, Render, or Docker.

---

## Author

Developed by Afonso, Biomedical Engineer & Aspiring Data Scientist.  
Feel free to connect on [LinkedIn](www.linkedin.com/in/afonso-frança) or check more work on [GitHub](https://github.com/Afonsofranca1).
