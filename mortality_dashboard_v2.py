
import joblib
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Load trained pipeline (ensure correct filename)
model = joblib.load("best_model_xgb.pkl")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# ---- Helper to build input rows ----
def input_row(label_text, component):
    return dbc.Row(
        [
            dbc.Col(dbc.Label(label_text, className="fw-bold"), width=4),
            dbc.Col(component, width=8),
        ],
        className="mb-2",
    )

# ---- Layout ----
app.layout = dbc.Container(
    [
        html.H2("Hospital Mortality Risk Predictor", className="text-center my-4"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        input_row(
                            "Age",
                            dbc.Input(id="age", type="number", min=0, max=120, step=1, value=60),
                        ),
                        input_row(
                            "Gender",
                            dcc.Dropdown(
                                id="gender",
                                options=[{"label": "Male", "value": "MALE"}, {"label": "Female", "value": "FEMALE"}],
                                value="MALE",
                                clearable=False,
                            ),
                        ),
                        input_row(
                            "Rural",
                            dcc.Dropdown(
                                id="rural",
                                options=[{"label": "Yes", "value": "YES"}, {"label": "No", "value": "NO"}],
                                value="NO",
                                clearable=False,
                            ),
                        ),
                        dbc.Button("Predict", id="predict-btn", color="primary", className="mt-3", n_clicks=0),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.H4("Prediction"),
                        html.Div(id="prediction-output", className="lead"),
                    ],
                    width=8,
                ),
            ]
        ),
    ],
    fluid=True,
)

# ---- Dynamic preprocessing ----
def preprocess(age, gender, rural):
    """Build a single-row DataFrame with exactly the columns the pipeline expects."""
    expected_cols = list(model.named_steps["pre"].feature_names_in_)
    # Initialise all numeric columns to 0 and categorical to empty string
    base = {col: 0 for col in expected_cols}
    # Fill categorical placeholders
    if "GENDER" in base:
        base["GENDER"] = gender
    if "RURAL" in base:
        base["RURAL"] = rural
    if "TYPE_OF_ADMISSION-EMERGENCY/OPD" in base:
        base["TYPE_OF_ADMISSION-EMERGENCY/OPD"] = "EMERGENCY"
    # Fill numeric
    if "AGE" in base:
        base["AGE"] = age
    return pd.DataFrame([base])

# ---- Callback ----
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("age", "value"),
    State("gender", "value"),
    State("rural", "value"),
    prevent_initial_call=True,
)
def make_prediction(n_clicks, age, gender, rural):
    try:
        df = preprocess(age, gender, rural)
        proba = model.predict_proba(df)[0, 1]
        return f"Predicted mortality risk: {proba:.2%}"
    except Exception as e:
        return f"⚠️ Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
