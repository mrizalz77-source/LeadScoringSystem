from flask import Flask, render_template, request
import pandas as pd
import joblib
import shap
import sqlite3

from openai import OpenAI


app = Flask(__name__)
client = OpenAI()

def init_db():
    conn = sqlite3.connect("lead_scoring.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            age INTEGER,
            job TEXT,
            lead_score REAL,
            priority TEXT,
            recommendation TEXT
        )
    """)

    conn.commit()
    conn.close()

def save_prediction(customer_data, lead_score, priority, recommendation):
    conn = sqlite3.connect("lead_scoring.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (age, job, lead_score, priority, recommendation)
        VALUES (?, ?, ?, ?, ?)
    """, (
        customer_data["age"],
        customer_data["job"],
        lead_score,
        priority,
        recommendation
    ))

    conn.commit()

    prediction_id = cursor.lastrowid

    conn.close()

    return prediction_id    

# -------------------------------------------------
# Load trained model
# -------------------------------------------------

model = joblib.load("models/lead_scoring_model.pkl")

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

explainer = shap.TreeExplainer(classifier)


# -------------------------------------------------
# Format SHAP feature names
# -------------------------------------------------

def format_feature_name(feature, customer_data):

    # Numerical features
    if feature.startswith("num__"):
        name = feature.replace("num__", "")

        labels = {
            "age": "Age",
            "balance": "Account Balance",
            "day": "Contact Day",
            "campaign": "Campaign Contacts",
            "pdays": "Days Since Previous Contact",
            "previous": "Previous Contacts"
        }

        return f"{labels.get(name, name)}: {customer_data[name]}"

    # Categorical features
    if feature.startswith("cat__"):
        encoded_feature = feature.replace("cat__", "")

        categorical_fields = [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "poutcome"
        ]

        labels = {
            "job": "Job",
            "marital": "Marital Status",
            "education": "Education",
            "default": "Credit Default",
            "housing": "Housing Loan",
            "loan": "Personal Loan",
            "contact": "Contact Type",
            "month": "Contact Month",
            "poutcome": "Previous Campaign Outcome"
        }

        for field in categorical_fields:
            value = str(customer_data[field])

            if encoded_feature == f"{field}_{value}":
                return f"{labels[field]}: {value.title()}"

    return None


# -------------------------------------------------
# Generate Sales Recommendation
# -------------------------------------------------

def generate_sales_recommendation(priority, lead_score, explanations):

    positive_factors = [
        item["feature"]
        for item in explanations
        if item["shap_value"] > 0
    ]

    if priority == "High":
        recommendation = (
            f"This is a high-priority lead with a lead score of {lead_score}/100. "
            "The customer should be prioritised for sales follow-up. "
        )

    elif priority == "Medium":
        recommendation = (
            f"This is a medium-priority lead with a lead score of {lead_score}/100. "
            "The customer shows some potential and may be considered for follow-up "
            "after higher-priority leads. "
        )

    else:
        recommendation = (
            f"This is a low-priority lead with a lead score of {lead_score}/100. "
            "Immediate sales follow-up may not be necessary. "
            "The customer may be considered for future marketing activities. "
        )

    if positive_factors:
        recommendation += (
            "Positive factors identified by the model include "
            + ", ".join(positive_factors[:3])
            + "."
        )

    return recommendation

def generate_llm_recommendation(priority, lead_score, explanations):

    # Build SHAP factor text
    factor_lines = []

    for item in explanations:
        influence = "Positive" if item["shap_value"] > 0 else "Negative"

        factor_lines.append(
            f"- {item['feature']}: {influence}"
        )

    factors_text = "\n".join(factor_lines)

    prompt = f"""
You are assisting an SME salesperson by providing a short sales recommendation.

The lead score, priority, and model factors below were generated from
historical customer and marketing data.

Lead Score: {lead_score}/100
Priority: {priority}

Historical factors influencing the machine learning prediction:
{factors_text}

Generate a practical recommendation for the salesperson.

Important rules:
- Treat all model factors as historical information only.
- Do not interpret historical month, day, contact method, campaign,
  or other recorded information as instructions for future actions.
- Do not invent customer information.
- Do not change the lead score or priority.
- Do not claim that the customer will definitely convert.
- Use the factors only to help explain why the lead may deserve
  more or less sales attention.
- Recommend a reasonable next sales action based mainly on the
  lead score and priority.
- Keep the recommendation to 2 or 3 short sentences.
- Use simple business language.
"""

    try:
        response = client.responses.create(
            model="gpt-5.6-luna",
            input=prompt
        )

        return response.output_text

    except Exception as error:
        print("LLM API error:", error)

        # Fall back to our existing rule-based recommendation
        return generate_sales_recommendation(
            priority,
            lead_score,
            explanations
        )    

# -------------------------------------------------
# Home Page
# -------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect("lead_scoring.db")
    cursor = conn.cursor()

    total = cursor.execute(
        "SELECT COUNT(*) FROM predictions"
    ).fetchone()[0]

    high = cursor.execute(
        "SELECT COUNT(*) FROM predictions WHERE priority = 'High'"
    ).fetchone()[0]

    medium = cursor.execute(
        "SELECT COUNT(*) FROM predictions WHERE priority = 'Medium'"
    ).fetchone()[0]

    low = cursor.execute(
        "SELECT COUNT(*) FROM predictions WHERE priority = 'Low'"
    ).fetchone()[0]

    recent_predictions = cursor.execute("""
        SELECT id, 
        datetime(created_at, '+8 hours') AS created_at, age, job, lead_score, priority
        FROM predictions
        ORDER BY id DESC
        LIMIT 5
    """).fetchall()

    conn.close()

    return render_template(
        "dashboard.html",
        total=total,
        high=high,
        medium=medium,
        low=low,
        recent_predictions=recent_predictions
    )

# ---------------------------------------------
# Prediction History
# ---------------------------------------------
@app.route("/history")
def history():
    conn = sqlite3.connect("lead_scoring.db")
    cursor = conn.cursor()

    predictions = cursor.execute("""
        SELECT id, 
                datetime(created_at, '+8 hours') AS created_at, age, job,
               lead_score, priority, recommendation
        FROM predictions
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return render_template(
        "history.html",
        predictions=predictions
    )

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/upload-predict", methods=["POST"])
def upload_predict():

    # Check that a file was uploaded
    if "file" not in request.files:
        return "No file uploaded."

    file = request.files["file"]

    if file.filename == "":
        return "No file selected."

     # Validate file type
    if not file.filename.lower().endswith(".csv"):
        return "Invalid file type. Please upload a CSV file only."

    try:

        
        
        
        # Read uploaded CSV
        batch_data = pd.read_csv(file)

        # Required model input columns
        required_columns = [
            "age",
            "job",
            "marital",
            "education",
            "default",
            "balance",
            "housing",
            "loan",
            "contact",
            "day",
            "month",
            "campaign",
            "pdays",
            "previous",
            "poutcome"
        ]

        # Check for missing columns
        missing_columns = [
            column for column in required_columns
            if column not in batch_data.columns
        ]

        if missing_columns:
            return (
                "CSV is missing required columns: "
                + ", ".join(missing_columns)
            )

        # Keep columns in exactly the expected order
        X_batch = batch_data[required_columns]

        # Generate conversion probabilities
        probabilities = model.predict_proba(X_batch)[:, 1]

        # Convert probability to 0-100 lead score
        batch_data["lead_score"] = (
            probabilities * 100
        ).round(1)

        # Assign priority
        def assign_priority(score):
            if score >= 70:
                return "High"
            elif score >= 40:
                return "Medium"
            else:
                return "Low"

        batch_data["priority"] = batch_data[
            "lead_score"
        ].apply(assign_priority)

        # Save batch predictions to SQLite
        conn = sqlite3.connect("lead_scoring.db")
        cursor = conn.cursor()

        lead_ids = []

        for _, row in batch_data.iterrows():
            cursor.execute("""
                INSERT INTO predictions
                (age, job, lead_score, priority, recommendation)
                VALUES (?, ?, ?, ?, ?)
            """, (
                int(row["age"]),
                str(row["job"]),
                float(row["lead_score"]),
                str(row["priority"]),
                "Batch prediction"
            ))

            prediction_id = cursor.lastrowid
            lead_ids.append(f"L{prediction_id:03d}")

        conn.commit()
        conn.close()

        batch_data["lead_id"] = lead_ids

        # Highest scoring leads first
        batch_data = batch_data.sort_values(
            by="lead_score",
            ascending=False
        )

        # Convert to records for HTML
        results = batch_data.to_dict(
            orient="records"
        )

        return render_template(
            "batch_results.html",
            results=results
        )

    except pd.errors.EmptyDataError:
        return "The uploaded CSV file is empty. Please upload a CSV file containing lead data."

    except ValueError:
        return "Invalid data detected. Please check that numerical fields contain valid numbers."

    except Exception as error:
        return f"Error processing CSV: {error}"

# -------------------------------------------------
# Prediction Route
# -------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Collect customer information
    customer_data = {
        "age": int(request.form["age"]),
        "job": request.form["job"],
        "marital": request.form["marital"],
        "education": request.form["education"],
        "default": request.form["default"],
        "balance": int(request.form["balance"]),
        "housing": request.form["housing"],
        "loan": request.form["loan"],
        "contact": request.form["contact"],
        "day": int(request.form["day"]),
        "month": request.form["month"],
        "campaign": int(request.form["campaign"]),
        "pdays": int(request.form["pdays"]),
        "previous": int(request.form["previous"]),
        "poutcome": request.form["poutcome"]
    }

    # Convert customer data to DataFrame
    customer_df = pd.DataFrame([customer_data])


    # -------------------------------------------------
    # Generate Lead Score
    # -------------------------------------------------

    probability = model.predict_proba(customer_df)[0, 1]

    lead_score = round(probability * 100, 1)


    # -------------------------------------------------
    # Assign Lead Priority
    # -------------------------------------------------

    if lead_score >= 70:
        priority = "High"
    elif lead_score >= 40:
        priority = "Medium"
    else:
        priority = "Low"


    # -------------------------------------------------
    # Generate SHAP Explanation
    # -------------------------------------------------

    customer_processed = preprocessor.transform(customer_df)

    feature_names = preprocessor.get_feature_names_out()

    shap_values = explainer.shap_values(customer_processed)

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values[0]
    })

    shap_df["abs_shap"] = shap_df["shap_value"].abs()

    shap_df = shap_df.sort_values(
        by="abs_shap",
        ascending=False
    )


    # -------------------------------------------------
    # Create User-Friendly Explanations
    # -------------------------------------------------

    explanations = []

    for _, row in shap_df.iterrows():

        readable_name = format_feature_name(
            row["feature"],
            customer_data
        )

        if readable_name is not None:
            explanations.append({
                "feature": readable_name,
                "shap_value": row["shap_value"]
            })

        if len(explanations) == 5:
            break


    # -------------------------------------------------
    # Generate Sales Recommendation
    # -------------------------------------------------

    recommendation = generate_llm_recommendation(
    priority,
    lead_score,
    explanations

    
    )

    prediction_id = save_prediction(
    customer_data,
    lead_score,
    priority,
    recommendation
    )

    lead_id = f"L{prediction_id:03d}"
    
    # -------------------------------------------------
    # Display Result
    # -------------------------------------------------

    return render_template(
        "result.html",
        lead_id=lead_id,
        lead_score=lead_score,
        priority=priority,
        shap_features=explanations,
        recommendation=recommendation
    )


# -------------------------------------------------
# Initialize Database
# -------------------------------------------------

init_db()

# -------------------------------------------------
# Run Flask Application
# -------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
