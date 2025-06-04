from fastapi import FastAPI, Query, HTTPException
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Union
import requests
import json
import re
import joblib
import numpy as np
import shap
from pathlib import Path

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "admission_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

# Load SHAP explainer once
explainer = shap.Explainer(model.predict, masker=shap.maskers.Independent(data=np.zeros((1, 7))))

class AdmissionInput(BaseModel):
    gre_score: float
    toefl_score: float
    university_rating: int
    sop: float
    lor: float
    cgpa: float
    research: int

@app.post("/predict_admission")
def predict_admission(input: AdmissionInput) -> Union[Dict[str, float], Dict[str, str]]:
    data = np.array([[
        input.gre_score,
        input.toefl_score,
        input.university_rating,
        input.sop,
        input.lor,
        input.cgpa,
        input.research
    ]])

    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)[0]
    admission_prob = prediction * 100

    if admission_prob < 0:
        return {"message": "Admission unlikely: predicted score is negative."}
    else:
        return {"admission_probability": round(float(admission_prob), 2)}

@app.post("/explain_prediction")
def explain_prediction(input: AdmissionInput):
    data = np.array([[
        input.gre_score, input.toefl_score, input.university_rating,
        input.sop, input.lor, input.cgpa, input.research
    ]])

    scaled_data = scaler.transform(data)
    shap_values = explainer(scaled_data)

    feature_names = ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]
    shap_value_array = shap_values.values[0]
    base_value = shap_values.base_values[0]
    final_prediction = model.predict(scaled_data)[0]

    feature_contributions = dict(zip(feature_names, shap_value_array))

    recommendations = []
    for feature, value in feature_contributions.items():
        if value < -0.05:
            if feature == "GRE Score":
                recommendations.append("Consider retaking the GRE to improve your score.")
            elif feature == "TOEFL Score":
                recommendations.append("Improve your TOEFL score to boost your chances.")
            elif feature == "University Rating":
                recommendations.append("Applying to better-rated universities could help.")
            elif feature == "SOP":
                recommendations.append("Enhance the clarity or relevance of your SOP.")
            elif feature == "LOR":
                recommendations.append("Stronger Letters of Recommendation may help.")
            elif feature == "CGPA":
                recommendations.append("A higher CGPA could significantly improve your profile.")
            elif feature == "Research":
                recommendations.append("Gaining research experience can positively influence your chances.")

    return {
        "base_prediction": round(float(base_value * 100), 2),
        "final_prediction": round(float(final_prediction * 100), 2),
        "shap_values": feature_contributions,
        "recommendations": recommendations or ["Your profile looks strong based on current data."]
    }

df = pd.read_excel(BASE_DIR / "data" / "UpdatedWorldUniRank23.xlsx")

def safe(value):
    if pd.isna(value):
        return "Unavailable"
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)

@app.get("/lookup")
def lookup_university(name: str = Query(...)):
    result = df[df['University Name'].str.lower() == name.lower()]

    if not result.empty:
        row = result.iloc[0]
        country = safe(row.get("Country"))

        try:
            rank_str = str(row["Rank"])
            if "-" in rank_str:
                low, high = map(int, rank_str.split("-"))
                avg_rank = (low + high) / 2
            else:
                avg_rank = int(rank_str)

            if avg_rank <= 100:
                rating = 5
            elif avg_rank <= 250:
                rating = 4
            elif avg_rank <= 500:
                rating = 3
            else:
                rating = 2

        except Exception:
            rating = 1

        return {
            "University": safe(name),
            "Country": country,
            "Rating (out of 5)": int(rating),
            "Details": {
                "Rank": safe(row.get("Rank")),
                "No of Students": safe(row.get("No of student")),
                "Students per Staff": safe(row.get("No of student per staff")),
                "International Students": safe(row.get("International Student")),
                "Female:Male Ratio": safe(row.get("Female:Male Ratio")),
                "Overall Score": safe(row.get("OverAll Score")),
                "Teaching Score": safe(row.get("Teaching Score")),
                "Research Score": safe(row.get("Research Score")),
                "Citations Score": safe(row.get("Citations Score")),
                "Industry Income Score": safe(row.get("Industry Income Score")),
                "International Outlook Score": safe(row.get("International Outlook Score"))
            }
        }
    else:
        return {
            "University": name,
            "Country": "Unavailable",
            "Rating (out of 5)": 1,
            "Details": {
                "Note": "University not found in database. Assigned default lowest rating."
            }
        }

class SOPRequest(BaseModel):
    sop: str

@app.post("/score_sop")
def score_sop(data: SOPRequest):
    prompt = f"""
You are an expert admission reviewer. Evaluate the following Statement of Purpose based on these 7 metrics (each out of 5):

1. Clarity & Coherence
2. Grammar & Language Quality
3. Purpose & Goal Alignment
4. Motivation & Passion
5. Relevance of Background
6. Research Fit
7. Originality & Insight

Respond ONLY in JSON format with the metric names as keys and their scores as values. For example:

{{
  "Clarity & Coherence": 4.5,
  "Grammar & Language Quality": 5,
  "Purpose & Goal Alignment": 4,
  "Motivation & Passion": 4.5,
  "Relevance of Background": 4,
  "Research Fit": 3.5,
  "Originality & Insight": 4
}}

SOP:
'''{data.sop}'''
"""

    try:
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        })
        res.raise_for_status()
        output = res.json()["response"]

        # Use a non-greedy regex to extract the first JSON object
        match = re.search(r"\{.*?\}", output, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM response. Response was: " + output)

        try:
            scores_dict = json.loads(match.group())
        except json.JSONDecodeError as json_err:
            raise ValueError(f"Failed to parse JSON from LLM response: {json_err}. Response was: {output}")

        corrected_keys = {
            "Clairty & Coherenc": "Clarity & Coherence",
            "Relevance to subject": "Relevance of Background",
            "Grammatical Accuracy": "Grammar & Language Quality",
            "Originality": "Originality & Insight"
        }

        final_scores = {}
        for key, value in scores_dict.items():
            fixed_key = corrected_keys.get(key.strip(), key.strip())
            final_scores[fixed_key] = float(value)

        average_score = round(sum(final_scores.values()) / len(final_scores), 2)

        return {
            "individual_scores": final_scores,
            "average_score": average_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring SOP: {e}")