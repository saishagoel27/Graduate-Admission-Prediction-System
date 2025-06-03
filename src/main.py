from fastapi import FastAPI, Query, HTTPException
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Union
import requests
import json
import re
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("admission_model.pkl")
scaler = joblib.load("scaler.pkl") 

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
    admission_prob = prediction * 100  # convert to percentage

    if admission_prob < 0:
        return {"message": "Admission unlikely: predicted score is negative."}
    else:
        return {"admission_probability": round(float(admission_prob), 2)}

df = pd.read_excel("UpdatedWorldUniRank23.xlsx")

def safe(val):
    return "Unavailable" if pd.isna(val) or val in [None, ""] else val

def safe(value):
    if pd.isna(value):
        return "Unavailable"
    if isinstance(value, (np.integer, np.floating)):
        return value.item()  # Convert numpy types to native Python types
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
    sop_text: str

class SOPScoreResponse(BaseModel):
    individual_scores: Dict[str, float]
    average_score: float

def build_prompt(sop_text: str) -> str:
    return f"""
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
\"\"\"
{sop_text}
\"\"\"
"""

OLLAMA_URL = "http://localhost:11434/api/generate"

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
\"\"\"{data.sop}\"\"\"
"""

    try:
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        })
        res.raise_for_status()
        output = res.json()["response"]

        # Extract JSON object from the response
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM response.")

        scores_dict = json.loads(match.group())
        
        # Fix any known misspellings or inconsistent keys if needed
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
