from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import shap

from backend.utils import score_sop

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "models" / "admission_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

class StudentProfile(BaseModel):
    gre_score: float
    toefl_score: float
    university_rating: int
    sop: float
    lor: float
    cgpa: float
    research: int

class SOPText(BaseModel):
    sop: str

@app.post("/predict")
def predict_admission(profile: StudentProfile):
    features = np.array([[
        profile.gre_score, profile.toefl_score, profile.university_rating,
        profile.sop, profile.lor, profile.cgpa, profile.research
    ]])
    
    scaled_features = scaler.transform(features)
    prob = model.predict(scaled_features)[0] * 100
    
    return {"probability": max(0, round(prob, 2))}

@app.post("/explain")
def explain_prediction(profile: StudentProfile):
    features = np.array([[
        profile.gre_score, profile.toefl_score, profile.university_rating,
        profile.sop, profile.lor, profile.cgpa, profile.research
    ]])
    
    scaled_features = scaler.transform(features)
    explainer = shap.Explainer(model.predict, masker=shap.maskers.Independent(data=np.zeros((1, 7))))
    shap_values = explainer(scaled_features)
    
    feature_names = ["GRE", "TOEFL", "University", "SOP", "LOR", "CGPA", "Research"]
    contributions = dict(zip(feature_names, shap_values.values[0]))
    
    # Generate suggestions for weak areas
    suggestions = []
    for feature, impact in contributions.items():
        if impact < -0.05:
            suggestions.append(get_suggestion(feature))
    
    return {
        "base_score": round(shap_values.base_values[0] * 100, 2),
        "final_score": round(model.predict(scaled_features)[0] * 100, 2),
        "contributions": contributions,
        "suggestions": suggestions or ["Your profile looks competitive!"]
    }

def get_suggestion(feature):
    suggestions = {
        "GRE": "Consider retaking the GRE for a higher score",
        "TOEFL": "Improve your TOEFL score if possible", 
        "University": "Consider applying to higher-ranked universities",
        "SOP": "Strengthen your statement of purpose",
        "LOR": "Seek stronger letters of recommendation",
        "CGPA": "A higher GPA would significantly help",
        "Research": "Try to gain research experience"
    }
    return suggestions.get(feature, "Focus on strengthening this area")

@app.get("/university")
def get_university_rating(name: str = Query(...)):
    df = pd.read_excel(BASE_DIR.parent / "data" / "UpdatedWorldUniRank23.xlsx")
    
    result = df[df['University Name'].str.lower() == name.lower()]
    
    if result.empty:
        return {"name": name, "rating": 1, "found": False}
    
    row = result.iloc[0]
    rank_str = str(row.get("Rank", "1000"))
    
    # Parse rank (handle ranges like "101-150")
    try:
        if "-" in rank_str:
            low, high = map(int, rank_str.split("-"))
            rank = (low + high) // 2
        else:
            rank = int(rank_str)
    except:
        rank = 1000
    
    # Convert rank to rating
    if rank <= 100:
        rating = 5
    elif rank <= 250:
        rating = 4
    elif rank <= 500:
        rating = 3
    else:
        rating = 2
    
    return {
        "name": name,
        "rating": rating,
        "rank": rank_str,
        "country": str(row.get("Country", "Unknown")),
        "found": True
    }

@app.post("/sop")
def evaluate_sop(data: SOPText):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    return score_sop(data.sop, api_key)