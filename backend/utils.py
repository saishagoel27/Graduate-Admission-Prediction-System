import google.generativeai as genai
import json
import re

def score_sop(text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = f"""Rate this Statement of Purpose on 7 criteria (1-5 scale):
1. Clarity & Coherence
2. Grammar & Language Quality  
3. Purpose & Goal Alignment
4. Motivation & Passion
5. Relevance of Background
6. Research Fit
7. Originality & Insight

Return JSON format:
{{"Clarity & Coherence": 4.2, "Grammar & Language Quality": 4.8, ...}}

SOP: {text}"""
    
    try:
        response = model.generate_content([prompt])
        
        # Extract JSON from response
        json_match = re.search(r'\{.*?\}', response.text, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON in response")
        
        scores = json.loads(json_match.group())
        avg_score = sum(scores.values()) / len(scores)
        
        return {
            "scores": scores,
            "average": round(avg_score, 2)
        }
        
    except Exception as e:
        return {
            "scores": {},
            "average": 0,
            "error": str(e)
        }