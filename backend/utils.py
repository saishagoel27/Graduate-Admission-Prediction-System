import google.generativeai as genai
import json
import re

def score_sop(text, api_key):
    # Validate API key first
    if not api_key or api_key == "":
        raise ValueError("GEMINI_API_KEY is not set or empty")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""Rate this Statement of Purpose on 7 criteria (1-5 scale).
Return ONLY a valid JSON object with no additional text or markdown formatting.

Criteria to rate:
1. Clarity & Coherence
2. Grammar & Language Quality  
3. Purpose & Goal Alignment
4. Motivation & Passion
5. Relevance of Background
6. Research Fit
7. Originality & Insight

Expected JSON format (no markdown, no code blocks):
{{"Clarity & Coherence": 4.2, "Grammar & Language Quality": 4.8, "Purpose & Goal Alignment": 4.5, "Motivation & Passion": 4.0, "Relevance of Background": 3.8, "Research Fit": 4.3, "Originality & Insight": 3.9}}

SOP Text:
{text}"""
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        print(f"Raw Gemini Response: {response_text}")  # Debug logging
        
        # Try to extract JSON - handle markdown code blocks
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try without code blocks
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                raise ValueError(f"No valid JSON found in response: {response_text[:200]}")
        
        scores = json.loads(json_str)
        
        # Validate that we got all 7 scores
        expected_criteria = [
            "Clarity & Coherence",
            "Grammar & Language Quality",
            "Purpose & Goal Alignment",
            "Motivation & Passion",
            "Relevance of Background",
            "Research Fit",
            "Originality & Insight"
        ]
        
        if len(scores) != 7:
            print(f"Warning: Expected 7 scores, got {len(scores)}")
        
        avg_score = sum(scores.values()) / len(scores)
        
        return {
            "scores": scores,
            "average": round(avg_score, 2)
        }
        
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        return {
            "scores": {},
            "average": 0,
            "error": f"Failed to parse JSON: {str(e)}"
        }
    except Exception as e:
        print(f"SOP Scoring Error: {e}")
        return {
            "scores": {},
            "average": 0,
            "error": str(e)
        }