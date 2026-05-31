import json
import re
import requests

def score_sop(text, api_key):
    """Score SOP using Groq API via direct HTTP requests"""
    if not api_key or api_key == "":
        raise ValueError("GROQ_API_KEY is not set or empty")

    try:
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

        # Use direct HTTP requests instead of Groq SDK
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        models_to_try = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
        
        response_text = None
        last_error = None
        
        for model in models_to_try:
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 1024
                }
                
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data["choices"][0]["message"]["content"].strip()
                    if response_text:
                        break
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
            except Exception as e:
                last_error = str(e)
                continue

        if not response_text:
            return {"scores": {}, "average": 0, "error": f"Groq API failed: {last_error}"}

        # Extract JSON from response
        json_match = re.search(r'```json\s*(.+?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1:
                json_str = response_text[start:end+1]
            else:
                json_str = response_text

        # Parse JSON
        try:
            scores = json.loads(json_str)
        except:
            return {"scores": {}, "average": 0, "error": "Failed to parse JSON response"}

        if not isinstance(scores, dict) or not scores:
            return {"scores": {}, "average": 0, "error": "Invalid scores format"}

        try:
            avg_score = sum(float(v) for v in scores.values()) / len(scores)
        except:
            return {"scores": scores, "average": 0, "error": "Score values are not numeric"}

        return {"scores": scores, "average": round(avg_score, 2)}
    
    except Exception as e:
        return {"scores": {}, "average": 0, "error": f"SOP scoring error: {str(e)}"}
