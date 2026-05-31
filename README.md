# Graduate Admission Prediction System

Predicts your chances of getting into grad school using machine learning. Also analyzes your Statement of Purpose and tells you what to improve.

## What it does

- Predicts admission probability based on GRE, TOEFL, CGPA, etc.
- Shows which factors help or hurt your chances (SHAP analysis)
- Evaluates your SOP using AI and gives scores on 7 criteria
- Suggests improvements for weak areas
- Auto-fills university ratings from world rankings

## Tech used

- **Frontend**: Streamlit
- **Backend**: FastAPI  
- **ML Model**: Trained on admission dataset
- **Explainability**: SHAP
- **SOP Analysis**: Groq API (LLaMA models)

## Setup

### Prerequisites

- Python 3.8+
- Groq API key (free at https://console.groq.com)

### Installation

1. Clone the repo
```bash
git clone https://github.com/saishagoel27/Graduate-Admission-Prediction-System
cd Graduate-Admission-Prediction-System
```

2. Create virtual environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Mac/Linux  
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Add your Groq API key

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your-api-key-here"
```

### Running the app

**Option 1: Using the script (Git Bash/Linux/Mac)**
```bash
chmod +x run_all.sh
./run_all.sh
```

**Option 2: Manual (Windows/any OS)**

Terminal 1 - Backend:
```bash
uvicorn backend.main:app --reload
```

Terminal 2 - Frontend:
```bash
streamlit run frontend/app.py
```

Open http://localhost:8501 in your browser.

## How to use

1. Fill in your academic details (GRE, TOEFL, CGPA)
2. Enter university name (optional - auto-fills rating)
3. Paste your Statement of Purpose
4. Click "Predict My Admission Chances"
5. Check the sidebar for detailed analysis:
   - SHAP Analysis: see what's helping/hurting
   - SOP Analysis: get scores on clarity, grammar, etc.
   - Recommendations: actionable tips to improve

## Project structure

```
.
├── backend/
│   ├── main.py              # FastAPI routes
│   ├── utils.py             # SOP scoring logic
│   └── models/              # Trained ML models
├── frontend/
│   └── app.py               # Streamlit UI
├── data/
│   ├── admission_data.csv   # Training data
│   └── UpdatedWorldUniRank23.xlsx
├── notebooks/
│   └── admission.ipynb      # Model training notebook
├── .streamlit/
│   └── secrets.toml         # API keys (don't commit!)
├── requirements.txt
└── run_all.sh
```

## Demo

https://github.com/user-attachments/assets/7c1ad5bc-b2e8-444e-835c-8b421d905c67

## Notes

- The SOP analysis uses Groq's LLaMA models (free tier available)
- Predictions are based on historical data - actual results may vary
- University ratings are from 2023 world rankings

## Troubleshooting

**Backend won't start:**
- Make sure port 8000 isn't in use
- Check if all dependencies installed correctly

**SOP analysis fails:**
- Verify your Groq API key is correct
- Check your internet connection
- Free tier has rate limits - wait a bit and retry

**Frontend can't connect to backend:**
- Ensure backend is running on port 8000
- Check firewall settings

## License

MIT License - do whatever you want with it

## Authors

Built by Anishaa and Saisha for NTCC In-House Practical
