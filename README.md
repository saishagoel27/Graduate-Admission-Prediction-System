# 🎓 Graduate Admission Prediction System

A full-stack machine learning web app to predict your chances of getting into a graduate program, explain feature contributions, and evaluate your Statement of Purpose (SOP) using Gemini AI.

---

## 🌟 Features

* 🔮 **Admission Prediction** using a trained ML model
* 🧠 **SHAP Explainability** for feature impact
* 📝 **SOP Evaluation** with Gemini AI (via Google Generative AI)
* 📊 Smart Recommendations to improve your profile
* 🏫 University Rating Auto-Fill based on world rank
* 📄 Downloadable summary report

---

## 🛠 Tech Stack

* **Frontend**: Streamlit
* **Backend**: FastAPI
* **ML Model**: Scikit-learn (Linear Regression)
* **Explainability**: SHAP
* **SOP Evaluation**: Gemini (via Google Generative AI)

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/saishagoel27/GAPS_NTCC
cd GAPS_NTCC
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Set Up Secrets

Create a `secrets.toml` file:

```toml
GEMINI_API_KEY = "your-google-gemini-api-key"
```

> 🔑 Get your API key from: [Google AI Studio](https://makersuite.google.com/app/apikey)

### 5. Run the App

```bash
1. source venv/Scripts/activate
2. export GEMINI_API_KEY="your-key-here"
3. chmod +x run_all.sh
4. ./run_all.sh or bash run_all.sh
```

* FastAPI backend runs on: `http://localhost:8000`
* Streamlit app runs on: `http://localhost:8501`

---

## 📂 Project Structure

```
graduate-admission/
├── backend/           # FastAPI endpoints
│   ├── main.py
│   ├── utils.py       # Gemini SOP scoring, SHAP, preprocess
│   └── models/        # Trained model .pkl files
├── frontend/
│   └── app.py         # Streamlit UI
├── data/              # Admission dataset, university ranks
├── notebooks/         # Jupyter notebook for training
├── run_all.sh         # Script to run both frontend + backend
├── requirements.txt
└── secrets.toml
```

---

## 📸 Project Walkthrough

https://github.com/user-attachments/assets/7c1ad5bc-b2e8-444e-835c-8b421d905c67

---

## 📄 License

Under MIT License - free to use and modify

---

## 🙋‍♂️ Authors

Built by Anishaa and Saisha 
