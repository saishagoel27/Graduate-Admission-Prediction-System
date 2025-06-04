import streamlit as st
import requests
import json

st.set_page_config(page_title="🎓 Graduate Admission Predictor", layout="wide")

st.markdown("""
    <style>
    .big-font {
        font-size: 36px !important;
        font-weight: bold;
    }
    .sub-font {
        font-size: 18px;
        color: #666;
    }
    .recommendation-box {
        background-color: #f0f2f6;
        padding: 12px;
        border-left: 5px solid #2b8cbe;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-font'>Graduate Admission Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-font'>Get insights, advice, and your estimated chances of getting into your dream grad school.</div>", unsafe_allow_html=True)
st.markdown("---")

# Check if backend is running
def check_backend():
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        return True
    except:
        return False

if not check_backend():
    st.error("🚨 Backend server is not running! Please start the FastAPI server first.")
    st.code("uvicorn main:app --reload --port 8000")
    st.stop()

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📄 Applicant Profile")
    gre = st.slider("GRE Score", 260, 340, 320)
    toefl = st.slider("TOEFL Score", 0, 120, 110)

    st.subheader("🏫 University Info")
    uni_name = st.text_input("Enter University Name to Auto-Fill Rating")
    if st.button("🔍 Lookup University") and uni_name:
        try:
            response = requests.get("http://localhost:8000/lookup", params={"name": uni_name}, timeout=10)
            result = response.json()
            university_rating = result.get("Rating (out of 5)", 3)
            st.session_state.uni_rating = university_rating
            st.success(f"{uni_name} found! Rating: {university_rating}/5")
        except requests.exceptions.RequestException as e:
            st.error(f"Lookup failed: {e}")

    university_rating = st.slider("University Rating (1-5)", 1, 5, st.session_state.get("uni_rating", 3))

    sop = st.slider("SOP Strength (1-5)", 1.0, 5.0, 3.5, step=0.5)
    lor = st.slider("LOR Strength (1-5)", 1.0, 5.0, 4.0, step=0.5)
    cgpa = st.slider("CGPA (out of 10)", 0.0, 10.0, 8.6, step=0.1)
    research = st.radio("Research Experience", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    st.subheader("✍️ Statement of Purpose")
    sop_text = st.text_area("Paste your SOP (1 paragraph)", height=200)

    if st.button("🔍 Predict My Admission Chance"):
        if not sop_text.strip():
            st.warning("Please enter your Statement of Purpose!")
        else:
            with st.spinner("Running prediction..."):
                input_data = {
                    "gre_score": gre,
                    "toefl_score": toefl,
                    "university_rating": university_rating,
                    "sop": sop,
                    "lor": lor,
                    "cgpa": cgpa,
                    "research": research
                }

                try:
                    # Make API calls with timeout
                    pred = requests.post("http://localhost:8000/predict_admission", 
                                       json=input_data, timeout=30).json()
                    explain = requests.post("http://localhost:8000/explain_prediction", 
                                          json=input_data, timeout=30).json()
                    sop_eval = requests.post("http://localhost:8000/score_sop", 
                                           json={"sop": sop_text}, timeout=60).json()

                    st.session_state.prediction = pred
                    st.session_state.explain = explain
                    st.session_state.sop_eval = sop_eval
                    st.rerun()  # Fixed: replaced st.experimental_rerun()

                except requests.exceptions.Timeout:
                    st.error("❌ Request timed out. The server might be overloaded.")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to the backend server.")
                except Exception as e:
                    st.error(f"❌ Error occurred: {e}")

with col2:
    if "prediction" in st.session_state:
        # Display prediction results
        if "admission_probability" in st.session_state.prediction:
            st.success(f"🎯 Estimated Admission Probability: {st.session_state.prediction['admission_probability']}%")
        else:
            st.warning(st.session_state.prediction.get("message", "No prediction available"))

        # Display SHAP explanations
        if "explain" in st.session_state and "shap_values" in st.session_state.explain:
            st.subheader("📊 Feature Contributions (SHAP)")
            shap_values = st.session_state.explain["shap_values"]
            sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
            for feat, val in sorted_shap:
                color = "🟢" if val > 0 else "🔴" if val < 0 else "⚪"
                st.markdown(f"{color} **{feat}**: {val:+.3f}")

        # Display recommendations
        if "explain" in st.session_state and "recommendations" in st.session_state.explain:
            st.subheader("✅ Smart Recommendations")
            for rec in st.session_state.explain["recommendations"]:
                st.markdown(f"<div class='recommendation-box'>{rec}</div>", unsafe_allow_html=True)

        # Display SOP analysis
        if "sop_eval" in st.session_state:
            st.subheader("📝 SOP Quality Analysis")
            scores = st.session_state.sop_eval["individual_scores"]
            avg = st.session_state.sop_eval["average_score"]
            
            # Display scores in columns
            cols = st.columns(2)
            for idx, (k, v) in enumerate(scores.items()):
                cols[idx % 2].metric(label=k, value=f"{v}/5")
            
            # Show average with color coding
            if avg >= 4.0:
                st.success(f"🌟 Excellent SOP Score: **{avg} / 5**")
            elif avg >= 3.0:
                st.info(f"👍 Good SOP Score: **{avg} / 5**")
            else:
                st.warning(f"⚠️ SOP Needs Improvement: **{avg} / 5**")
    else:
        st.info("👆 Enter your details and click 'Predict My Admission Chance' to see results!")