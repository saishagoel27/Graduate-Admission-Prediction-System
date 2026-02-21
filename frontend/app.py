import streamlit as st
import requests
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import base64

# Page config
st.set_page_config(
    page_title="Graduate Admission Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)
# Load Gemini API Key from secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    print(f"✅ API Key loaded successfully! Length: {len(GEMINI_API_KEY)}")  # Debug line
except Exception as e:
    st.error("⚠️ Gemini API key not found in secrets.toml")
    st.info("Please create a .streamlit/secrets.toml file with your GEMINI_API_KEY")
    GEMINI_API_KEY = None
    print(f"❌ Error loading API key: {e}")  


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FFD700 0%, #FF8C00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .success-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #ff6b6b;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        color: #2c3e50;
        font-weight: 500;
    }
    
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .navigation-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        border: none;
        font-weight: bold;
        margin: 0.25rem;
        cursor: pointer;
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
        color: #222 !important;
    }
    .stSelectbox label {
        color: #fff !important;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .stNumberInput > div > div > input {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #f8f9fa;
        border-radius: 10px;
        color: #222 !important;
    }
    /* Make selectbox label (University Rating) white and bold */
    label[for^="uni_rating"] {
        color: #fff !important;
        font-weight: bold;
        font-size: 1.1rem;
    }
    /* Make the main header a bright gold gradient for contrast */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FFD700 0%, #FF8C00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None

# Sidebar Navigation
st.sidebar.markdown("## 🧭 Navigation")

# Navigation buttons
if st.sidebar.button("🏠 Main Dashboard", key="nav_main"):
    st.session_state.page = 'main'

if st.sidebar.button("📊 SHAP Analysis", key="nav_shap"):
    if st.session_state.prediction_data:
        st.session_state.page = 'shap'
    else:
        st.sidebar.error("Run prediction first!")

if st.sidebar.button("📝 SOP Analysis", key="nav_sop"):
    if st.session_state.prediction_data:
        st.session_state.page = 'sop'
    else:
        st.sidebar.error("Run prediction first!")

if st.sidebar.button("💡 Recommendations", key="nav_recommendations"):
    if st.session_state.prediction_data:
        st.session_state.page = 'recommendations'
    else:
        st.sidebar.error("Run prediction first!")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Quick Guide")
st.sidebar.info("""
**Steps:**
1. Fill your academic profile
2. Enter university details  
3. Add your SOP text
4. Click 'Predict' 
5. Explore detailed analysis using navigation buttons
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Score Ranges")
st.sidebar.markdown("""
**GRE:** 260-340  
**TOEFL:** 0-120  
**CGPA:** 0-10  
**University Rating:** 1-5  
**SOP/LOR Quality:** 1-5
""")

# Main content based on selected page
if st.session_state.page == 'main':
    # Header
    st.markdown('<div class="main-header">🎓 Graduate Admission Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered admission probability with detailed insights</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📚 Academic Profile")
        
        with st.container():
            col_gre, col_toefl = st.columns(2)
            with col_gre:
                gre = st.number_input("GRE Score", 260, 340, 320, help="Graduate Record Examination score")
            with col_toefl:
                toefl = st.number_input("TOEFL Score", 0, 120, 110, help="Test of English as Foreign Language")
            
            cgpa = st.number_input("CGPA (out of 10)", 0.0, 10.0, 8.5, 0.1, help="Cumulative Grade Point Average")
        
        st.markdown("### 🏫 University & Application")
        
        # University lookup with better UX
        uni_name = st.text_input("🔍 University Name", placeholder="e.g., Stanford University, Princeton University")
        
        col_lookup, col_rating = st.columns([1, 1])
        with col_lookup:
            if st.button("🔍 Get University Rating", type="secondary"):
                if uni_name:
                    with st.spinner("Looking up university..."):
                        try:
                            resp = requests.get("http://localhost:8000/university", params={"name": uni_name})
                            data = resp.json()
                            if data['found']:
                                st.session_state.uni_rating = data['rating']
                                st.success(f"✅ Found: {data['name']}\n📊 Rating: {data['rating']}/5\n🌍 Country: {data['country']}")
                            else:
                                st.warning("⚠️ University not found in database")
                                st.session_state.uni_rating = 1
                        except:
                            st.error("❌ Failed to lookup university. Check if backend is running.")
                else:
                    st.warning("Please enter university name first")
        with col_rating:
            university_rating = st.selectbox(
                "University Rating",
                [1, 2, 3, 4, 5],
                key="uni_rating",
                help="1=Low ranked, 5=Top tier"
            )
        
        col_sop, col_lor = st.columns(2)
        with col_sop:
            sop_rating = st.slider("SOP Quality", 1.0, 5.0, 3.5, 0.1, help="Statement of Purpose strength")
        with col_lor:
            lor_rating = st.slider("LOR Quality", 1.0, 5.0, 4.0, 0.1, help="Letter of Recommendation strength")
        has_research = st.checkbox("🔬 Research Experience", help="Do you have research publications or experience?")
        
        st.markdown("### ✍️ Statement of Purpose")
        
        sop_text = st.text_area(
            "Paste your SOP here", 
            height=150, 
            placeholder="Enter your statement of purpose text here...",
            help="This will be analyzed for quality metrics"
        )
        
        uploaded_file = st.file_uploader("📁 Or upload SOP file", type=['txt'])
        if uploaded_file:
            sop_text = uploaded_file.read().decode('utf-8')
            st.success("✅ SOP loaded from file")
    
    with col2:
        st.markdown("### 🎯 Prediction Results")
        
        if st.button("🚀 Predict My Admission Chances", type="primary", use_container_width=True):
            if not sop_text.strip():
                st.error("⚠️ Please provide your SOP text for complete analysis")
            else:
                with st.spinner("🤖 AI is analyzing your profile..."):
                    profile_data = {
                        "gre_score": gre,
                        "toefl_score": toefl, 
                        "university_rating": university_rating,
                        "sop": sop_rating,
                        "lor": lor_rating,
                        "cgpa": cgpa,
                        "research": int(has_research)
                    }
                    
                    try:
                        pred_resp = requests.post("http://localhost:8000/predict", json=profile_data)
                        prediction = pred_resp.json()
                        
                        exp_resp = requests.post("http://localhost:8000/explain", json=profile_data)  
                        explanation = exp_resp.json()
                        
                        # Passing API key with SOP request
                        if GEMINI_API_KEY:
                            sop_resp = requests.post("http://localhost:8000/sop", json={
                                "sop": sop_text,
                                "api_key": GEMINI_API_KEY
                            })
                            sop_scores = sop_resp.json()
                        else:
                            st.error("⚠️ Cannot analyze SOP - Gemini API key not configured")
                            sop_scores = {"scores": {}, "average": 0}
                        
                        st.session_state.prediction_data = {
                            'prediction': prediction,
                            'explanation': explanation,
                            'sop_scores': sop_scores,
                            'profile_data': profile_data,
                            'sop_text': sop_text
                        }
                        prob = prediction['probability']
                        
                        if prob >= 70:
                            st.markdown(f'<div class="success-card"><h2>🎉 Excellent Chances!</h2><h1>{prob}%</h1><p>Your profile is highly competitive</p></div>', unsafe_allow_html=True)
                        elif prob >= 50:
                            st.markdown(f'<div class="warning-card"><h2>📊 Good Chances</h2><h1>{prob}%</h1><p>Solid profile with room for improvement</p></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="info-card"><h2>📈 Building Phase</h2><h1>{prob}%</h1><p>Great potential, focus on key areas</p></div>', unsafe_allow_html=True)
                        
                        # Quick overview
                        st.markdown("### 📈 Quick Overview")
                        
                        col_base, col_sop_score = st.columns(2)
                        
                        with col_base:
                            st.markdown(
                                f'<div style="background: #f8f9fa; color: #222; padding: 2rem 1rem; border-radius: 15px; text-align: center; font-weight: 600; box-shadow: 0 2px 10px rgba(0,0,0,0.04);">'
                                f'<div style="font-size: 1.2rem; margin-bottom: 0.5rem;">Base Score</div>'
                                f'<div style="font-size: 2.2rem;">{explanation["base_score"]}%</div>'
                                f'<div style="font-size: 1rem; color: #666; margin-top: 0.5rem;">Model baseline output</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        
                        with col_sop_score:
                            st.markdown(
                                f'<div style="background: #f8f9fa; color: #222; padding: 2rem 1rem; border-radius: 15px; text-align: center; font-weight: 600; box-shadow: 0 2px 10px rgba(0,0,0,0.04);">'
                                f'<div style="font-size: 1.2rem; margin-bottom: 0.5rem;">SOP Quality</div>'
                                f'<div style="font-size: 2.2rem;">{sop_scores["average"]}/5</div>'
                                f'<div style="font-size: 1rem; color: #666; margin-top: 0.5rem;">AI-evaluated score</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Navigation prompt
                        st.markdown("---")
                        st.info("🧭 **Explore Detailed Analysis:** Use the navigation buttons in the sidebar to view SHAP analysis, SOP breakdown, and personalized recommendations!")
                        
                        # Download option
                        if st.button("📄 Generate Full Report", type="secondary"):
                            report = (st.session_state.prediction_data)
                            b64 = base64.b64encode(report.encode()).decode()
                            href = f'<a href="data:file/txt;base64,{b64}" download="admission_analysis_report.txt" target="_blank">📥 Download Complete Report</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("💡 Make sure the backend server is running on localhost:8000")
        
        # Helper function for report generation
        def generate_report(data):
            prediction = data['prediction']
            explanation = data['explanation']
            sop_scores = data['sop_scores']
            profile_data = data['profile_data']
            
            report = f"""
GRADUATE ADMISSION PREDICTION REPORT
Generated by AI-Powered Admission Predictor
=============================================

OVERALL RESULT:
Admission Probability: {prediction['probability']}%

PROFILE SUMMARY:
- GRE Score: {profile_data['gre_score']}
- TOEFL Score: {profile_data['toefl_score']}
- CGPA: {profile_data['cgpa']}/10
- University Rating: {profile_data['university_rating']}/5
- SOP Quality: {profile_data['sop']}/5
- LOR Quality: {profile_data['lor']}/5
- Research Experience: {'Yes' if profile_data['research'] else 'No'}

FEATURE CONTRIBUTIONS (SHAP Analysis):
Base Score: {explanation['base_score']}%
Final Score: {explanation['final_score']}%

Individual Feature Impacts:
"""
            
            for feature, impact in explanation['contributions'].items():
                report += f"- {feature}: {impact:+.4f}\n"
            
            report += f"\nRECOMMENDations:\n"
            for rec in explanation['suggestions']:
                report += f"• {rec}\n"
            
            if sop_scores.get('scores'):
                report += f"\nSOP ANALYSIS:\n"
                report += f"Overall SOP Score: {sop_scores['average']}/5\n\n"
                report += "Individual Criteria:\n"
                for criterion, score in sop_scores['scores'].items():
                    report += f"- {criterion}: {score}/5\n"
            
            report += f"\n\nGenerated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
            report += f"\nThis report is for guidance only. Actual admission decisions depend on many factors."
            
            return report
                
        # Show placeholder if no prediction yet
        if not st.session_state.prediction_data:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 15px; border: 2px dashed #dee2e6; color: #222;">
                <h3 style="color: #222;">🎯 Ready for Analysis</h3>
                <p style="color: #222;">Fill in your details and click 'Predict' to see your admission probability and detailed insights!</p>
            </div>
            """, unsafe_allow_html=True)

elif st.session_state.page == 'shap':
    data = st.session_state.prediction_data
    
    st.markdown('<div class="main-header">📊 SHAP Feature Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Understanding what drives your admission probability</div>', unsafe_allow_html=True)
    
    if data:
        explanation = data['explanation']
        contribs = explanation['contributions']
        
        # Create interactive plotly chart
        features = list(contribs.keys())
        values = list(contribs.values())
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
        
        fig = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker_color=colors,
            text=[f'{v:+.3f}' for v in values],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Feature Impact on Admission Probability",
            xaxis_title="SHAP Value (Impact on Probability)",
            yaxis_title="Features",
            height=500,
            template="plotly_white"
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation
        st.markdown("### 🔍 What This Means")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Positive Contributors")
            positive_features = {k: v for k, v in contribs.items() if v > 0}
            if positive_features:
                for feature, impact in sorted(positive_features.items(), key=lambda x: x[1], reverse=True):
                    st.success(f"**{feature}**: +{impact:.3f} boost")
            else:
                st.info("No strongly positive features found")
        
        with col2:
            st.markdown("#### ⚠️ Negative Contributors") 
            negative_features = {k: v for k, v in contribs.items() if v < 0}
            if negative_features:
                for feature, impact in sorted(negative_features.items(), key=lambda x: x[1]):
                    st.error(f"**{feature}**: {impact:.3f} reduction")
            else:
                st.success("No negative contributors - great profile!")
        
        # Base vs Final
        st.markdown("### 📈 Score Breakdown")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Base Score", f"{explanation['base_score']}%", help="Model's baseline prediction")
        
        with col2:
            total_impact = sum(contribs.values())
            st.metric("Feature Impact", f"{total_impact:+.2f}%", help="Net impact of all features")
        
        with col3:
            st.metric("Final Score", f"{explanation['final_score']}%", help="Final admission probability")

elif st.session_state.page == 'sop':
    data = st.session_state.prediction_data
    
    st.markdown('<div class="main-header">📝 SOP Quality Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered evaluation of your Statement of Purpose</div>', unsafe_allow_html=True)
    
    if data:
        sop_scores = data['sop_scores']
        
        if sop_scores.get('scores'):
            scores = sop_scores['scores']
            avg = sop_scores['average']
            
            # Overall score display
            if avg >= 4:
                st.markdown(f'<div class="success-card"><h2>🌟 Excellent SOP!</h2><h1>{avg}/5</h1><p>Your statement of purpose is highly compelling</p></div>', unsafe_allow_html=True)
            elif avg >= 3:
                st.markdown(f'<div class="warning-card"><h2>📊 Good SOP</h2><h1>{avg}/5</h1><p>Solid foundation with room for enhancement</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="info-card"><h2>📈 Needs Improvement</h2><h1>{avg}/5</h1><p>Focus on strengthening key areas</p></div>', unsafe_allow_html=True)
            
            # Detailed breakdown
            st.markdown("### 📊 Detailed Breakdown")
            
            # Create radar chart
            categories = list(scores.keys())
            values = list(scores.values())
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Your SOP',
                line_color='#667eea'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5]
                    )),
                showlegend=False,
                title="SOP Quality Radar Chart",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Individual metrics
            st.markdown("### 📋 Individual Scores")
            
            cols = st.columns(2)
            for i, (criterion, score) in enumerate(scores.items()):
                with cols[i % 2]:
                    if score >= 4:
                        st.success(f"**{criterion}**: {score}/5 ✨")
                    elif score >= 3:
                        st.warning(f"**{criterion}**: {score}/5 📊")
                    else:
                        st.error(f"**{criterion}**: {score}/5 ⚠️")
            
            # Improvement suggestions
            st.markdown("### 💡 Improvement Areas")
            
            weak_areas = {k: v for k, v in scores.items() if v < 3.5}
            if weak_areas:
                for area, score in weak_areas.items():
                    if area == "Clarity & Coherence":
                        st.info("🎯 **Clarity & Coherence**: Structure your ideas more logically. Use clear transitions between paragraphs.")
                    elif area == "Grammar & Language Quality":
                        st.info("📝 **Grammar & Language**: Proofread carefully. Consider using tools like Grammarly or having someone review it.")
                    elif area == "Purpose & Goal Alignment":
                        st.info("🎯 **Purpose & Goals**: Be more specific about your career objectives and how this program fits.")
                    elif area == "Motivation & Passion":
                        st.info("🔥 **Motivation & Passion**: Show more enthusiasm! Share personal stories that demonstrate your commitment.")
                    elif area == "Relevance of Background":
                        st.info("🎓 **Background Relevance**: Better connect your past experiences to your future goals.")
                    elif area == "Research Fit":
                        st.info("🔬 **Research Fit**: Research faculty and programs more thoroughly. Show specific alignment.")
                    elif area == "Originality & Insight":
                        st.info("💡 **Originality & Insight**: Add unique perspectives or experiences that set you apart.")
            else:
                st.success("🎉 Great job! All areas are performing well. Keep up the excellent work!")
        
        else:
            st.error("❌ SOP analysis failed. Please try again or check your SOP text.")

elif st.session_state.page == 'recommendations':
    data = st.session_state.prediction_data
    
    st.markdown('<div class="main-header">💡 Personalized Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Actionable insights to boost your admission chances</div>', unsafe_allow_html=True)
    
    if data:
        explanation = data['explanation']
        suggestions = explanation['suggestions']
        contribs = explanation['contributions']
        
        # Priority recommendations
        st.markdown("### 🚨 Priority Actions")
        
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                st.markdown(f'<div class="recommendation-card">💡 <strong>#{i}:</strong> {suggestion}</div>', unsafe_allow_html=True)
        else:
            st.success("🎉 Your profile looks strong! No major improvements needed.")
        
        # Detailed improvement roadmap
        st.markdown("### 🗺️ Improvement Roadmap")
        
        # Sort features by negative impact
        negative_contribs = {k: v for k, v in contribs.items() if v < -0.02}
        
        if negative_contribs:
            sorted_negative = sorted(negative_contribs.items(), key=lambda x: x[1])
            
            for feature, impact in sorted_negative:
                with st.expander(f"🎯 Improve {feature} (Impact: {impact:.3f})"):
                    if feature == "GRE":
                        st.markdown("""
                        **📚 GRE Improvement Strategy:**
                        - Take practice tests to identify weak areas
                        - Focus on vocabulary building (use apps like Magoosh)
                        - Practice quantitative reasoning daily
                        - Consider a prep course if self-study isn't working
                        - Retake if current score is below 315
                        """)
                    elif feature == "TOEFL":
                        st.markdown("""
                        **🗣️ TOEFL Enhancement Plan:**
                        - Practice speaking English daily
                        - Watch English movies/podcasts without subtitles
                        - Take mock tests to improve timing
                        - Focus on writing structured essays
                        - Consider language exchange programs
                        """)
                    elif feature == "CGPA":
                        st.markdown("""
                        **📈 Academic Performance:**
                        - If still in school: Focus on remaining courses
                        - Consider additional coursework to boost GPA
                        - Highlight upward grade trends in applications
                        - Get strong LORs to explain any low grades
                        - Consider post-bacc programs if needed
                        """)
                    elif feature == "University":
                        st.markdown("""
                        **🏫 University Strategy:**
                        - Apply to a mix of reach, match, and safety schools
                        - Research faculty alignment carefully
                        - Consider rankings in your specific field
                        - Look into up-and-coming programs
                        - Don't just focus on overall rankings
                        """)
                    elif feature == "Research":
                        st.markdown("""
                        **🔬 Research Experience:**
                        - Reach out to professors for research opportunities
                        - Consider summer research programs (REUs)
                        - Volunteer in labs even without pay
                        - Attend conferences and present work
                        - Publish papers or technical reports
                        """)
                    elif feature == "SOP":
                        st.markdown("""
                        **✍️ SOP Enhancement:**
                        - Get feedback from multiple people
                        - Show, don't just tell (use specific examples)
                        - Connect your story cohesively
                        - Research the program thoroughly
                        - Tailor your SOP for each application
                        """)
                    elif feature == "LOR":
                        st.markdown("""
                        **📄 LOR Strengthening:**
                        - Choose recommenders who know you well
                        - Provide them with your CV and achievements
                        - Ask for letters well in advance
                        - Follow up with a thank-you note
                        - Consider additional letters if allowed
                        """)
                    # General advice for all features
                    st.markdown(f"""
                    **🔑 General Improvement Tips:**
                    - Start early to avoid last-minute stress
                    - Seek help from mentors or advisors
                    - Use online resources and forums
                    - Stay organized and keep track of deadlines
                    - Maintain a positive and proactive attitude
                    """)
        else:
            st.success("🎉 No critical improvements identified. Maintain your strengths!")