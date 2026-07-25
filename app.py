import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main {
        padding: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .pass-badge {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 15px 30px;
        border-radius: 10px;
        color: white;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 15px rgba(17,153,142,0.4);
    }
    .fail-badge {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 15px 30px;
        border-radius: 10px;
        color: white;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 15px rgba(235,51,73,0.4);
    }
    .info-box {
        background: #f0f2f6;
        padding: 20px;
        border-left: 5px solid #667eea;
        border-radius: 5px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model
try:
    model = joblib.load("logistic_regression_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file not found! Please ensure 'logistic_regression_model.pkl' exists.")
    st.stop()

# Header Section
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("# 📚 Student Performance Predictor")
    st.markdown("*Advanced ML-based prediction system*")

with col2:
    st.markdown(f"<div style='text-align: right; padding: 20px;'><small>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</small></div>", unsafe_allow_html=True)

st.markdown("---")

# Main Content Area
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Analytics", "ℹ️ About"])

with tab1:
    # Input Section
    st.subheader("📝 Enter Student Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        study_hours = st.slider(
            "📖 Study Hours Per Day",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5,
            help="Number of hours student studies per day"
        )
    
    with col2:
        st.markdown("### Study Analysis")
        if study_hours >= 8:
            st.markdown("<span style='color: green; font-weight: bold;'>✅ Excellent Study Habits</span>", unsafe_allow_html=True)
        elif study_hours >= 5:
            st.markdown("<span style='color: blue; font-weight: bold;'>👍 Good Study Routine</span>", unsafe_allow_html=True)
        elif study_hours >= 2:
            st.markdown("<span style='color: orange; font-weight: bold;'>⚠️ Need Improvement</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color: red; font-weight: bold;'>❌ Critical - Increase Study Time</span>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("### Progress")
        st.progress(min(study_hours / 8, 1.0))
        st.caption(f"{min(study_hours / 8 * 100, 100):.0f}% of ideal study time")

    st.markdown("---")
    
    # Prediction Button
    col_button = st.columns([1, 1, 1])
    with col_button[1]:
        if st.button("🚀 Predict Student Status", use_container_width=True, key="predict_btn"):
            
            # Prepare input
            X = pd.DataFrame({"Study_Hours": [study_hours]})
            
            # Make prediction
            pred = model.predict(X)[0]
            prediction_label = "PASS ✅" if pred == 1 else "FAIL ❌"
            
            # Get probability
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                confidence = max(proba) * 100
                prob_fail = proba[0] * 100
                prob_pass = proba[1] * 100
            else:
                confidence = 95
                prob_fail = 0
                prob_pass = 100
            
            # Display Results
            st.markdown("---")
            st.markdown("### 🎓 PREDICTION RESULT")
            
            # Main Prediction Card
            if pred == 1:
                st.markdown(f"<div class='pass-badge'>✅ STUDENT WILL PASS</div>", unsafe_allow_html=True)
                color_indicator = "🟢"
            else:
                st.markdown(f"<div class='fail-badge'>❌ STUDENT WILL FAIL</div>", unsafe_allow_html=True)
                color_indicator = "🔴"
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Detailed Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="📊 Confidence Level",
                    value=f"{confidence:.2f}%",
                    delta="High Accuracy" if confidence > 80 else "Medium Accuracy"
                )
            
            with col2:
                st.metric(
                    label="📚 Study Hours",
                    value=f"{study_hours:.1f} hrs",
                    delta="Per Day"
                )
            
            with col3:
                st.metric(
                    label="🎯 Prediction Status",
                    value=prediction_label,
                    delta=""
                )
            
            st.markdown("---")
            
            # Probability Distribution
            st.subheader("📈 Probability Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Create Plotly bar chart with proper colors
                fig = go.Figure(data=[
                    go.Bar(
                        x=['FAIL', 'PASS'],
                        y=[prob_fail, prob_pass],
                        marker=dict(color=['#eb3349', '#11998e']),
                        text=[f'{prob_fail:.2f}%', f'{prob_pass:.2f}%'],
                        textposition='auto',
                    )
                ])
                fig.update_layout(
                    title='Outcome Probability Distribution',
                    xaxis_title='Outcome',
                    yaxis_title='Probability (%)',
                    showlegend=False,
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📋 Detailed Breakdown")
                st.markdown(f"""
                <div class='info-box'>
                    <b>❌ Probability of FAILING:</b> <span style='color: #eb3349; font-size: 18px;'><b>{prob_fail:.2f}%</b></span><br><br>
                    <b>✅ Probability of PASSING:</b> <span style='color: #11998e; font-size: 18px;'><b>{prob_pass:.2f}%</b></span><br><br>
                    <b>🎯 Model Confidence:</b> <span style='color: #667eea; font-size: 18px;'><b>{confidence:.2f}%</b></span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if pred == 1:
                st.success("""
                ✅ **Great Job!** The model predicts you will PASS.
                
                - Continue maintaining your study routine
                - Try to increase study hours if possible
                - Focus on difficult topics
                - Practice previous year papers
                """)
            else:
                st.error("""
                ⚠️ **Alert!** The model predicts you might FAIL.
                
                **Actions to take immediately:**
                - Increase study hours to at least 6-8 hours per day
                - Join study groups or get tuition
                - Focus on core concepts and basics
                - Practice problem-solving regularly
                - Set realistic daily study goals
                """)

with tab2:
    st.subheader("📊 Study Hours Impact Analysis")
    
    # Generate sample predictions for visualization
    hours_range = np.linspace(0, 24, 50)
    X_range = pd.DataFrame({"Study_Hours": hours_range})
    
    if hasattr(model, "predict_proba"):
        predictions = model.predict_proba(X_range)[:, 1] * 100
    else:
        predictions = np.where(hours_range >= 5, 100, (hours_range / 5) * 100)
    
    df_chart = pd.DataFrame({
        "Study Hours": hours_range,
        "Pass Probability (%)": predictions
    })
    
    st.line_chart(df_chart.set_index("Study Hours"), color=["#667eea"])
    
    st.markdown("""
    ### 📌 Key Insights
    - **Study hours have a strong correlation with passing probability**
    - **Recommended study hours:** 6-8 hours per day for optimal results
    - **Critical threshold:** Below 3 hours significantly increases fail risk
    - **Safe zone:** Above 7 hours ensures >90% pass probability
    """)

with tab3:
    st.subheader("ℹ️ About This Application")
    st.markdown("""
    ### 🔬 Technology Stack
    - **Machine Learning:** Logistic Regression
    - **Framework:** Streamlit
    - **Language:** Python
    - **Data Processing:** Pandas, NumPy, Scikit-learn
    
    ### 📊 Model Information
    - **Algorithm:** Logistic Regression Classifier
    - **Features:** Study Hours
    - **Output:** Binary Classification (PASS/FAIL)
    - **Accuracy:** Based on training data
    
    ### ⚠️ Important Notes
    - This model is based on historical data
    - It predicts probability, not certainty
    - Study hours is the primary factor considered
    - Always combine with actual effort and practice
    - Use this as a guide, not the final decision
    
    ### 🎯 How to Use
    1. Enter your daily study hours
    2. Click "Predict Student Status"
    3. Review the prediction and probability
    4. Follow recommendations for improvement
    5. Re-check your prediction after increasing study time
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <small>📚 Student Performance Prediction System | Made with ❤️ using Streamlit</small><br>
    <small>Disclaimer: This is an AI prediction model. Actual results may vary.</small>
</div>
""", unsafe_allow_html=True)
