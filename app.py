"""
Streamlit Application - Student Grade Predictor
Interactive dashboard with data overview and what-if predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.preprocess import preprocess_oulad_data
from utils.download_oulad import load_or_generate_oulad
from utils.visualize import (
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_linear_coefficients,
    plot_predictions_vs_actual,
    plot_residuals
)
from models.train import StudentGradePredictor


# Configure page
st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained classification models from disk."""
    models_dir = 'models'
    
    try:
        with open(os.path.join(models_dir, 'logistic_regression.pkl'), 'rb') as f:
            lr_model = pickle.load(f)
        
        with open(os.path.join(models_dir, 'random_forest.pkl'), 'rb') as f:
            rf_model = pickle.load(f)
        
        with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        
        with open(os.path.join(models_dir, 'encoders.pkl'), 'rb') as f:
            encoders = pickle.load(f)
        
        with open(os.path.join(models_dir, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open(os.path.join(models_dir, 'feature_names.pkl'), 'rb') as f:
            feature_data = pickle.load(f)
        
        # Handle both dict and list formats
        if isinstance(feature_data, dict):
            feature_names = feature_data.get('feature_names', [])
        else:
            feature_names = feature_data
        
        return {
            'lr_model': lr_model,
            'rf_model': rf_model,
            'scaler': scaler,
            'encoders': encoders,
            'label_encoder': label_encoder,
            'feature_names': feature_names
        }
    except FileNotFoundError:
        return None


@st.cache_data
def load_data():
    """Load or create OULAD dataset."""
    data_dir = Path('data')
    students_path = data_dir / 'students.csv'
    assessments_path = data_dir / 'assessments.csv'
    vle_path = data_dir / 'studentVle.csv'

    is_real = all(p.exists() for p in [students_path, assessments_path, vle_path])
    df = load_or_generate_oulad()
    return df, is_real


def train_models_if_needed():
    """Train models if they don't exist."""
    models_dir = 'models'
    
    required_files = ['logistic_regression.pkl', 'random_forest.pkl', 'scaler.pkl', 
                      'encoders.pkl', 'label_encoder.pkl', 'feature_names.pkl']
    
    if not all(os.path.exists(os.path.join(models_dir, f)) for f in required_files):
        st.info("📊 Training classification models... This may take a moment.")
        predictor = StudentGradePredictor(models_dir=models_dir)
        predictor.train_models()
        st.success("✅ Models trained successfully!")
        st.cache_resource.clear()
        return True
    return False


def page_overview():
    """Page 1: Data overview and visualizations."""
    st.title("📊 Student Grade Predictor - Overview")
    
    # Load data (returns tuple: df, is_real)
    df, is_real = load_data()

    # Notify user whether real OULAD CSVs were used or sample data was generated
    if is_real:
        st.success("Using real OULAD CSV files from the `data/` directory.")
    else:
        st.warning("OULAD CSV files not found — the app is using generated sample data. To use real data, add the CSV files to the `data/` folder.")
    
    st.markdown("### Dataset Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        # For OULAD data, show target class distribution instead
        if 'final_result' in df.columns:
            target_classes = df['final_result'].nunique()
            st.metric("Outcome Classes", target_classes)
        else:
            st.metric("Features", len(df.columns))
    
    # Display raw data
    with st.expander("📋 View Raw Data"):
        st.dataframe(df.head(20), width='stretch')
    
    # Statistics
    with st.expander("📈 Data Statistics"):
        st.dataframe(df.describe(), width='stretch')
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("### Visualizations")
    
    # Correlation heatmap
    st.markdown("#### Correlation Heatmap")
    fig_corr = plot_correlation_heatmap(df)
    st.pyplot(fig_corr, width='stretch')
    
    # Feature importance
    st.markdown("#### Feature Importance (Random Forest)")
    models = load_models()
    if models and models['rf_model']:
        importances = models['rf_model'].feature_importances_
        fig_imp = plot_feature_importance(importances, models['feature_names'], 'Random Forest')
        st.plotly_chart(fig_imp, use_container_width=False)
    
    # Linear regression coefficients
    st.markdown("#### Linear Regression Coefficients")
    if models and models['lr_model']:
        coefficients = models['lr_model'].coef_
        fig_coef = plot_linear_coefficients(coefficients, models['feature_names'])
        st.plotly_chart(fig_coef, use_container_width=False)


def page_prediction():
    """Page 2: What-if prediction with interactive inputs for OULAD features."""
    st.title("🔮 What-If Prediction")
    
    # Load models
    models = load_models()
    
    if models is None:
        st.error("❌ Models not found. Please train models first.")
        return
    
    scaler = models['scaler']
    lr_model = models['lr_model']
    rf_model = models['rf_model']
    encoders = models['encoders']
    label_encoder = models['label_encoder']
    feature_names = models['feature_names']
    
    st.markdown("### Adjust Student Parameters (OULAD Features)")
    st.info("Configure student characteristics to get outcome predictions.")
    
    # Get valid options from encoders
    age_options = sorted(encoders['age_band'].classes_.tolist())
    gender_options = sorted(encoders['gender'].classes_.tolist())
    disability_options = sorted(encoders['disability'].classes_.tolist())
    region_options = sorted(encoders['region'].classes_.tolist())
    
    # Create columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Categorical Features:**")
        age_band = st.selectbox(
            "Age Band",
            options=age_options,
            help="Student age group"
        )
        
        gender = st.selectbox(
            "Gender",
            options=gender_options,
            help="Student gender"
        )
        
        disability = st.selectbox(
            "Disability",
            options=disability_options,
            help="Student has disability"
        )
        
        region = st.selectbox(
            "Region",
            options=region_options,
            help="Student region"
        )
    
    with col2:
        st.markdown("**Numeric Features:**")
        studied_credits = st.slider(
            "Studied Credits",
            min_value=0.0, max_value=350.0, value=150.0, step=10.0,
            help="Number of credits studied"
        )
        
        num_of_prev_attempts = st.slider(
            "Previous Attempts",
            min_value=0, max_value=10, value=2, step=1,
            help="Number of previous course attempts"
        )
        
        clicks_total = st.slider(
            "Total Clicks",
            min_value=0, max_value=5000, value=1000, step=100,
            help="Total clicks in VLE (Virtual Learning Environment)"
        )
        
        assessment_score_avg = st.slider(
            "Assessment Score Avg",
            min_value=0.0, max_value=100.0, value=65.0, step=1.0,
            help="Average assessment score"
        )
        
        days_since_registration = st.slider(
            "Days Since Registration",
            min_value=0, max_value=365, value=180, step=10,
            help="Days elapsed since student registration"
        )
    
    # Encode categorical features
    input_categorical = {}
    input_categorical['age_band'] = encoders['age_band'].transform([age_band])[0]
    input_categorical['gender'] = encoders['gender'].transform([gender])[0]
    input_categorical['disability'] = encoders['disability'].transform([disability])[0]
    input_categorical['region'] = encoders['region'].transform([region])[0]
    
    # Create input DataFrame matching the training format
    # Order: age_band, gender, disability, region, studied_credits, num_of_prev_attempts, clicks_total, assessment_score_avg, days_since_registration
    input_df = pd.DataFrame({
        'age_band': [input_categorical['age_band']],
        'gender': [input_categorical['gender']],
        'disability': [input_categorical['disability']],
        'region': [input_categorical['region']],
        'studied_credits': [studied_credits],
        'num_of_prev_attempts': [num_of_prev_attempts],
        'clicks_total': [clicks_total],
        'assessment_score_avg': [assessment_score_avg],
        'days_since_registration': [days_since_registration]
    })
    
    # Scale only numeric features
    numeric_features = ['studied_credits', 'num_of_prev_attempts', 'clicks_total', 'assessment_score_avg', 'days_since_registration']
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])
    
    # Convert to array for prediction
    input_scaled = input_df.values
    
    # Make predictions
    lr_prediction = lr_model.predict(input_scaled)[0]
    rf_prediction = rf_model.predict(input_scaled)[0]
    
    # Get class labels
    lr_class = label_encoder.inverse_transform([lr_prediction])[0]
    rf_class = label_encoder.inverse_transform([rf_prediction])[0]
    
    # Display results
    st.markdown("---")
    st.markdown("### 📊 Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Logistic Regression",
            f"{lr_class}",
            help="Classification result"
        )
    
    with col2:
        st.metric(
            "Random Forest",
            f"{rf_class}",
            help="Classification result"
        )
    
    # Interpretation
    st.markdown("### 💡 Interpretation")
    
    interpretations = []
    
    # Assessment score insights
    if assessment_score_avg > 80:
        interpretations.append("✅ Strong assessment performance - likely indicator of success.")
    elif assessment_score_avg < 50:
        interpretations.append("⚠️ Low assessment scores - may need additional support.")
    else:
        interpretations.append("📊 Moderate assessment performance.")
    
    # Engagement insights (clicks)
    if clicks_total > 2000:
        interpretations.append("✅ High VLE engagement - strong interaction with course materials.")
    elif clicks_total < 500:
        interpretations.append("⚠️ Low engagement - consider increasing interaction with learning materials.")
    else:
        interpretations.append("📊 Moderate engagement with course materials.")
    
    # Study credits
    if studied_credits >= 240:
        interpretations.append("✅ Full-time commitment - high course load.")
    elif studied_credits < 120:
        interpretations.append("📊 Part-time study commitment.")
    
    # Previous attempts
    if num_of_prev_attempts > 2:
        interpretations.append("⚠️ Multiple previous attempts - may indicate struggling with material.")
    elif num_of_prev_attempts == 0:
        interpretations.append("✅ First attempt - fresh start.")
    
    # Days registered
    if days_since_registration > 250:
        interpretations.append("📊 Well into the course - significant progress made.")
    elif days_since_registration < 50:
        interpretations.append("📊 Early in the course - still in learning phase.")
    
    for interpretation in interpretations:
        st.info(interpretation)
    
    st.markdown("---")
    st.markdown("**Note:** Predictions are based on OULAD learning analytics data. Actual outcomes may vary based on individual circumstances.")


def main():
    """Main app function."""
    # Sidebar
    st.sidebar.title("🎓 Student Grade Predictor")
    st.sidebar.markdown("---")
    
    # Train models if needed
    if st.sidebar.button("🚀 Train Models", help="Train or retrain the models"):
        train_models_if_needed()
    
    # Navigation
    page = st.sidebar.radio(
        "Navigate to:",
        options=["Overview", "What-If Prediction"],
        help="Choose a page to explore"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This app predicts student final grades using machine learning.
    
    **Models used:**
    - Linear Regression
    - Random Forest Regressor
    
    **Features:**
    - Study hours
    - Attendance
    - Previous grade
    - Sleep hours
    - Internet usage
    """)
    
    # Display selected page
    if page == "Overview":
        page_overview()
    elif page == "What-If Prediction":
        page_prediction()


if __name__ == '__main__':
    main()
