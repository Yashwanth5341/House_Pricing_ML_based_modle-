import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model
@st.cache_data
def load_model():
    try:
        with open('best_housing_model_ridge_regression.pkl', 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except FileNotFoundError:
        st.error("❌ Model file 'best_housing_model_ridge_regression.pkl' not found!")
        st.stop()

# Main app
def main():
    # Load model
    model_package = load_model()
    model = model_package['model']
    feature_columns = model_package['feature_columns']
    performance_metrics = model_package['performance_metrics']
    
    # Header
    st.title("🏠 Smart House Price Predictor")
    st.markdown("### Predict house prices using AI-powered Ridge Regression")
    
    # Sidebar for model info
    with st.sidebar:
        st.header("📊 Model Information")
        
        # Get actual metrics from the model
        test_r2 = performance_metrics.get('test_r2_score', 0)
        test_rmse = performance_metrics.get('test_rmse', 0)
        test_mae = performance_metrics.get('test_mae', 0)
        train_r2 = performance_metrics.get('train_r2_score', 0)
        
        # Get training info
        training_info = model_package.get('training_info', {})
        training_samples = training_info.get('training_samples', 'N/A')
        testing_samples = training_info.get('testing_samples', 'N/A')
        total_features = training_info.get('total_features', len(feature_columns))
        
        # Display metrics
        st.metric("🎯 Test R² Score", f"{test_r2:.3f}")
        st.metric("📊 Test RMSE", f"${test_rmse:,.0f}")
        st.metric("📈 Test MAE", f"${test_mae:,.0f}")
        st.metric("🏋️ Train R² Score", f"{train_r2:.3f}")
        
        st.markdown("---")
        
        # Model details
        st.markdown("### 🤖 Model Details")
        st.markdown(f"**Model Type:** Ridge Regression")
        st.markdown(f"**Features Used:** {total_features} housing attributes")
        st.markdown(f"**Training Data:** {training_samples} houses")
        st.markdown(f"**Test Data:** {testing_samples} houses")
        
        # Model parameters
        model_params = model_package.get('model_parameters', {})
        if model_params:
            st.markdown("---")
            st.markdown("### ⚙️ Model Parameters")
            alpha = model_params.get('alpha', 'N/A')
            solver = model_params.get('solver', 'N/A')
            st.markdown(f"**Alpha:** {alpha}")
            st.markdown(f"**Solver:** {solver}")
        
        # Performance indicator
        if test_r2 > 0.7:
            st.success("🟢 Excellent Performance")
        elif test_r2 > 0.5:
            st.info("🟡 Good Performance")
        else:
            st.warning("🔴 Needs Improvement")
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🏡 Enter House Details")
        
        # Input fields based on typical housing features
        area = st.number_input(
            "🏠 Area (sq ft)", 
            min_value=500, 
            max_value=20000, 
            value=2000, 
            step=100,
            help="Total area of the house in square feet"
        )
        
        bedrooms = st.number_input(
            "🛏️ Number of Bedrooms", 
            min_value=1, 
            max_value=10, 
            value=3, 
            step=1
        )
        
        bathrooms = st.number_input(
            "🚿 Number of Bathrooms", 
            min_value=1, 
            max_value=10, 
            value=2, 
            step=1
        )
        
        stories = st.number_input(
            "🏢 Number of Stories", 
            min_value=1, 
            max_value=5, 
            value=2, 
            step=1
        )
        
        # Amenities (binary features)
        st.subheader("🏅 Amenities")
        
        col1a, col1b = st.columns(2)
        
        with col1a:
            mainroad = st.checkbox("🛣️ Main Road Access", value=True)
            guestroom = st.checkbox("🏨 Guest Room", value=False)
            basement = st.checkbox("🏠 Basement", value=False)
        
        with col1b:
            hotwaterheating = st.checkbox("🔥 Hot Water Heating", value=False)
            airconditioning = st.checkbox("❄️ Air Conditioning", value=True)
            parking = st.checkbox("🚗 Parking", value=True)
        
        # Preferred area
        prefarea = st.checkbox("⭐ Preferred Area", value=False)
        
        # Furnishing status
        furnishingstatus = st.selectbox(
            "🪑 Furnishing Status",
            options=["furnished", "semi-furnished", "unfurnished"],
            index=1
        )
        
        # Predict button
        if st.button("🔮 Predict House Price", type="primary", use_container_width=True):
            # Create input dataframe
            input_data = create_input_dataframe(
                area, bedrooms, bathrooms, stories, mainroad, guestroom, 
                basement, hotwaterheating, airconditioning, parking, 
                prefarea, furnishingstatus, feature_columns
            )
            
            # Make prediction
            try:
                prediction = model.predict(input_data)[0]
                
                # Store prediction for insights
                st.session_state['last_prediction'] = prediction
                
                # Display prediction in col2
                with col2:
                    display_prediction(prediction, performance_metrics)
                    
                    # Additional insights
                    display_insights(input_data, area, bedrooms, bathrooms, prediction)
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    with col2:
        if 'last_prediction' not in st.session_state:
            st.header("💰 Prediction Results")
            st.info("👈 Enter house details and click 'Predict' to see the estimated price!")
            
            # Show sample predictions
            st.subheader("📊 Sample Predictions")
            show_sample_predictions(model, feature_columns)

def create_input_dataframe(area, bedrooms, bathrooms, stories, mainroad, guestroom, 
                          basement, hotwaterheating, airconditioning, parking, 
                          prefarea, furnishingstatus, feature_columns):
    """Create input dataframe for prediction"""
    
    # Convert boolean inputs to binary
    input_dict = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'mainroad': 1 if mainroad else 0,
        'guestroom': 1 if guestroom else 0,
        'basement': 1 if basement else 0,
        'hotwaterheating': 1 if hotwaterheating else 0,
        'airconditioning': 1 if airconditioning else 0,
        'parking': 1 if parking else 0,
        'prefarea': 1 if prefarea else 0
    }
    
    # Handle furnishing status (assuming it's encoded as numbers in the model)
    furnishing_map = {"furnished": 0, "semi-furnished": 1, "unfurnished": 2}
    input_dict['furnishingstatus'] = furnishing_map.get(furnishingstatus, 1)
    
    # Create dataframe with all required features
    input_df = pd.DataFrame([input_dict])
    
    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Default value for missing features
    
    # Reorder columns to match training data
    input_df = input_df[feature_columns]
    
    return input_df

def display_prediction(prediction, performance_metrics):
    """Display the prediction results"""
    st.header("💰 Predicted House Price")
    
    # Main prediction display
    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
            <h1 style="color: white; font-size: 3em; margin: 0;">
                ${prediction:,.0f}
            </h1>
            <p style="color: white; font-size: 1.2em; margin: 10px 0;">
                Estimated House Price
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Confidence interval
    rmse = performance_metrics.get('test_rmse', 100000)  # Use actual test RMSE
    lower_bound = prediction - rmse
    upper_bound = prediction + rmse
    
    st.markdown("### 📊 Price Range (±1 RMSE)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lower Bound", f"${max(0, lower_bound):,.0f}")
    with col2:
        st.metric("Expected Price", f"${prediction:,.0f}")
    with col3:
        st.metric("Upper Bound", f"${upper_bound:,.0f}")
    
    # Model confidence based on actual R² score
    test_r2 = performance_metrics.get('test_r2_score', 0)
    if test_r2 > 0.7:
        confidence_level = "High"
        confidence_color = "success"
    elif test_r2 > 0.5:
        confidence_level = "Good"
        confidence_color = "info"
    else:
        confidence_level = "Moderate"
        confidence_color = "warning"
    
    if confidence_color == "success":
        st.success(f"🎯 Model Confidence: {confidence_level} (R² = {test_r2:.3f})")
    elif confidence_color == "info":
        st.info(f"🎯 Model Confidence: {confidence_level} (R² = {test_r2:.3f})")
    else:
        st.warning(f"🎯 Model Confidence: {confidence_level} (R² = {test_r2:.3f})")

def display_insights(input_data, area, bedrooms, bathrooms, prediction):
    """Display additional insights about the house"""
    st.markdown("### 🔍 Property Analysis")
    
    # Price per sq ft
    price_per_sqft = prediction / area
    st.metric("Price per Sq Ft", f"${price_per_sqft:.0f}")
    
    # Room ratios
    if bedrooms > 0:
        area_per_bedroom = area / bedrooms
        st.metric("Area per Bedroom", f"{area_per_bedroom:.0f} sq ft")
    
    bathroom_bedroom_ratio = bathrooms / bedrooms if bedrooms > 0 else 0
    st.metric("Bathroom to Bedroom Ratio", f"{bathroom_bedroom_ratio:.2f}")
    
    # Property category
    if area < 1500:
        category = "🏠 Compact Home"
        color = "blue"
    elif area < 3000:
        category = "🏡 Standard Home"
        color = "green"
    else:
        category = "🏰 Luxury Home"
        color = "gold"
    
    st.markdown(f"**Property Category:** :{color}[{category}]")
    
    # Price category
    if prediction < 3000000:
        price_category = "💚 Affordable"
    elif prediction < 6000000:
        price_category = "💛 Mid-range"
    else:
        price_category = "💎 Premium"
    
    st.markdown(f"**Price Category:** {price_category}")

def show_sample_predictions(model, feature_columns):
    """Show sample predictions for reference"""
    
    # Sample house configurations
    samples = [
        {"name": "Starter Home", "area": 1200, "bedrooms": 2, "bathrooms": 1, "stories": 1, "amenities": 2},
        {"name": "Family Home", "area": 2000, "bedrooms": 3, "bathrooms": 2, "stories": 2, "amenities": 4},
        {"name": "Luxury Home", "area": 3500, "bedrooms": 4, "bathrooms": 3, "stories": 2, "amenities": 6},
    ]
    
    for sample in samples:
        # Create sample input
        sample_input = pd.DataFrame([{
            'area': sample['area'],
            'bedrooms': sample['bedrooms'],
            'bathrooms': sample['bathrooms'],
            'stories': sample['stories'],
            'mainroad': 1,
            'guestroom': 1 if sample['amenities'] >= 4 else 0,
            'basement': 1 if sample['amenities'] >= 5 else 0,
            'hotwaterheating': 1 if sample['amenities'] >= 3 else 0,
            'airconditioning': 1 if sample['amenities'] >= 2 else 0,
            'parking': 1,
            'prefarea': 1 if sample['amenities'] >= 6 else 0,
            'furnishingstatus': 1
        }])
        
        # Ensure all columns are present
        for col in feature_columns:
            if col not in sample_input.columns:
                sample_input[col] = 0
        
        sample_input = sample_input[feature_columns]
        
        try:
            pred = model.predict(sample_input)[0]
            
            st.markdown(f"**{sample['name']}**")
            st.markdown(f"📐 {sample['area']} sq ft • 🛏️ {sample['bedrooms']} bed • 🚿 {sample['bathrooms']} bath")
            st.markdown(f"💰 **${pred:,.0f}**")
            st.markdown("---")
        except Exception as e:
            st.markdown(f"**{sample['name']}** - Unable to predict")
            st.markdown("---")

# Add custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        font-size: 16px;
        padding: 10px 20px;
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    
    .stNumberInput > div > div {
        background-color: #f8f9fa;
    }
    
    .stCheckbox {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
