import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
from PIL import Image
import io
import base64
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="🦴 Osteoporosis Classification AI",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .model-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .upload-section {
        background: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #667eea;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Model information
MODEL_INFO = {
    "AlexNet": {
        "file": "AlexNet_knee_osteo_model.keras",
        "description": "Classic CNN architecture with 8 layers",
        "params": "60M parameters",
        "accuracy": "N/A"
    },
    "DenseNet121": {
        "file": "DenseNet121_osteo_model.keras",
        "description": "Dense connections for better gradient flow",
        "params": "8M parameters",
        "accuracy": "N/A"
    },
    "InceptionV3": {
        "file": "InceptionV3_knee_osteo_model.keras",
        "description": "🏆 RECOMMENDED - Multi-scale feature extraction with inception modules",
        "params": "23M parameters",
        "accuracy": "88%"
    },
    "MobileNetV2": {
        "file": "MobileNetV2_knee_osteo_model.keras",
        "description": "Lightweight model for mobile deployment",
        "params": "3.4M parameters",
        "accuracy": "N/A"
    },
    "ResNet50": {
        "file": "ResNet50_knee_osteo_model.keras",
        "description": "Deep residual network with skip connections",
        "params": "25M parameters",
        "accuracy": "N/A"
    },
    "VGG16": {
        "file": "VGG16_knee_osteo_model.keras",
        "description": "Classic architecture with 16 layers",
        "params": "138M parameters",
        "accuracy": "N/A"
    },
    "VGG19": {
        "file": "VGG19_knee_osteo_model.keras",
        "description": "Deeper VGG with 19 layers",
        "params": "143M parameters",
        "accuracy": "N/A"
    },
    "Xception": {
        "file": "Xception_knee_osteo_model.keras",
        "description": "Depthwise separable convolutions",
        "params": "22M parameters",
        "accuracy": "N/A"
    }
}

CLASS_NAMES = ['Normal', 'Osteopenia', 'Osteoporosis']
CLASS_COLORS = ['#2ecc71', '#f39c12', '#e74c3c']

@st.cache_resource
def load_model_cached(model_path):
    """Load and cache the model"""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

def predict_image(model, image):
    """Make prediction on the image"""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_class])
    
    return predicted_class, confidence, prediction[0]

def create_confidence_chart(probabilities):
    """Create confidence visualization"""
    fig = go.Figure(data=[
        go.Bar(
            x=CLASS_NAMES,
            y=probabilities,
            marker_color=CLASS_COLORS,
            text=[f'{p:.2%}' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Classification Confidence",
        xaxis_title="Bone Condition",
        yaxis_title="Confidence Score",
        yaxis=dict(range=[0, 1]),
        height=400,
        template="plotly_white"
    )
    
    return fig

def create_history_chart():
    """Create prediction history chart"""
    if not st.session_state.prediction_history:
        return None
    
    df = pd.DataFrame(st.session_state.prediction_history)
    
    fig = px.pie(
        df, 
        names='prediction', 
        title="Prediction Distribution",
        color_discrete_map={
            'Normal': CLASS_COLORS[0],
            'Osteopenia': CLASS_COLORS[1],
            'Osteoporosis': CLASS_COLORS[2]
        }
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🦴 Osteoporosis Classification AI</h1>
        <p>Advanced Deep Learning System for Bone Health Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Model Configuration")
        
        # Set InceptionV3 as default (best performing model)
        model_options = list(MODEL_INFO.keys())
        default_index = model_options.index("InceptionV3")
        
        selected_model = st.selectbox(
            "Choose Model Architecture",
            model_options,
            index=default_index,
            help="Select the deep learning model for classification (InceptionV3 recommended - 88% accuracy)"
        )
        
        # Model information with performance highlight
        with st.expander("📊 Model Information"):
            model_info = MODEL_INFO[selected_model]
            
            # Special highlighting for InceptionV3
            if selected_model == "InceptionV3":
                st.markdown("🏆 **BEST PERFORMING MODEL**")
                st.success("Recommended for highest accuracy (88%)")
            
            st.markdown(f"""
            **Architecture:** {selected_model}
            
            **Description:** {model_info['description']}
            
            **Parameters:** {model_info['params']}
            
            **Accuracy:** {model_info.get('accuracy', 'N/A')}
            """)
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        show_preprocessing = st.checkbox("Show Image Preprocessing", value=True)
        show_confidence = st.checkbox("Show Detailed Confidence", value=True)
        show_history = st.checkbox("Show Prediction History", value=True)
        
        # Clear history button
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.prediction_history = []
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload X-Ray Image")
        
        uploaded_file = st.file_uploader(
            "Choose an X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a knee X-ray image for osteoporosis classification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray Image", use_column_width=True)
            
            # Image information
            st.markdown(f"""
            <div class="model-info">
                <strong>Image Details:</strong><br>
                📏 Size: {image.size[0]} x {image.size[1]} pixels<br>
                🎨 Mode: {image.mode}<br>
                📁 Format: {image.format}
            </div>
            """, unsafe_allow_html=True)
            
            # Show preprocessing if enabled
            if show_preprocessing:
                with st.expander("🔍 Image Preprocessing"):
                    processed_img = preprocess_image(image)
                    st.image(processed_img[0], caption="Preprocessed Image (224x224)", use_column_width=True)
    
    with col2:
        st.markdown("### 🎯 Classification Results")
        
        if uploaded_file is not None:
            # Load model
            model_path = MODEL_INFO[selected_model]["file"]
            
            with st.spinner(f"Loading {selected_model} model..."):
                model = load_model_cached(model_path)
            
            if model is not None:
                # Make prediction
                with st.spinner("Analyzing X-ray image..."):
                    time.sleep(1)  # Simulate processing time
                    predicted_class, confidence, probabilities = predict_image(model, image)
                    predicted_label = CLASS_NAMES[predicted_class]
                
                # Display results
                st.markdown(f"""
                <div class="prediction-card">
                    <h3 style="margin-top:0;">🔬 Diagnosis Result</h3>
                    <h2 style="color:{CLASS_COLORS[predicted_class]}; margin:1rem 0;">
                        {predicted_label}
                    </h2>
                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                    <p><strong>Model Used:</strong> {selected_model}</p>
                    <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Progress bar for confidence
                st.progress(confidence)
                
                # Detailed confidence scores
                if show_confidence:
                    st.markdown("#### 📊 Confidence Breakdown")
                    for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
                        st.metric(
                            label=class_name,
                            value=f"{prob:.2%}",
                            delta=None
                        )
                
                # Add to history
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now(),
                    'model': selected_model,
                    'prediction': predicted_label,
                    'confidence': confidence
                })
            
            else:
                st.error("❌ Failed to load the selected model. Please check if the model file exists.")
        
        else:
            st.info("👆 Please upload an X-ray image to start classification")
    
    # Visualization section
    if uploaded_file is not None and 'probabilities' in locals():
        st.markdown("---")
        st.markdown("### 📈 Detailed Analysis")
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            # Confidence chart
            confidence_fig = create_confidence_chart(probabilities)
            st.plotly_chart(confidence_fig, use_container_width=True)
        
        with col4:
            # Prediction history
            if show_history and st.session_state.prediction_history:
                history_fig = create_history_chart()
                if history_fig:
                    st.plotly_chart(history_fig, use_container_width=True)
            else:
                st.info("No prediction history available yet")
    
    # Information section
    st.markdown("---")
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.markdown("""
        <div class="metric-card">
            <h4>🩻 Normal Bone</h4>
            <p>Healthy bone density with no signs of deterioration. Regular bone density is maintained.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown("""
        <div class="metric-card">
            <h4>⚠️ Osteopenia</h4>
            <p>Lower than normal bone density but not severe enough to be classified as osteoporosis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col7:
        st.markdown("""
        <div class="metric-card">
            <h4>🚨 Osteoporosis</h4>
            <p>Significant bone density loss, making bones fragile and more likely to fracture.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🔬 <strong>Osteoporosis Classification AI</strong> | Powered by Deep Learning</p>
        <p><small>⚠️ This tool is for research purposes only. Please consult healthcare professionals for medical diagnosis.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()