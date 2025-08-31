# app_ui.py - UI Components for Rock Paper Scissors Classifier
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import json
from PIL import Image
import glob
import random
import pathlib
import time
import tensorflow as tf

# Configure page
def setup_page():
    """Configure Streamlit page settings and custom CSS"""
    st.set_page_config(
        page_title="Rock-Paper-Scissors AI Classifier",
        page_icon="✂️", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1e88e5;
            margin-bottom: 2rem;
        }
        .subtitle {
            font-size: 1.2rem;
            text-align: center;
            color: #666;
            margin-bottom: 3rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
        .prediction-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            font-size: 1.5rem;
            font-weight: bold;
            margin: 1rem 0;
        }
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.5rem;
            border-radius: 10px;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render the main header section"""
    st.markdown('<p class="main-header">🤖 Rock-Paper-Scissors AI Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Computer Vision with MobileNetV2 Transfer Learning</p>', unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar navigation and return selected page"""
    st.sidebar.title("🎯 Navigation")
    return st.sidebar.selectbox(
        "Choose a page:",
        ["🔮 Live Prediction", "📊 Model Performance", "🗂️ Technical Details", "ℹ️ About"]
    )

def render_prediction_card(predicted_class, emoji, confidence):
    """Render styled prediction result card"""
    st.markdown(f"""
    <div class="prediction-card">
        {emoji} Prediction: {predicted_class}<br>
        Confidence: {confidence:.1%}
    </div>
    """, unsafe_allow_html=True)

def render_probability_chart(class_names, probabilities, emojis):
    """Render interactive probability distribution chart"""
    # Create DataFrame for plotting
    prob_data = {
        'Class': class_names,
        'Probability': probabilities,
        'Emoji': emojis
    }
    prob_df = pd.DataFrame(prob_data)
    
    # Create bar chart
    fig = px.bar(
        prob_df, 
        x='Class', 
        y='Probability', 
        color='Class',
        hover_data=['Probability', 'Emoji'],
        color_discrete_map={'Rock': '#636EFA', 'Paper': '#EF553B', 'Scissors': '#00CC96'},
        title="Prediction Probabilities"
    )
    fig.update_layout(showlegend=False, height=400)
    return fig

def render_instructions():
    """Render usage instructions"""
    st.markdown("---")
    st.subheader("📋 How to Use")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🪨 Rock**
        - Make a fist
        - Keep fingers tucked in
        - Position hand clearly in frame
        """)
    
    with col2:
        st.markdown("""
        **📄 Paper**
        - Open hand flat
        - All fingers extended
        - Palm facing camera
        """)
    
    with col3:
        st.markdown("""
        **✂️ Scissors**
        - Index and middle finger extended
        - Other fingers tucked in
        - Clear V-shape
        """)

def render_debug_sections(processed_image, raw_predictions, class_names):
    """Render debug information sections"""
    # Debug: Show the preprocessed image
    with st.expander("🔍 Debug: Preprocessed Image"):
        if processed_image is not None:
            # Show preprocessed image for debugging
            display_image = processed_image[0].copy()
            st.image(display_image, caption="Preprocessed Image (for debugging)", use_container_width=True)
        
    # Debug: Show image statistics
    with st.expander("🔍 Debug: Image Statistics"):
        if processed_image is not None:
            st.write(f"Image shape: {processed_image.shape}")
            st.write(f"Pixel value range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
            st.write(f"Mean pixel value: {processed_image.mean():.3f}")
    
    # Debug: Show raw predictions
    with st.expander("🔍 Debug: Raw Predictions"):
        if raw_predictions is not None:
            st.write(f"Raw prediction values: {raw_predictions}")
            # Create a bar chart of the predictions
            df = pd.DataFrame({
                'Class': class_names,
                'Probability': raw_predictions
            })
            st.bar_chart(df.set_index('Class'))

def render_metric_card(title, value, format_str=None):
    """Render a styled metric card"""
    if format_str:
        formatted_value = format_str.format(value)
    else:
        formatted_value = str(value)
    
    st.markdown(f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <h2>{formatted_value}</h2>
    </div>
    """, unsafe_allow_html=True)

def render_performance_page():
    """Render model performance dashboard with dynamic data loading"""
    st.header("📊 Model Performance Dashboard")
    
    # Try to load metrics from the actual saved file
    try:
        # Load from metrics_report.txt (the actual saved file)
        metrics_path = pathlib.Path("models/metrics_report.txt")
        if metrics_path.exists():
            # Parse the classification report
            with open(metrics_path, 'r') as f:
                content = f.read()
            
            # Extract overall accuracy
            import re
            accuracy_match = re.search(r'accuracy\s+(\d+\.\d+)', content)
            test_accuracy = float(accuracy_match.group(1)) if accuracy_match else None
            
            # Parse per-class metrics
            lines = content.split('\n')
            metrics_data = []
            for line in lines:
                if any(cls in line for cls in ['paper', 'rock', 'scissors']):
                    parts = line.split()
                    if len(parts) >= 5:
                        class_name = parts[0]
                        precision = float(parts[1])
                        recall = float(parts[2])
                        f1_score = float(parts[3])
                        support = int(parts[4])
                        metrics_data.append({
                            'Class': class_name.capitalize(),
                            'Precision': precision,
                            'Recall': recall,
                            'F1-Score': f1_score,
                            'Support': support
                        })
            
            # Display key metrics in cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if test_accuracy:
                    render_metric_card("Test Accuracy", test_accuracy, "{:.1%}")
            
            with col2:
                # Calculate average F1-score
                if metrics_data:
                    avg_f1 = np.mean([d['F1-Score'] for d in metrics_data])
                    render_metric_card("Avg F1-Score", avg_f1, "{:.3f}")
            
            with col3:
                # Total support (test samples)
                if metrics_data:
                    total_support = sum([d['Support'] for d in metrics_data])
                    render_metric_card("Test Samples", total_support)
            
            with col4:
                # Model size
                model_path = pathlib.Path("models/best_rps_mobilenetv2.keras")
                if model_path.exists():
                    size_mb = model_path.stat().st_size / (1024*1024)
                    render_metric_card("Model Size", size_mb, "{:.1f} MB")
            
            # Detailed classification metrics
            if metrics_data:
                st.subheader("📈 Detailed Performance by Class")
                
                # Create interactive performance chart
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Precision & Recall', 'F1-Score by Class'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Precision and Recall chart
                fig.add_trace(
                    go.Bar(name='Precision', x=[d['Class'] for d in metrics_data], 
                           y=[d['Precision'] for d in metrics_data], marker_color='lightblue'),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Bar(name='Recall', x=[d['Class'] for d in metrics_data], 
                           y=[d['Recall'] for d in metrics_data], marker_color='lightcoral'),
                    row=1, col=1
                )
                
                # F1-Score chart
                fig.add_trace(
                    go.Bar(name='F1-Score', x=[d['Class'] for d in metrics_data], 
                           y=[d['F1-Score'] for d in metrics_data], marker_color='lightgreen',
                           showlegend=False),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display metrics table
                st.subheader("📋 Detailed Metrics Table")
                st.table(pd.DataFrame(metrics_data))
        else:
            st.warning("⚠️ No metrics report found. Please run model training first.")
            
    except Exception as e:
        st.error(f"Error loading performance metrics: {e}")
        st.info("Ensure the model has been trained and metrics_report.txt exists in models/")
    
    # Display confusion matrix if available
    confusion_matrix_path = pathlib.Path("models/confusion_matrix.png")
    if confusion_matrix_path.exists():
        st.subheader("🎯 Confusion Matrix")
        st.image(str(confusion_matrix_path), 
                caption="Model Confusion Matrix - Shows how often each class is confused with others")
    else:
        st.info("Confusion matrix visualization not found. Run training to generate.")

def render_technical_page():
    """Render technical implementation details with live model data"""
    st.header("🗂️ Technical Implementation")

    # --- Model Architecture (with live stats) ---
    st.subheader("🧠 Model Architecture")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Transfer Learning with MobileNetV2

        **Base Model**
        - MobileNetV2 (`include_top=False`), ImageNet pre-trained
        - Input: **224×224×3** RGB images

        **In-Model Pipeline**
        - **Augmentation:** RandomFlip(horizontal), RandomRotation(0.10), RandomZoom(0.15), RandomCrop(90% → 224×224), RandomContrast(0.20)
        - **Preprocessing:** Rescaling(1/127.5, offset=-1) for MobileNetV2 compatibility

        **Classifier Head**
        - GlobalAveragePooling2D → Dropout(0.25) → Dense(3, softmax)

        **Training Strategy**
        - **Stage 1:** Base frozen, **8 epochs**, Adam lr=1e-3
        - **Stage 2:** Unfreeze **top 30 layers**, **10 epochs**, Adam lr=1e-5
        - **Class Weighting:** Scissors upweighted to **1.5** when present
        """)

    with col2:
        # Live model stats
        try:
            models_dir = pathlib.Path("models")
            model_path = models_dir / "best_rps_mobilenetv2.keras"
            
            if model_path.exists():
                # Load model to get live stats
                model = tf.keras.models.load_model(model_path, compile=False)
                total_params = model.count_params()
                
                # Calculate trainable vs non-trainable
                trainable_params = int(sum(np.prod(w.shape) for w in model.trainable_weights)) if model.trainable_weights else 0
                non_trainable_params = total_params - trainable_params
                
                # File size
                size_mb = model_path.stat().st_size / (1024*1024)
                
                # Quick latency test (small sample to keep UI responsive)
                x_test = np.random.rand(1, 224, 224, 3).astype("float32")
                # Warm up
                for _ in range(3):
                    model.predict(x_test, verbose=0)
                
                # Measure latency
                import time
                start_time = time.time()
                for _ in range(10):
                    model.predict(x_test, verbose=0)
                avg_latency_ms = (time.time() - start_time) / 10 * 1000
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 Live Model Stats</h3>
                    <p><strong>File:</strong> {model_path.name}</p>
                    <p><strong>Total params:</strong> {total_params:,}</p>
                    <p><strong>Trainable:</strong> {trainable_params:,}</p>
                    <p><strong>Non-trainable:</strong> {non_trainable_params:,}</p>
                    <p><strong>Size:</strong> {size_mb:.1f} MB</p>
                    <p><strong>Latency:</strong> {avg_latency_ms:.1f} ms</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Model not found. Run training script first to see live stats.")
                
        except Exception as e:
            st.warning(f"Could not load model for live stats: {e}")

    # --- Training Details ---
    st.subheader("🎯 Training Configuration")
    
    with st.expander("🔧 Model Architecture Code"):
        st.code("""
# Data augmentation pipeline (embedded in model)
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomCrop(int(224*0.9), int(224*0.9)),
    tf.keras.layers.Resizing(224, 224),
    tf.keras.layers.RandomContrast(0.20),
], name="data_augmentation")

# Preprocessing for MobileNetV2
preprocessing_layer = tf.keras.layers.Rescaling(1.0/127.5, offset=-1.0)

# Build model
base = tf.keras.applications.MobileNetV2(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
base.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_aug(inputs)
x = preprocessing_layer(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
        """, language="python")

    with st.expander("🎯 Two-Stage Training"):
        st.code("""
# Stage 1: Train classifier head with frozen base
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(train_ds, validation_data=val_ds, epochs=8,
          callbacks=[checkpoint_callback], class_weight=class_weight)

# Stage 2: Fine-tune top layers
base.trainable = True
for layer in base.layers[:-30]:  # Keep bottom layers frozen
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(train_ds, validation_data=val_ds, epochs=10,
          callbacks=[checkpoint_callback], class_weight=class_weight)
        """, language="python")

    # --- Key Design Decisions ---
    st.subheader("💡 Key Design Decisions")
    
    decisions = [
        {
            "Decision": "MobileNetV2 as backbone",
            "Rationale": "Optimal balance of accuracy, model size (9.1MB), and inference speed (~26.5ms)",
            "Alternative": "EfficientNet or ResNet50 for potentially higher accuracy at larger size"
        },
        {
            "Decision": "Two-stage training approach", 
            "Rationale": "Stabilize classifier head first, then fine-tune pretrained features for domain adaptation",
            "Alternative": "End-to-end training or longer frozen training period"
        },
        {
            "Decision": "In-model augmentation pipeline",
            "Rationale": "Ensures consistency between training and inference, improves deployment robustness",
            "Alternative": "External augmentation with tf.data or imgaug"
        },
        {
            "Decision": "Class weighting (1.5x scissors)",
            "Rationale": "Address observed class imbalance and reduce false negatives for scissors",
            "Alternative": "Data oversampling, focal loss, or additional data collection"
        },
        {
            "Decision": "Rescaling layer vs Lambda preprocessing",
            "Rationale": "Better deployment compatibility, avoids serialization issues with Lambda layers",
            "Alternative": "External preprocessing or tf.keras.applications.mobilenet_v2.preprocess_input"
        }
    ]

    for i, decision in enumerate(decisions, 1):
        with st.expander(f"Decision {i}: {decision['Decision']}"):
            st.write(f"**Rationale:** {decision['Rationale']}")
            st.write(f"**Alternatives:** {decision['Alternative']}")

def render_about_page():
    """Render about page with dynamic project information"""
    st.header("🎯 Project Overview")

    st.markdown("""
    This **Rock-Paper-Scissors AI Classifier** demonstrates an end-to-end computer vision pipeline using transfer learning with MobileNetV2. The project focuses on production-ready deployment, explainability, and continuous improvement through human feedback.
    """)

    # Load and display actual model performance
    st.subheader("📊 Model Performance")
    
    try:
        # Load actual metrics
        metrics_path = pathlib.Path("models/metrics_report.txt")
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                content = f.read()
            
            # Extract accuracy
            import re
            accuracy_match = re.search(r'accuracy\s+(\d+\.\d+)', content)
            if accuracy_match:
                accuracy = float(accuracy_match.group(1))
                st.write(f"**Current Test Accuracy:** {accuracy:.1%}")
            
            # Extract support (total test samples)
            support_match = re.search(r'accuracy\s+\d+\.\d+\s+(\d+)', content)
            if support_match:
                total_samples = int(support_match.group(1))
                correct_predictions = int(accuracy * total_samples) if accuracy_match else None
                if correct_predictions:
                    st.write(f"**Test Results:** {correct_predictions} / {total_samples} correct predictions")
            
            # Parse and display per-class metrics
            lines = content.split('\n')
            st.write("**Per-Class Performance:**")
            for line in lines:
                if any(cls in line for cls in ['paper', 'rock', 'scissors']):
                    parts = line.split()
                    if len(parts) >= 5:
                        class_name = parts[0].capitalize()
                        precision = float(parts[1])
                        recall = float(parts[2]) 
                        f1_score = float(parts[3])
                        support = int(parts[4])
                        st.write(f"- **{class_name}**: {precision:.1%} precision, {recall:.1%} recall, {f1_score:.3f} F1-score ({support} samples)")
        else:
            st.info("Model metrics not found. Train the model to see performance data.")
            
    except Exception as e:
        st.warning(f"Could not load performance metrics: {e}")

    # Load actual model stats
    st.subheader("🔧 Model Specifications")
    try:
        model_path = pathlib.Path("models/best_rps_mobilenetv2.keras")
        if model_path.exists():
            model = tf.keras.models.load_model(model_path, compile=False)
            total_params = model.count_params()
            trainable_params = int(sum(np.prod(w.shape) for w in model.trainable_weights)) if model.trainable_weights else 0
            size_mb = model_path.stat().st_size / (1024*1024)
            
            # Quick latency test
            x_test = np.random.rand(1, 224, 224, 3).astype("float32")
            import time
            start_time = time.time()
            for _ in range(20):
                model.predict(x_test, verbose=0)
            avg_latency_ms = (time.time() - start_time) / 20 * 1000
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Total Parameters:** {total_params:,}")
                st.write(f"**Trainable Parameters:** {trainable_params:,}")
                st.write(f"**Model Size:** {size_mb:.1f} MB")
            with col2:
                st.write(f"**Average Latency:** {avg_latency_ms:.1f} ms")
                st.write(f"**Input Shape:** 224×224×3")
                st.write(f"**Output Classes:** 3 (rock, paper, scissors)")
                
    except Exception as e:
        st.info("Load model to see detailed specifications.")

    st.subheader("🛠️ Technologies Used")
    
    # Load actual requirements
    tech_used = []
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        key_tech = {
            'tensorflow': 'Deep Learning Framework',
            'streamlit': 'Web Application Framework', 
            'opencv-python': 'Computer Vision (with PIL fallback)',
            'plotly': 'Interactive Visualizations',
            'pandas': 'Data Manipulation',
            'numpy': 'Numerical Computing',
            'scikit-learn': 'Machine Learning Utilities',
            'pillow': 'Image Processing'
        }
        
        for package, description in key_tech.items():
            if package in requirements.lower():
                # Extract version if present
                for line in requirements.split('\n'):
                    if package in line.lower():
                        tech_used.append(f"**{package.title()}**: {description}")
                        break
                        
    except:
        # Fallback to static list
        tech_used = [
            "**TensorFlow**: Deep Learning Framework",
            "**Streamlit**: Web Application Framework", 
            "**OpenCV/PIL**: Computer Vision and Image Processing",
            "**Plotly**: Interactive Visualizations",
            "**NumPy, Pandas**: Data Science Stack"
        ]
    
    for tech in tech_used:
        st.markdown(f"- {tech}")

    st.subheader("🚀 Key Features")
    features = [
        "**Real-time Prediction**: Webcam and upload support with instant classification",
        "**Transfer Learning**: Fine-tuned MobileNetV2 with two-stage training approach",
        "**Explainable AI**: Grad-CAM visualization for model interpretability", 
        "**Human-in-the-Loop**: Save corrected samples directly from UI for retraining",
        "**Production Ready**: Optimized .keras format, Docker containerization",
        "**Comprehensive Monitoring**: Automated metrics, confusion matrix, classification report"
    ]
    
    for feature in features:
        st.markdown(f"- {feature}")

    st.subheader("🎓 Learning Outcomes")
    st.markdown("""
    This project demonstrates practical ML engineering skills:
    
    - **Computer Vision**: Image preprocessing, augmentation, and classification
    - **Deep Learning**: Transfer learning, fine-tuning strategies, model optimization
    - **Software Engineering**: Modular design, separation of concerns, professional UI
    - **MLOps**: Model evaluation, performance monitoring, deployment pipeline
    - **Explainable AI**: Grad-CAM implementation for visual model explanations
    - **Continuous Learning**: Human-in-the-loop data collection and feedback integration
    
    **Built with production best practices**: TensorFlow, Streamlit, and modern ML deployment patterns.
    """)


def render_test_buttons(project_root, predict_function, class_names, emojis):
    """Render test buttons for dataset images"""
    
    col1, col2, col3 = st.columns(3)
    
    categories = [
        ("Rock", "data/raw/rps-test-set/rock", col1),
        ("Paper", "data/raw/rps-test-set/paper", col2),
        ("Scissors", "data/raw/rps-test-set/scissors", col3)
    ]
    
    # Create the test_images list with random images
    test_images = []

    for label, folder_path, col in categories:
        image_paths = glob.glob(f"{folder_path}/*.png")
        if len(image_paths) > 0:
            # Use random.sample to avoid picking duplicates
            selected_image = random.sample(image_paths, 1)[0]
            test_images.append((label, selected_image, col))
    
    for class_name, image_path, col in test_images:
        with col:
            if st.button(f"Test {class_name} Image"):
                full_path = project_root / image_path
                if full_path.exists():
                    image = Image.open(full_path)
                    st.image(image, caption=f"{class_name} Test Image", use_container_width=True)
                    
                    with st.spinner("🤔 Analyzing image..."):
                        predicted_idx, confidence, all_probs = predict_function(image)
                    
                    if predicted_idx is not None:
                        predicted_class = class_names[predicted_idx]
                        predicted_emoji = emojis[predicted_idx]
                        render_prediction_card(predicted_class, predicted_emoji, confidence)
                else:
                    st.warning(f"{class_name} test image not found")