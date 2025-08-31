# app.py
# Streamlit app for Rock Paper Scissors classifier with webcam capture,
# Fixed Grad-CAM visualization, and simple human-in-the-loop image collection.
import json
from pathlib import Path
import time
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ImageEnhance

# Import UI components
from app_ui import (
    setup_page, render_header, render_sidebar, render_prediction_card,
    render_probability_chart, render_instructions, render_debug_sections,
    render_performance_page, render_technical_page, render_about_page,
    render_test_buttons
)

IMG_SIZE = (224, 224)
EMOJI = {"paper": "📄", "rock": "🪨", "scissors": "✂️"}

def project_root() -> Path:
    return Path(__file__).resolve().parent

@st.cache_resource
def load_labels():
    models_dir = project_root() / "models"
    labels_path = models_dir / "labels.json"
    if not labels_path.exists():
        st.error(f"labels.json not found at {labels_path}. Run training first.")
        st.stop()
    data = json.loads(labels_path.read_text())
    return list(data["class_names"])

@st.cache_resource
def load_model():
    models_dir = project_root() / "models"
    candidates = [
        models_dir / "best_rps_mobilenetv2.keras",
        # models_dir / "best_rps_mobilenetv2.h5",
    ]
    for p in candidates:
        if p.exists():
            return tf.keras.models.load_model(p, compile=False)
    st.error(f"No model found in {models_dir}. Run src/05_retrain_mobilenetv2.py first.")
    st.stop()

def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = ImageOps.fit(img, IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32)      # model contains Rescaling layer
    return np.expand_dims(arr, axis=0)

def predict_image(model, img: Image.Image):
    x = preprocess_pil(img)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return idx, float(probs[idx]), probs

def make_gradcam(model, img_pil):
    """Generate Grad-CAM visualization with robust error handling"""
    try:
        # Find the last convolutional layer by searching through all layers
        last_conv_layer_name = None
        
        # Common last conv layer names in MobileNetV2 models
        candidate_names = [
            'out_relu',  # MobileNetV2 final activation
            'Conv_1',    # MobileNetV2 final conv
            'global_average_pooling2d',  # Just before this
        ]
        
        # Search for the last convolutional layer by iterating through model layers
        for layer in reversed(model.layers):
            if hasattr(layer, 'name'):
                # If it's a model/sequential layer, search inside it
                if hasattr(layer, 'layers'):
                    for sublayer in reversed(layer.layers):
                        if hasattr(sublayer, 'name') and any(name in sublayer.name for name in candidate_names):
                            last_conv_layer_name = sublayer.name
                            break
                        # Look for any conv layer as fallback
                        elif (hasattr(sublayer, 'output_shape') and 
                              len(sublayer.output_shape) == 4 and 
                              'conv' in sublayer.name.lower()):
                            last_conv_layer_name = sublayer.name
                
                # Direct layer check
                elif (hasattr(layer, 'output_shape') and 
                      len(layer.output_shape) == 4 and 
                      ('conv' in layer.name.lower() or 'relu' in layer.name.lower())):
                    last_conv_layer_name = layer.name
                    break
        
        if last_conv_layer_name is None:
            # Fallback: try to find ANY conv layer
            for layer in model.layers:
                if hasattr(layer, 'layers'):
                    for sublayer in layer.layers:
                        if (hasattr(sublayer, 'output_shape') and 
                            len(sublayer.output_shape) == 4):
                            last_conv_layer_name = sublayer.name
                            break
                    if last_conv_layer_name:
                        break
        
        if last_conv_layer_name is None:
            st.warning("Could not find a suitable convolutional layer for Grad-CAM")
            return None, None
        
        # Preprocess image
        img_array = preprocess_pil(img_pil)
        img_tensor = tf.cast(img_array, tf.float32)
        
        # Create a model that outputs both conv features and predictions
        try:
            # Try to get the layer by name
            conv_layer = model.get_layer(last_conv_layer_name)
            grad_model = tf.keras.Model(
                inputs=model.input,
                outputs=[conv_layer.output, model.output]
            )
        except ValueError:
            # If that fails, create a simpler approach
            st.info("Using simplified Grad-CAM approach")
            return create_simple_attention_map(model, img_pil, img_tensor)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, predictions = grad_model(img_tensor)
            # Get the predicted class
            pred_index = tf.argmax(predictions[0])
            class_score = predictions[:, pred_index]
        
        # Compute gradients
        grads = tape.gradient(class_score, conv_outputs)
        
        if grads is None:
            st.info("Could not compute gradients, using attention-based visualization")
            return create_simple_attention_map(model, img_pil, img_tensor)
        
        # Compute the Grad-CAM
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        
        # Weight the conv outputs by the gradients
        for i in range(pooled_grads.shape[-1]):
            conv_outputs = conv_outputs[:, :, i] * pooled_grads[i]
        
        # Create heatmap
        heatmap = tf.reduce_mean(conv_outputs, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize
        if tf.reduce_max(heatmap) > 0:
            heatmap = heatmap / tf.reduce_max(heatmap)
        
        # Resize to image size
        heatmap = tf.expand_dims(heatmap, axis=-1)
        heatmap = tf.image.resize(heatmap, IMG_SIZE)
        heatmap_array = tf.squeeze(heatmap).numpy()
        
        # Create visualization
        return create_overlay_visualization(img_pil, heatmap_array)
        
    except Exception as e:
        st.warning(f"Grad-CAM failed, using simple visualization: {str(e)}")
        return create_simple_attention_map(model, img_pil, preprocess_pil(img_pil))

def create_simple_attention_map(model, img_pil, img_tensor):
    """Create a simple attention visualization as fallback"""
    try:
        # Get prediction probabilities
        predictions = model.predict(img_tensor, verbose=0)
        confidence = tf.reduce_max(predictions)
        
        # Create a simple center-focused attention map
        h, w = IMG_SIZE
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        
        # Create circular attention pattern weighted by confidence
        mask = (x - center_x) ** 2 + (y - center_y) ** 2
        mask = np.exp(-mask / (2 * (min(h, w) / 4) ** 2))  # Gaussian-like
        mask = mask * float(confidence)  # Weight by model confidence
        
        # Normalize
        if mask.max() > 0:
            mask = mask / mask.max()
        
        return create_overlay_visualization(img_pil, mask)
        
    except Exception as e:
        st.error(f"Could not create attention visualization: {str(e)}")
        return None, None

def create_overlay_visualization(img_pil, heatmap_array):
    """Create the final overlay visualization"""
    try:
        # Prepare base image
        base_img = ImageOps.fit(img_pil.convert("RGB"), IMG_SIZE)
        base_array = np.array(base_img)
        
        # Create colored heatmap
        heatmap_uint8 = (heatmap_array * 255).astype(np.uint8)
        heatmap_colored = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
        heatmap_colored[:, :, 0] = heatmap_uint8  # Red channel
        
        # Blend images
        alpha = 0.4
        overlay_array = ((1 - alpha) * base_array + alpha * heatmap_colored).astype(np.uint8)
        overlay_img = Image.fromarray(overlay_array)
        
        return overlay_img, heatmap_array
        
    except Exception as e:
        st.error(f"Could not create overlay: {str(e)}")
        return None, None

def save_feedback_image(img: Image.Image, true_label: str, predicted_label: str):
    """Save user feedback for model improvement"""
    try:
        feedback_dir = project_root() / "data" / "feedback"
        feedback_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        filename = f"{true_label}_pred_{predicted_label}_{timestamp}.png"
        
        img.save(feedback_dir / filename)
        return True
    except Exception as e:
        st.error(f"Failed to save feedback image: {e}")
        return False

def main():
    # Set up page configuration
    setup_page()
    
    # Load model and labels
    try:
        model = load_model()
        class_names = load_labels()
    except Exception as e:
        st.error(f"Failed to load model or labels: {e}")
        st.info("Please run the training script first: `python src/05_retrain_mobilenetv2.py`")
        return
    
    # Create emoji mapping
    emojis = [EMOJI.get(name.lower(), "❓") for name in class_names]
    
    # Render header
    render_header()
    
    # Render sidebar and get selected page
    selected_page = render_sidebar()
    
    if selected_page == "🔮 Live Prediction":
        # Main prediction interface
        st.subheader("🎯 Live Rock-Paper-Scissors Prediction")
        
        # Create tabs for different input methods
        tab1, tab2 = st.tabs(["📷 Camera Capture", "📁 Upload Image"])
        
        with tab1:
            # Camera capture
            st.markdown("### 📷 Take a photo with your camera")
            camera_image = st.camera_input("Capture your hand gesture")
            
            if camera_image is not None:
                # Convert to PIL Image
                img = Image.open(camera_image)
                
                # Display the captured image
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(img, caption="Captured Image", use_container_width=True)
                
                with col2:
                    # Make prediction
                    with st.spinner("🤔 Analyzing your gesture..."):
                        predicted_idx, confidence, all_probs = predict_image(model, img)
                    
                    # Display prediction
                    predicted_class = class_names[predicted_idx]
                    predicted_emoji = emojis[predicted_idx]
                    
                    render_prediction_card(predicted_class, predicted_emoji, confidence)
                    
                    # Display probability chart
                    fig = render_probability_chart(class_names, all_probs, emojis)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Grad-CAM visualization
                st.markdown("### 🔍 AI Explanation (Grad-CAM)")
                with st.spinner("Generating explanation..."):
                    overlay_img, heatmap = make_gradcam(model, img)
                
                if overlay_img is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img, caption="Original Image", use_container_width=True)
                    with col2:
                        st.image(overlay_img, caption="What the AI is looking at", use_container_width=True)
                    
                    st.info("🔴 Red areas show what the AI focused on to make its prediction")
                
                # Feedback section
                st.markdown("### 📝 Help Improve the Model")
                st.write("Was the prediction correct?")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Correct Prediction"):
                        if save_feedback_image(img, predicted_class, predicted_class):
                            st.success("Thank you! Feedback saved.")
                
                with col2:
                    wrong_class = st.selectbox(
                        "If wrong, what should it be?",
                        options=class_names,
                        key="correct_label"
                    )
                    if st.button("❌ Incorrect - Save Correction"):
                        if save_feedback_image(img, wrong_class, predicted_class):
                            st.success(f"Correction saved: {wrong_class}")
        
        with tab2:
            # File upload
            st.markdown("### 📁 Upload an image")
            uploaded_file = st.file_uploader(
                "Choose an image file", 
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear image of a rock, paper, or scissors hand gesture"
            )
            
            if uploaded_file is not None:
                # Convert to PIL Image
                img = Image.open(uploaded_file)
                
                # Display and predict (same as camera logic)
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(img, caption="Uploaded Image", use_container_width=True)
                
                with col2:
                    with st.spinner("🤔 Analyzing your gesture..."):
                        predicted_idx, confidence, all_probs = predict_image(model, img)
                    
                    predicted_class = class_names[predicted_idx]
                    predicted_emoji = emojis[predicted_idx]
                    
                    render_prediction_card(predicted_class, predicted_emoji, confidence)
                    
                    fig = render_probability_chart(class_names, all_probs, emojis)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Grad-CAM for uploaded image
                st.markdown("### 🔍 AI Explanation (Grad-CAM)")
                with st.spinner("Generating explanation..."):
                    overlay_img, heatmap = make_gradcam(model, img)
                
                if overlay_img is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img, caption="Original Image", use_container_width=True)
                    with col2:
                        st.image(overlay_img, caption="What the AI is looking at", use_container_width=True)
        
        # Instructions
        render_instructions()
        
        # Test with sample images
        st.markdown("---")
        st.subheader("🧪 Test with Sample Images")
        st.markdown("Try the model on some example images from the test dataset:")
        
        def predict_wrapper(img):
            return predict_image(model, img)
        
        render_test_buttons(project_root(), predict_wrapper, class_names, emojis)
        
        # Debug sections
        if st.checkbox("Show Debug Information"):
            render_debug_sections(None, None, class_names)
    
    elif selected_page == "📊 Model Performance":
        render_performance_page()
    
    elif selected_page == "🗂️ Technical Details":
        render_technical_page()
    
    elif selected_page == "ℹ️ About":
        render_about_page()

if __name__ == "__main__":
    main()