# Rock-Paper-Scissors Image Classifier App

### Access the app [here](https://rps-classifier.streamlit.app/) ☺️

This is a deep learning demo that classifies images of rock, paper, or scissors hand gestures using a fine-tuned MobileNetV2 model. The app includes Grad-CAM visualization and a human-in-the-loop flow to collect corrected samples from the UI.

## Project Overview

This project demonstrates the complete pipeline for building and deploying an image classification model with modern ML best practices:

1. **Data Collection**: Using the TensorFlow Rock-Paper-Scissors dataset
2. **Data Preprocessing**: Advanced augmentation and preprocessing pipeline
3. **Model Development**: Two-phase fine-tuning of MobileNetV2 with class weighting
4. **Deployment**: Professional Streamlit web application with modular UI
5. **Explainability**: Grad-CAM visualization for model interpretability
6. **Continuous Learning**: Human-in-the-loop data collection for model improvement

## Features

- **High-Performance Classification**: 91.4% test accuracy with balanced performance across all classes
- **Real-Time Prediction**: ~26.5ms average inference latency
- **Professional Web Interface**: Multi-page Streamlit application with modern UI
- **Explainable AI**: Grad-CAM visualization showing model attention
- **Multiple Input Methods**: Image upload, webcam capture, and test dataset samples
- **Human-in-the-Loop**: Collect corrected samples for continuous model improvement
- **Comprehensive Monitoring**: Detailed performance metrics and confusion matrices

## Model Performance

**Current Model Stats:**
- **Test Accuracy:** 91.40% (340 / 372 samples)
- **Model File:** `models/best_rps_mobilenetv2.keras` (9.1 MB)
- **Total Parameters:** 2,261,827 (1,530,243 trainable, 731,584 non-trainable)
- **Inference Latency:** ~26.5 ms per prediction (hardware dependent)

**Per-Class Performance:**
- **Paper**: 100.0% precision, 74.2% recall, 85.2% F1-score (124 samples)
- **Rock**: 80.0% precision, 100.0% recall, 88.9% F1-score (124 samples)  
- **Scissors**: 99.2% precision, 100.0% recall, 99.6% F1-score (124 samples)

## Project Structure

```
rps-classifier/
├── app.py                      # Main Streamlit application
├── app_ui.py                   # UI components module
├── requirements.txt            # Python dependencies
├── README.md                  # Project documentation
├── data/                      # Data directories
│   ├── raw/                   # Raw dataset
│   │   ├── rps/              # Training data
│   │   └── rps-test-set/     # Test data
│   └── real/                  # Human-corrected samples
├── models/                    # Trained models and artifacts
│   ├── best_rps_mobilenetv2.keras    # Optimized model (9.1MB)
│   ├── best_rps_mobilenetv2.h5       # HDF5 checkpoint (21MB)
│   ├── labels.json                   # Class labels
│   ├── metrics_report.txt            # Classification report
│   └── confusion_matrix.png          # Confusion matrix visualization
└── src/                       # Training scripts
    ├── 01_download_data.py           # Dataset download script
    ├── 02_data_preprocessing.py      # Data preprocessing
    └── 05_retrain_mobilenetv2.py     # Current training script
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Git

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd rps-classifier
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv rps_env
   source rps_env/bin/activate  # On Windows: rps_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**
   ```bash
   python src/01_download_data.py
   ```

5. **Pre-process the data:**
   ```bash
   python src/02_data_preprocessing.py
   ```

6. **Train the model:**
   ```bash
   python src/05_retrain_mobilenetv2.py
   ```

7. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

   The app will be available at http://localhost:8501

## Technical Architecture

### Model Architecture
- **Base Model**: MobileNetV2 (pre-trained on ImageNet, include_top=False)
- **Input**: 224×224×3 RGB images
- **Augmentation**: RandomFlip, RandomRotation(0.10), RandomZoom(0.15), RandomCrop(90%), RandomContrast(0.20)
- **Preprocessing**: Rescaling(1/127.5, offset=-1) for MobileNetV2 compatibility
- **Custom Head**: GlobalAveragePooling2D → Dropout(0.25) → Dense(3, softmax)

### Training Configuration
- **Phase 1**: Frozen base, 8 epochs, Adam lr=0.001
- **Phase 2**: Fine-tuning top 30 layers, 10 epochs, Adam lr=0.00001
- **Class Weighting**: 1.5x weight for scissors class
- **Optimization**: ModelCheckpoint saving best validation accuracy

### Key Technical Features
- **Grad-CAM Integration**: Visual explanation of model decisions
- **Modular UI Architecture**: Separated logic and presentation layers
- **Performance Monitoring**: Comprehensive metrics saved as classification report
- **Human-in-the-Loop**: Automated collection of corrected samples to data/real/
- **Production Optimization**: .keras format for deployment compatibility

## Usage

### Web Application Features

1. **Live Prediction Page**:
   - Upload images or use webcam capture
   - Real-time prediction with confidence scores
   - Grad-CAM visualization for explainability
   - Test with built-in dataset samples
   - Save corrected predictions for model improvement

2. **Model Performance Page**:
   - Detailed accuracy metrics and confusion matrices
   - Per-class performance analysis
   - Interactive charts and visualizations

3. **Technical Details Page**:
   - Model architecture explanations
   - Training methodology and decisions
   - Live model statistics and configurations

4. **About Page**:
   - Project overview and learning outcomes
   - Technology stack and implementation highlights

### Using the Classifier

1. **Image Upload**: Choose a clear image of a hand gesture
2. **Webcam Capture**: Take a photo using your camera
3. **View Results**: See prediction, confidence, and probability distribution
4. **Grad-CAM Analysis**: Understand what the model is focusing on
5. **Provide Feedback**: Save corrected labels to improve the model

## Performance Analysis

The model achieves excellent performance with some interesting characteristics:
- **Scissors**: Nearly perfect classification (99.6% F1-score)
- **Rock**: Perfect recall but some false positives (88.9% F1-score)  
- **Paper**: High precision but some missed detections (85.2% F1-score)

This suggests the model is conservative about paper classification, which is appropriate given the gesture's similarity to other hand positions.

## Technical Contributions

This project demonstrates advanced ML engineering practices:
- **Transfer Learning**: Effective use of pre-trained MobileNetV2
- **Data Engineering**: Robust preprocessing and augmentation pipelines
- **Model Optimization**: Balanced accuracy and inference speed
- **User Experience**: Professional interface with explainable AI
- **MLOps**: Comprehensive evaluation, monitoring, and deployment-ready structure
- **Continuous Learning**: Human-in-the-loop data collection pipeline

## Dependencies

Core technologies used:
- **TensorFlow 2.13.0**: Deep learning framework
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision utilities (optional, with headless fallback)
- **PIL/Pillow**: Image processing
- **NumPy, Pandas**: Data manipulation
- **Plotly**: Interactive visualizations
- **scikit-learn**: Evaluation metrics

See `requirements.txt` for complete dependency list.
