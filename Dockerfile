# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

COPY src/05_retrain_mobilenetv2.py ./src/

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY app.py .
COPY app_ui.py .

# Copy models directory if it exists (optional)
COPY models/best_rps_mobilenetv2.keras ./models/


# Copy data directory if it exists (optional) 
# COPY data/ ./data/ 2>/dev/null || true

# Create necessary directories
RUN mkdir -p models data/raw data/processed data/real data/feedback

# Expose Streamlit port
EXPOSE 8502

# Set Streamlit environment variables for Docker
ENV STREAMLIT_SERVER_PORT=8502
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8502/_stcore/health || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]