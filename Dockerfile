# GenAI-RAG-EEG Docker Image
# 
# Build: docker build -t genai-rag-eeg .
# Run:   docker run -it --gpus all genai-rag-eeg python main.py --mode demo

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL maintainer="Praveen Asthana <praveenasthana123@gmail.com>"
LABEL description="EEG-Based Stress Classification with RAG Explainability"
LABEL version="3.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p logs outputs results/figures models/checkpoints

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "main.py", "--mode", "demo"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(torch.__version__)" || exit 1
