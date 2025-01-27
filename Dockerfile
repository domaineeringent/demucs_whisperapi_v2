# Use the full development CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/app/torch \
    DEMUCS_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/transformers

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    git \
    libffi-dev \
    libssl-dev \
    ffmpeg \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy application files
COPY app.py requirements.txt ./

# Create cache directories with proper permissions
RUN mkdir -p /app/torch /app/models /app/transformers /tmp && \
    chmod -R 777 /app/torch /app/models /app/transformers /tmp

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache models (this makes the image larger but startup faster)
RUN python3 -c "\
import torch; \
import os; \
os.environ['TORCH_HOME'] = '/app/torch'; \
os.environ['TRANSFORMERS_CACHE'] = '/app/transformers'; \
from demucs.pretrained import get_model; \
import whisper; \
print('Downloading and caching models...'); \
model = get_model('htdemucs_ft'); \
model.to('cuda' if torch.cuda.is_available() else 'cpu'); \
whisper.load_model('turbo'); \
print('Model caching complete.')"

# Expose the FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set the default command to run the application
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
