# Use CUDA 12.1.1 for best performance
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/app/torch \
    DEMUCS_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/transformers \
    CUDA_AUTO_TUNE=1 \
    TORCH_CUDA_ARCH_LIST="8.6" \
    CUDA_MODULE_LOADING=LAZY

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

# Set up working directory and cache directories
WORKDIR /app
RUN mkdir -p /app/torch /app/models /app/transformers /tmp && \
    chmod -R 777 /app/torch /app/models /app/transformers /tmp

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .

# Pre-download, cache, and optimize models
RUN python3 -c "\
import torch; \
import os; \
import numpy as np; \
import gc; \
\
# Set environment \
os.environ['TORCH_HOME'] = '/app/torch'; \
os.environ['TRANSFORMERS_CACHE'] = '/app/transformers'; \
torch.backends.cudnn.benchmark = True; \
torch.backends.cuda.matmul.allow_tf32 = True; \
torch.backends.cudnn.allow_tf32 = True; \
\
from demucs.pretrained import get_model; \
import whisper; \
\
print('Optimizing CUDA for RTX 4000 series...'); \
if torch.cuda.is_available(): \
    # Set optimal CUDA settings \
    torch.cuda.set_device(0); \
    torch.cuda.empty_cache(); \
    gc.collect(); \
\
print('Loading and optimizing Demucs...'); \
model = get_model('htdemucs_ft'); \
if torch.cuda.is_available(): \
    model = model.cuda(); \
    # Pre-compile with different input sizes for CUDA optimization \
    sizes = [(2, 44100), (2, 88200), (2, 176400)]; \
    for size in sizes: \
        print(f'Pre-warming Demucs with size {size}...'); \
        with torch.cuda.amp.autocast(): \
            with torch.no_grad(): \
                dummy_input = torch.randn(*size).cuda(); \
                _ = model(dummy_input.unsqueeze(0)); \
        torch.cuda.empty_cache(); \
\
print('Loading and optimizing Whisper...'); \
whisper_model = whisper.load_model('turbo'); \
if torch.cuda.is_available(): \
    # Pre-warm Whisper with different audio lengths \
    lengths = [16000, 32000, 48000]; \
    for length in lengths: \
        print(f'Pre-warming Whisper with length {length}...'); \
        dummy_audio = np.random.randn(length).astype(np.float32); \
        whisper_model.transcribe(dummy_audio); \
    torch.cuda.empty_cache(); \
\
print('Model optimization complete.'); \
print(f'CUDA Memory Summary:'); \
if torch.cuda.is_available(): \
    print(torch.cuda.memory_summary()); \
"

# Expose the FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set the default command to run the application
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
