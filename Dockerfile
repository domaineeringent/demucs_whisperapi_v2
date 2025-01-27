# Use latest CUDA 12.6.3 for best performance
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/app/torch \
    DEMUCS_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/transformers \
    CUDA_AUTO_TUNE=1 \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0" \
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

# Pre-download models without CUDA optimization (will optimize at runtime)
RUN python3 -c "\
import os; \
import sys; \
\
try: \
    # Set environment \
    os.environ['TORCH_HOME'] = '/app/torch'; \
    os.environ['TRANSFORMERS_CACHE'] = '/app/transformers'; \
    \
    print('Downloading models (CUDA optimization will happen at runtime)...'); \
    \
    # Download Demucs model \
    print('Downloading Demucs model...'); \
    from demucs.pretrained import get_model; \
    model = get_model('htdemucs_ft'); \
    print('Demucs model downloaded successfully'); \
    \
    # Download Whisper model \
    print('Downloading Whisper model...'); \
    import whisper; \
    whisper_model = whisper.load_model('turbo'); \
    print('Whisper model downloaded successfully'); \
    \
    print('All models downloaded successfully. CUDA optimization will occur at startup.'); \
    \
except Exception as e: \
    print(f'Error during model download: {str(e)}', file=sys.stderr); \
    sys.exit(1) \
"

# Create startup script for runtime CUDA optimization
RUN echo '\
#!/usr/bin/env python3\n\
import torch\n\
import os\n\
import gc\n\
from demucs.pretrained import get_model\n\
import whisper\n\
\n\
def optimize_models():\n\
    if torch.cuda.is_available():\n\
        print("CUDA available, optimizing models...")\n\
        torch.cuda.empty_cache()\n\
        gc.collect()\n\
        \n\
        # Optimize Demucs\n\
        model = get_model("htdemucs_ft")\n\
        model.cuda()\n\
        with torch.cuda.amp.autocast():\n\
            with torch.no_grad():\n\
                dummy_input = torch.randn(2, 44100).cuda()\n\
                _ = model(dummy_input.unsqueeze(0))\n\
        torch.cuda.empty_cache()\n\
        \n\
        # Optimize Whisper\n\
        whisper_model = whisper.load_model("turbo")\n\
        dummy_audio = whisper.pad_or_trim(torch.randn(16000))\n\
        whisper_model.transcribe(dummy_audio)\n\
        torch.cuda.empty_cache()\n\
        \n\
        print("CUDA optimization complete")\n\
    else:\n\
        print("CUDA not available, running in CPU mode")\n\
\n\
if __name__ == "__main__":\n\
    optimize_models()\n\
' > /app/optimize_cuda.py && chmod +x /app/optimize_cuda.py

# Expose the FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set the default command to run the application with CUDA optimization at startup
CMD python3 /app/optimize_cuda.py && python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
