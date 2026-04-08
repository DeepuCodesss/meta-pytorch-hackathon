# PhantomShield X Dockerfile
FROM python:3.11-slim

LABEL maintainer="PhantomShield X Team"
LABEL description="AI Cyber Defense Training Environment — OpenEnv"

# HuggingFace Spaces runs as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install Python deps first (layer cache)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user . .

# Expose Gradio default port
EXPOSE 7860

# Environment variables (override at runtime or via HuggingFace Space secrets)
# API_BASE_URL  — OpenAI-compatible LLM endpoint base URL
# MODEL_NAME    — Model identifier for inference
# HF_TOKEN      — HuggingFace / API key
# TASK          — easy | medium | hard | all (default: all)
# VERBOSE       — 1 for verbose step output

# Health check — ensure core modules load cleanly
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s \
  CMD python -c "from environment.env import PhantomShieldEnv; PhantomShieldEnv('easy').reset(); print('OK')" || exit 1

# Default: launch Gradio UI (HuggingFace Spaces entry point — validator pings port 7860)
# To run inference only: docker run <image> python inference.py
CMD ["python", "app.py"]
