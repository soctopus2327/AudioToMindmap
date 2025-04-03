FROM python:3.9-slim
WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    graphviz \  # System binaries
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Download NLTK/Spacy data
RUN python -c "import nltk; nltk.download('punkt')" && \
    python -m spacy download en_core_web_sm

# 4. Copy app code
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
