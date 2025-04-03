FROM python:3.9-slim
WORKDIR /app

# 1. First install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Then copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt  # This installs nltk

# 3. Now download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"
RUN python -m spacy download en_core_web_sm

# 4. Finally copy app code
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
