# FROM python:3.9-slim
# WORKDIR /app

# # 1. Force numpy version before other installs
# RUN pip install numpy==1.26.4

# # 2. Install system dependencies
# RUN apt-get update && apt-get install -y \
#     ffmpeg \
#     graphviz \
#     && rm -rf /var/lib/apt/lists/*

# # 3. Install remaining packages
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # 4. Download NLTK/Spacy data
# RUN python -c "import nltk; nltk.download('punkt')" && \
#     python -m spacy download en_core_web_sm

# COPY . .
# CMD ["streamlit", "run", "app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]

FROM python:3.9-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm && \
    python -c "import nltk; nltk.download('punkt')"

COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
