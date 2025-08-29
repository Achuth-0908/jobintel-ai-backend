# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install system dependencies (for spacy)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Flask, MongoDB, ML, spaCy, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model (English small model as example)
RUN python -m spacy download en_core_web_sm

# Copy project files
COPY . .

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
