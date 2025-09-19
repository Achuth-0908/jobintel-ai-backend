# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install only minimal system deps; rely on wheels for packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy small English model
RUN python -m spacy download en_core_web_sm --quiet

# Copy project files
COPY . .

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
