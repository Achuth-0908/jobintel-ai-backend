FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Use Gunicorn with 1 worker (low memory)
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:${PORT}", "app:app"]
