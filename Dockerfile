# Use an official lightweight Python runtime (3.11 for ML stability)
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install basic system build dependencies required for C++ extensions (like FAISS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY req.txt .

# Install dependencies without caching to keep the image small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r req.txt

# Copy the rest of the application files
COPY . .

# Install local package in editable mode if setup.py is present
RUN if [ -f setup.py ]; then pip install --no-cache-dir -e .; fi

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Command to run Gunicorn bound to 0.0.0.0:7860
CMD ["gunicorn", "--timeout", "120", "--workers", "1", "--threads", "2", "-b", "0.0.0.0:7860", "app.application:app"]