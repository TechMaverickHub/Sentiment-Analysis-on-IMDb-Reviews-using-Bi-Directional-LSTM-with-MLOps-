# Use Python 3.11.0-slim as the base image
FROM python:3.11.0-slim

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command to run your app
CMD ["python", "main.py"]
