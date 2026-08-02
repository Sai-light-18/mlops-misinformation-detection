# Base image — Python 3.10 slim for smaller size
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (Docker caching optimisation)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Create results directory
RUN mkdir -p results

# Expose port for Streamlit
EXPOSE 8501

# Default command — run baseline evaluation
CMD ["python", "src/baseline.py"]
