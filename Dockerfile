# Use the official Python image as the base
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy only requirements first (leveraging Docker cache for faster builds)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Download spaCy language model inside Docker
RUN python -m spacy download en_core_web_sm

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
