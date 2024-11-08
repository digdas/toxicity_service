# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy all files into /app
COPY . /app  

# Set the PYTHONPATH to include the /app directory
ENV PYTHONPATH=/app

# Install the dependencies
RUN pip install -r requirements.txt

# Run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
