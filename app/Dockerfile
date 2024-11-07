# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files and install dependencies
COPY . .
RUN pip install -r requirements.txt

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
