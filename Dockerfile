# Use a slim, secure base image
FROM python:3.13-slim

# Set environment variables for non-interactive installs and unbuffered python output
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy only the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a non-root user to run the application for better security
RUN useradd --create-home appuser
USER appuser

# Set the command to run the application
CMD ["python", "ollamacord.py"]
