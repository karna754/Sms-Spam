# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the cache, which reduces the image size
# --trusted-host pypi.python.org: Can help avoid SSL issues in some networks
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of your application's code to the working directory
# This includes your app.py, model files (.pkl), and any other assets.
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variables (optional, but good practice)
ENV FLASK_APP=app.py

# Run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]