# Base image
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install Python
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev && \
    apt-get clean

# Set workdir
WORKDIR /app

# Copy app code
COPY ml_example.py /app/

# Install dependencies
RUN pip3 install --no-cache-dir pandas scikit-learn fastapi uvicorn python-multipart

# Run FastAPI instance app inside ml_example, hence ml_example:app
# host 0.0.0.0 means the app will be accessible from any IP address
# port 8000 means the app will listen on port 8000 
CMD ["uvicorn", "ml_example:app", "--host", "0.0.0.0", "--port", "8000"]
