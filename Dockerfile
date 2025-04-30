FROM python:3.9-slim

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg portaudio19-dev python3-pyaudio libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -U yt-dlp

# Create uploads directory and set permissions
RUN mkdir -p /app/uploads && chmod 777 /app/uploads
# Create tmp directory that will be used for YouTube downloads
RUN mkdir -p /tmp/youtube_downloads && chmod 777 /tmp/youtube_downloads

# No longer copy credentials file - will be mounted at runtime
# COPY service-account.json /app/service-account.json

# Copy the application code
COPY . .

EXPOSE 8080

# Make sure the container runs with proper permissions
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
