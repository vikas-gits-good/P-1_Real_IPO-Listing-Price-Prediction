FROM python:3.11.14-slim
WORKDIR /app

# Copy the rest of the application code
COPY . .

# Update apt, install awscli, clean cache in one RUN to reduce layers and size
RUN apt-get update && \
    apt-get install -y --no-install-recommends awscli && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade --no-cache-dir pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", 'app.py']
