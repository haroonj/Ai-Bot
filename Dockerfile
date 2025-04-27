# Use an official Python runtime as a parent image
FROM python:3.12-slim

LABEL maintainer="Haroun Jaradat jj.haroon99@gmail.com"

# --- Install System Dependencies (if any) ---
# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# PORT is set by Cloud Run, but provide a default for local testing
ENV PORT 8080

# Install system dependencies (curl is useful for health checks/debug)
# faiss-cpu should not require extensive system libraries on slim-buster/bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Install Python Dependencies ---
# Set the working directory in the container
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip
# Install requirements
RUN pip install --no-cache-dir -r requirements.txt \
    # Clean up pip cache after installation
    && rm -rf /root/.cache/pip

# --- Copy Application Code ---
# Copy the rest of the application code, including the pre-built faiss_index if available
# Ensure load_kb.py was run *before* building this image if you need the index inside.
COPY . .

# --- Configuration & Execution ---
# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Expose the main application port that Cloud Run will route to
EXPOSE ${PORT}

# Define the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]

# Optional: Default command if entrypoint fails (useful for debugging)
# CMD ["sleep", "infinity"]