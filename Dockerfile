FROM python:3.12

# Set working directory
WORKDIR /app

# Copy only the requirements file first
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libmagic1 \
    libxml2-dev \
    libxslt1-dev \
    antiword \
    unrtf \
    poppler-utils \
    tesseract-ocr \
    flac \
    ffmpeg \
    lame \
    libmad0 \
    libsox-fmt-mp3 \
    sox \
    libjpeg-dev \
    swig \
    curl \
    redis-server \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install latest Rust and Cargo using rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Upgrade pip and install wheel
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install wheel

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies using uv
RUN uv pip install -r requirements.txt

# Now copy the rest of the application into the container
COPY . .

# Expose the port the app runs on and Redis default port
EXPOSE 8089 6379

# Command to run Redis in the background and then the application
CMD ["sh", "-c", "redis-server & python3 src/main.py"]
