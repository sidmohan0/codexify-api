FROM ubuntu:22.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies and Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    gpg-agent \
    dirmngr \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA and install Python 3.12
RUN apt-get update \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3.12-distutils \
    python3-pip \
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install latest Rust and Cargo using rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set Python 3.12 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --set python /usr/bin/python3.12

# Upgrade pip and install wheel
RUN python3.12 -m ensurepip --upgrade \
    && python3.12 -m pip install --upgrade pip \
    && python3.12 -m pip install wheel \
    && python3.12 -m pip install --upgrade setuptools

# Force reinstall blinker
RUN python3.12 -m pip install --ignore-installed blinker

# Copy only the requirements file first
COPY requirements.txt .

# Install dependencies using pip
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

# Copy the .env file
COPY .env .

# Copy the rest of the application
COPY src/ ./src/

# Set the Python path to include the src directory
ENV PYTHONPATH=/app/src

# Expose the port the app runs on
EXPOSE 8089 6379

# Set the working directory
WORKDIR /app

# Command to run the application
# Start Redis server and then run the application
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port 8089 --workers 2"]
