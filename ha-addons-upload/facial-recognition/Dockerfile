ARG BUILD_FROM
FROM ${BUILD_FROM}

# Set environment
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    deepface \
    tf-keras \
    python-multipart \
    pillow \
    numpy

# Copy application
COPY run.sh /
COPY server.py /

RUN chmod a+x /run.sh

CMD [ "/run.sh" ]
