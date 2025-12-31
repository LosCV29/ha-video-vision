#!/usr/bin/with-contenv bashio

# Read configuration from add-on options
DISTANCE_THRESHOLD=$(bashio::config 'distance_threshold')
MIN_FACE_CONFIDENCE=$(bashio::config 'min_face_confidence')
MIN_FACE_SIZE=$(bashio::config 'min_face_size')
MODEL_NAME=$(bashio::config 'model_name')
DETECTOR_BACKEND=$(bashio::config 'detector_backend')

# Set environment variables
export DISTANCE_THRESHOLD="${DISTANCE_THRESHOLD}"
export MIN_FACE_CONFIDENCE="${MIN_FACE_CONFIDENCE}"
export MIN_FACE_SIZE="${MIN_FACE_SIZE}"
export MODEL_NAME="${MODEL_NAME}"
export DETECTOR_BACKEND="${DETECTOR_BACKEND}"
export FACES_DIR="/share/faces"
export HOST="0.0.0.0"
export PORT="8100"

# Create faces directory if it doesn't exist
mkdir -p /share/faces

bashio::log.info "Starting Facial Recognition Server..."
bashio::log.info "Faces directory: ${FACES_DIR}"
bashio::log.info "Model: ${MODEL_NAME}"
bashio::log.info "Distance threshold: ${DISTANCE_THRESHOLD}"
bashio::log.info "Detector: ${DETECTOR_BACKEND}"

# Start the server
exec python3 /server.py
