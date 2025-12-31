"""
Facial Recognition API Server for Home Assistant Add-on
DeepFace-based face detection and recognition
"""

import base64
import io
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - From environment variables (set by run.sh from add-on config)
# =============================================================================

FACES_DIR = os.environ.get("FACES_DIR", "/share/faces")
DISTANCE_THRESHOLD = float(os.environ.get("DISTANCE_THRESHOLD", "0.45"))
MODEL_NAME = os.environ.get("MODEL_NAME", "Facenet512")
DETECTOR_BACKEND = os.environ.get("DETECTOR_BACKEND", "retinaface")
MIN_FACE_CONFIDENCE = float(os.environ.get("MIN_FACE_CONFIDENCE", "0.80"))
MIN_FACE_SIZE = int(os.environ.get("MIN_FACE_SIZE", "40"))
MAX_CONSIDERATION_DISTANCE = float(os.environ.get("MAX_CONSIDERATION_DISTANCE", "0.60"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8100"))

# =============================================================================
# GLOBALS
# =============================================================================

app = FastAPI(
    title="Facial Recognition API",
    description="Home Assistant Add-on for facial recognition",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

known_faces: dict[str, list[np.ndarray]] = {}
deepface = None


# =============================================================================
# MODELS
# =============================================================================

class IdentifyRequest(BaseModel):
    image_base64: str
    tolerance: float | None = None


class IdentifyResponse(BaseModel):
    success: bool
    faces_detected: int
    people: list[dict[str, Any]]
    summary: str
    error: str | None = None


class StatusResponse(BaseModel):
    status: str
    known_people: list[str]
    total_embeddings: int
    faces_dir: str
    model: str
    threshold: float
    detector: str
    min_confidence: float
    min_face_size: int


# =============================================================================
# FACE RECOGNITION FUNCTIONS
# =============================================================================

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine distance between two vectors."""
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    return 1 - similarity


def load_known_faces() -> dict[str, list[np.ndarray]]:
    """Load all reference faces from the faces directory."""
    global deepface

    try:
        from deepface import DeepFace
        deepface = DeepFace
        logger.info(f"DeepFace loaded successfully, using {MODEL_NAME} model")
    except ImportError as e:
        logger.error(f"DeepFace not installed: {e}")
        return {}

    faces = {}
    faces_path = Path(FACES_DIR)

    if not faces_path.exists():
        logger.warning(f"Faces directory does not exist: {FACES_DIR}")
        logger.info(f"Creating directory: {FACES_DIR}")
        faces_path.mkdir(parents=True, exist_ok=True)
        logger.info("Add photos to /share/faces/PersonName/ folders")
        return {}

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for person_dir in faces_path.iterdir():
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        embeddings = []

        for image_file in person_dir.iterdir():
            if image_file.suffix.lower() not in image_extensions:
                continue

            try:
                logger.info(f"Loading {image_file.name} for {person_name}...")

                result = deepface.represent(
                    img_path=str(image_file),
                    model_name=MODEL_NAME,
                    enforce_detection=True,
                    detector_backend=DETECTOR_BACKEND
                )

                if result and len(result) > 0:
                    embedding = np.array(result[0]["embedding"])
                    embeddings.append(embedding)
                    logger.info(f"  ✓ Loaded embedding from {image_file.name}")
                else:
                    logger.warning(f"  ✗ No face found in {image_file.name}")

            except Exception as e:
                logger.warning(f"  ✗ Error loading {image_file}: {e}")

        if embeddings:
            faces[person_name] = embeddings
            logger.info(f"Loaded {len(embeddings)} embeddings for {person_name}")

    logger.info(f"Total: {len(faces)} people loaded")
    return faces


def identify_faces_in_image(image_bytes: bytes, threshold: float) -> dict[str, Any]:
    """Identify faces in an image."""
    global deepface, known_faces

    if deepface is None:
        return {
            "success": False,
            "faces_detected": 0,
            "people": [],
            "summary": "DeepFace not loaded",
            "error": "DeepFace not initialized"
        }

    if not known_faces:
        return {
            "success": True,
            "faces_detected": 0,
            "people": [],
            "summary": f"No known faces loaded. Add photos to {FACES_DIR}/PersonName/",
            "error": None
        }

    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))

        if img.mode != "RGB":
            img = img.convert("RGB")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            img.save(tmp_path, "JPEG")

        try:
            # Extract faces
            logger.info(f"Extracting faces with {DETECTOR_BACKEND} detector...")
            face_objs = deepface.extract_faces(
                img_path=tmp_path,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True
            )

            logger.info(f"Raw detections: {len(face_objs)} potential faces found")

            # Filter faces
            valid_faces = []
            for i, face_obj in enumerate(face_objs):
                confidence = face_obj.get("confidence", 0)
                facial_area = face_obj.get("facial_area", {})
                width = facial_area.get("w", 0)
                height = facial_area.get("h", 0)

                logger.info(f"Face {i+1}: confidence={confidence:.3f}, size={width}x{height}")

                if confidence < MIN_FACE_CONFIDENCE:
                    logger.info(f"  ✗ Rejected: confidence {confidence:.3f} < {MIN_FACE_CONFIDENCE}")
                    continue

                if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
                    logger.info(f"  ✗ Rejected: size {width}x{height} < {MIN_FACE_SIZE}px")
                    continue

                if width > 0 and height > 0:
                    aspect_ratio = width / height
                    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                        logger.info(f"  ✗ Rejected: aspect ratio {aspect_ratio:.2f} out of range")
                        continue

                logger.info(f"  ✓ Accepted: valid face detection")
                valid_faces.append(face_obj)

            if not valid_faces:
                logger.info("No valid faces after filtering")
                return {
                    "success": True,
                    "faces_detected": 0,
                    "people": [],
                    "summary": "No faces detected"
                }

            # Get embeddings
            results = deepface.represent(
                img_path=tmp_path,
                model_name=MODEL_NAME,
                enforce_detection=False,
                detector_backend=DETECTOR_BACKEND
            )

        finally:
            os.unlink(tmp_path)

        if not results:
            return {
                "success": True,
                "faces_detected": 0,
                "people": [],
                "summary": "No faces detected"
            }

        results_to_process = results[:len(valid_faces)]

        # Identify each face
        identified = []
        for idx, face_result in enumerate(results_to_process):
            embedding = np.array(face_result["embedding"])

            best_match = None
            best_distance = float("inf")

            for person_name, known_embeddings in known_faces.items():
                for known_emb in known_embeddings:
                    distance = cosine_distance(embedding, known_emb)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = person_name

            logger.info(f"Face {idx+1}: best_match={best_match}, distance={best_distance:.3f}, threshold={threshold}")

            if best_match and best_distance <= threshold and best_distance <= MAX_CONSIDERATION_DISTANCE:
                confidence = round((1 - best_distance) * 100, 1)
                identified.append({
                    "name": best_match,
                    "confidence": confidence,
                    "distance": round(best_distance, 3),
                    "status": "identified"
                })
                logger.info(f"  → Identified as {best_match} ({confidence}%)")
            else:
                confidence = round((1 - best_distance) * 100, 1) if best_distance != float("inf") else 0
                identified.append({
                    "name": "Unknown",
                    "confidence": confidence,
                    "distance": round(best_distance, 3) if best_distance != float("inf") else None,
                    "status": "unknown",
                    "closest_match": best_match
                })
                logger.info(f"  → Unknown person (closest: {best_match})")

        # Build summary
        names = [p["name"] for p in identified]
        known_names = [n for n in names if n != "Unknown"]
        unknown_count = names.count("Unknown")

        if known_names and unknown_count:
            summary = f"Detected: {', '.join(known_names)} and {unknown_count} unknown"
        elif known_names:
            summary = f"Detected: {', '.join(known_names)}"
        elif unknown_count:
            summary = f"Unknown person detected" if unknown_count == 1 else f"{unknown_count} unknown people"
        else:
            summary = "No faces identified"

        return {
            "success": True,
            "faces_detected": len(valid_faces),
            "people": identified,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Error identifying faces: {e}", exc_info=True)
        return {
            "success": False,
            "faces_detected": 0,
            "people": [],
            "summary": f"Error: {str(e)}",
            "error": str(e)
        }


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load known faces on startup."""
    global known_faces
    logger.info("=" * 60)
    logger.info("Facial Recognition Add-on Starting")
    logger.info(f"Faces directory: {FACES_DIR}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Distance threshold: {DISTANCE_THRESHOLD}")
    logger.info(f"Detector: {DETECTOR_BACKEND}")
    logger.info(f"Min face confidence: {MIN_FACE_CONFIDENCE}")
    logger.info(f"Min face size: {MIN_FACE_SIZE}px")
    logger.info("=" * 60)
    known_faces = load_known_faces()
    if not known_faces:
        logger.info("")
        logger.info("No faces loaded! To add people:")
        logger.info("1. Go to /share/faces/ via Samba or SSH")
        logger.info("2. Create a folder for each person (e.g., /share/faces/John/)")
        logger.info("3. Add 3-5 photos of their face to the folder")
        logger.info("4. Restart this add-on or call POST /reload")
        logger.info("")
    logger.info("=" * 60)
    logger.info(f"Server ready on http://{HOST}:{PORT}")
    logger.info("=" * 60)


@app.get("/", response_model=StatusResponse)
@app.get("/status", response_model=StatusResponse)
async def status():
    """Get server status."""
    return StatusResponse(
        status="running",
        known_people=list(known_faces.keys()),
        total_embeddings=sum(len(e) for e in known_faces.values()),
        faces_dir=FACES_DIR,
        model=MODEL_NAME,
        threshold=DISTANCE_THRESHOLD,
        detector=DETECTOR_BACKEND,
        min_confidence=MIN_FACE_CONFIDENCE,
        min_face_size=MIN_FACE_SIZE
    )


@app.post("/identify", response_model=IdentifyResponse)
async def identify_base64(request: IdentifyRequest):
    """Identify faces in a base64-encoded image."""
    try:
        image_bytes = base64.b64decode(request.image_base64)
        threshold = request.tolerance or DISTANCE_THRESHOLD
        result = identify_faces_in_image(image_bytes, threshold)
        return IdentifyResponse(**result)
    except Exception as e:
        logger.error(f"Error in /identify: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload")
async def reload_faces():
    """Reload reference faces from disk."""
    global known_faces
    logger.info("Reloading known faces...")
    known_faces = load_known_faces()
    return {
        "success": True,
        "known_people": list(known_faces.keys()),
        "total_embeddings": sum(len(e) for e in known_faces.values())
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
