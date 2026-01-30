"""Constants for HA Video Vision integration - VIDEO ONLY with Auto-Discovery."""
from typing import Final

DOMAIN: Final = "ha_video_vision"

# =============================================================================
# PROVIDER CONFIGURATION (Video-Capable Only)
# =============================================================================
CONF_PROVIDER: Final = "provider"
CONF_API_KEY: Final = "api_key"

# Per-provider credential storage (for multi-provider switching)
CONF_PROVIDER_CONFIGS: Final = "provider_configs"

# Default provider selection (from configured providers)
CONF_DEFAULT_PROVIDER: Final = "default_provider"

# Provider choices - VIDEO ONLY (no image-only providers)
PROVIDER_LOCAL: Final = "local"
PROVIDER_GOOGLE: Final = "google"
PROVIDER_OPENROUTER: Final = "openrouter"

ALL_PROVIDERS: Final = [
    PROVIDER_LOCAL,
    PROVIDER_GOOGLE,
    PROVIDER_OPENROUTER,
]

PROVIDER_NAMES: Final = {
    PROVIDER_LOCAL: "Local vLLM (Video-Capable)",
    PROVIDER_GOOGLE: "Google Gemini (FREE - Video Support)",
    PROVIDER_OPENROUTER: "OpenRouter (Paid models with video)",
}

PROVIDER_BASE_URLS: Final = {
    PROVIDER_LOCAL: "http://localhost:1234/v1",
    PROVIDER_GOOGLE: "https://generativelanguage.googleapis.com/v1beta",
    PROVIDER_OPENROUTER: "https://openrouter.ai/api/v1",
}

PROVIDER_DEFAULT_MODELS: Final = {
    PROVIDER_LOCAL: "local-model",
    PROVIDER_GOOGLE: "gemini-2.0-flash",
    PROVIDER_OPENROUTER: "google/gemini-2.0-flash-001",
}

DEFAULT_PROVIDER: Final = PROVIDER_GOOGLE

# =============================================================================
# AI CONFIGURATION
# =============================================================================
CONF_VLLM_URL: Final = "vllm_url"
CONF_VLLM_MODEL: Final = "vllm_model"
CONF_VLLM_MAX_TOKENS: Final = "vllm_max_tokens"
CONF_VLLM_TEMPERATURE: Final = "vllm_temperature"

DEFAULT_VLLM_URL: Final = "http://localhost:1234/v1"
DEFAULT_VLLM_MODEL: Final = "local-model"
DEFAULT_VLLM_MAX_TOKENS: Final = 150
DEFAULT_VLLM_TEMPERATURE: Final = 0.2  # Lower = more deterministic

# =============================================================================
# CAMERA CONFIGURATION - AUTO-DISCOVERY
# =============================================================================
# Selected camera entity IDs from auto-discovery
CONF_SELECTED_CAMERAS: Final = "selected_cameras"
DEFAULT_SELECTED_CAMERAS: Final = []

# Voice aliases for easy voice commands
CONF_CAMERA_ALIASES: Final = "camera_aliases"
DEFAULT_CAMERA_ALIASES: Final = {}

# Camera context for natural responses (per-camera scene descriptions)
CONF_CAMERA_CONTEXTS: Final = "camera_contexts"
DEFAULT_CAMERA_CONTEXTS: Final = {}

# =============================================================================
# VIDEO SETTINGS
# =============================================================================
CONF_VIDEO_DURATION: Final = "video_duration"
CONF_VIDEO_WIDTH: Final = "video_width"
CONF_VIDEO_FPS_PERCENT: Final = "video_fps_percent"
CONF_NOTIFICATION_FRAME_POSITION: Final = "notification_frame_position"

DEFAULT_VIDEO_DURATION: Final = 3
DEFAULT_VIDEO_WIDTH: Final = 1280  # Match LLM Vision default for better detection
DEFAULT_VIDEO_FPS_PERCENT: Final = 100  # 100% of camera's native FPS
# Frame position for notification image (percentage of video duration)
# 0 = first frame (fastest), 50 = middle, 100 = last frame
DEFAULT_NOTIFICATION_FRAME_POSITION: Final = 0

# =============================================================================
# SNAPSHOT SETTINGS
# =============================================================================
CONF_SNAPSHOT_DIR: Final = "snapshot_dir"
CONF_SNAPSHOT_QUALITY: Final = "snapshot_quality"

DEFAULT_SNAPSHOT_DIR: Final = "/media/ha_video_vision"
DEFAULT_SNAPSHOT_QUALITY: Final = 85  # JPEG quality 1-100

# =============================================================================
# FACIAL RECOGNITION (LLM-based with reference photos)
# =============================================================================
CONF_FACIAL_RECOGNITION_ENABLED: Final = "facial_recognition_enabled"
CONF_FACIAL_RECOGNITION_DIRECTORY: Final = "facial_recognition_directory"
CONF_FACIAL_RECOGNITION_RESOLUTION: Final = "facial_recognition_resolution"
CONF_FACIAL_RECOGNITION_CONFIDENCE_THRESHOLD: Final = "facial_recognition_confidence_threshold"

DEFAULT_FACIAL_RECOGNITION_ENABLED: Final = False
DEFAULT_FACIAL_RECOGNITION_DIRECTORY: Final = "/config/camera_faces"  # Directory with subfolders per person
# Resolution for reference photos (0 = original/no resize, higher = sharper but more tokens)
DEFAULT_FACIAL_RECOGNITION_RESOLUTION: Final = 768  # Good balance of quality and token usage
# Minimum confidence threshold for facial recognition matches (0-100)
# This should match the prompt guidelines (70%+ is considered a valid match)
DEFAULT_FACIAL_RECOGNITION_CONFIDENCE_THRESHOLD: Final = 70

# =============================================================================
# TIMELINE (Calendar-based event history)
# =============================================================================
CONF_TIMELINE_ENABLED: Final = "timeline_enabled"
CONF_TIMELINE_RETENTION_DAYS: Final = "timeline_retention_days"

DEFAULT_TIMELINE_ENABLED: Final = True
DEFAULT_TIMELINE_RETENTION_DAYS: Final = 30

# =============================================================================
# SERVICE NAMES
# =============================================================================
SERVICE_ANALYZE_CAMERA: Final = "analyze_camera"
SERVICE_RECORD_CLIP: Final = "record_clip"
SERVICE_IDENTIFY_FACES: Final = "identify_faces"

# =============================================================================
# ATTRIBUTES
# =============================================================================
ATTR_CAMERA: Final = "camera"
ATTR_DURATION: Final = "duration"
ATTR_USER_QUERY: Final = "user_query"
ATTR_FACIAL_RECOGNITION: Final = "facial_recognition"
ATTR_REMEMBER: Final = "remember"
ATTR_FRAME_POSITION: Final = "frame_position"
ATTR_MAX_TOKENS: Final = "max_tokens"

# =============================================================================
# DETECTION KEYWORDS (for AI response parsing)
# =============================================================================
PERSON_KEYWORDS: Final = (
    "person", "people", "someone", "man", "woman", "child",
    "individual", "adult", "figure", "pedestrian", "walker",
    "visitor", "delivery", "carrier", "walking", "standing",
    "approaching", "leaving", "human", "resident", "guest"
)

ANIMAL_KEYWORDS: Final = (
    "dog", "cat", "pet", "animal", "puppy", "kitten",
    "canine", "feline", "bird", "squirrel", "rabbit",
    "deer", "raccoon", "fox", "coyote", "wildlife",
    "creature", "critter", "hound", "pup", "kitty"
)
