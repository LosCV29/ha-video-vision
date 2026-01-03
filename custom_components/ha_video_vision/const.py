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

# LEGACY: Manual RTSP config (kept for backward compatibility)
CONF_RTSP_HOST: Final = "rtsp_host"
CONF_RTSP_PORT: Final = "rtsp_port"
CONF_RTSP_USERNAME: Final = "rtsp_username"
CONF_RTSP_PASSWORD: Final = "rtsp_password"
CONF_RTSP_STREAM_TYPE: Final = "rtsp_stream_type"
CONF_CAMERAS: Final = "cameras"

DEFAULT_RTSP_HOST: Final = ""
DEFAULT_RTSP_PORT: Final = 554
DEFAULT_RTSP_USERNAME: Final = "admin"
DEFAULT_RTSP_PASSWORD: Final = ""
DEFAULT_RTSP_STREAM_TYPE: Final = "sub"
DEFAULT_CAMERAS: Final = {}

# =============================================================================
# VIDEO SETTINGS
# =============================================================================
CONF_VIDEO_DURATION: Final = "video_duration"
CONF_VIDEO_WIDTH: Final = "video_width"

DEFAULT_VIDEO_DURATION: Final = 3
DEFAULT_VIDEO_WIDTH: Final = 640

# =============================================================================
# SNAPSHOT SETTINGS
# =============================================================================
CONF_SNAPSHOT_DIR: Final = "snapshot_dir"
CONF_SNAPSHOT_QUALITY: Final = "snapshot_quality"

DEFAULT_SNAPSHOT_DIR: Final = "/media/ha_video_vision"
DEFAULT_SNAPSHOT_QUALITY: Final = 85  # JPEG quality 1-100

# =============================================================================
# FACIAL RECOGNITION (Optional - requires ha-addons/facial-recognition)
# =============================================================================
CONF_FACIAL_REC_ENABLED: Final = "facial_rec_enabled"
CONF_FACIAL_REC_URL: Final = "facial_rec_url"

DEFAULT_FACIAL_REC_ENABLED: Final = False
DEFAULT_FACIAL_REC_URL: Final = "http://localhost:8100"

# =============================================================================
# SERVICE NAMES
# =============================================================================
SERVICE_ANALYZE_CAMERA: Final = "analyze_camera"
SERVICE_RECORD_CLIP: Final = "record_clip"

# =============================================================================
# ATTRIBUTES
# =============================================================================
ATTR_CAMERA: Final = "camera"
ATTR_DURATION: Final = "duration"
ATTR_USER_QUERY: Final = "user_query"
ATTR_NOTIFY: Final = "notify"
ATTR_PROVIDER: Final = "provider"
ATTR_FACIAL_RECOGNITION: Final = "facial_recognition"
