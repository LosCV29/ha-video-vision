"""HA Video Vision - AI Camera Analysis with Auto-Discovery."""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import aiofiles
import aiohttp
import voluptuous as vol
from io import BytesIO
from PIL import Image

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.components.camera import async_get_image, async_get_stream_source

from .const import (
    DOMAIN,
    # Provider
    CONF_PROVIDER,
    CONF_API_KEY,
    CONF_PROVIDER_CONFIGS,
    CONF_DEFAULT_PROVIDER,
    PROVIDER_LOCAL,
    PROVIDER_GOOGLE,
    PROVIDER_OPENROUTER,
    PROVIDER_BASE_URLS,
    PROVIDER_DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    # AI Settings
    CONF_VLLM_URL,
    CONF_VLLM_MODEL,
    CONF_VLLM_MAX_TOKENS,
    CONF_VLLM_TEMPERATURE,
    DEFAULT_VLLM_URL,
    DEFAULT_VLLM_MODEL,
    DEFAULT_VLLM_MAX_TOKENS,
    DEFAULT_VLLM_TEMPERATURE,
    # Cameras - Auto-Discovery
    CONF_SELECTED_CAMERAS,
    DEFAULT_SELECTED_CAMERAS,
    CONF_CAMERA_ALIASES,
    DEFAULT_CAMERA_ALIASES,
    CONF_CAMERA_CONTEXTS,
    DEFAULT_CAMERA_CONTEXTS,
    # Video
    CONF_VIDEO_DURATION,
    CONF_VIDEO_WIDTH,
    CONF_VIDEO_FPS_PERCENT,
    CONF_NOTIFICATION_FRAME_POSITION,
    DEFAULT_VIDEO_DURATION,
    DEFAULT_VIDEO_WIDTH,
    DEFAULT_VIDEO_FPS_PERCENT,
    DEFAULT_NOTIFICATION_FRAME_POSITION,
    # Snapshot
    CONF_SNAPSHOT_DIR,
    CONF_SNAPSHOT_QUALITY,
    DEFAULT_SNAPSHOT_DIR,
    DEFAULT_SNAPSHOT_QUALITY,
    # Services
    SERVICE_ANALYZE_CAMERA,
    SERVICE_RECORD_CLIP,
    SERVICE_IDENTIFY_FACES,
    # Facial Recognition (LLM-based)
    CONF_FACIAL_RECOGNITION_ENABLED,
    CONF_FACIAL_RECOGNITION_DIRECTORY,
    CONF_FACIAL_RECOGNITION_RESOLUTION,
    DEFAULT_FACIAL_RECOGNITION_ENABLED,
    DEFAULT_FACIAL_RECOGNITION_DIRECTORY,
    DEFAULT_FACIAL_RECOGNITION_RESOLUTION,
    # Timeline
    CONF_TIMELINE_ENABLED,
    # Attributes
    ATTR_CAMERA,
    ATTR_DURATION,
    ATTR_USER_QUERY,
    ATTR_FACIAL_RECOGNITION,
    ATTR_FACIAL_RECOGNITION_FRAME_POSITION,
    ATTR_REMEMBER,
    ATTR_FRAME_POSITION,
    # Detection Keywords
    PERSON_KEYWORDS,
    ANIMAL_KEYWORDS,
)

_LOGGER = logging.getLogger(__name__)

# Platforms to set up
PLATFORMS = ["calendar"]

# Bundled blueprints
BLUEPRINTS = [
    {
        "domain": "automation",
        "filename": "camera_alert.yaml",
    },
]


async def async_import_blueprints(hass: HomeAssistant) -> None:
    """Import bundled blueprints to the user's blueprints directory."""
    try:
        # Get the blueprints directory in the integration
        integration_dir = Path(__file__).parent
        blueprints_source = integration_dir / "blueprints"

        # Get the target blueprints directory in config
        blueprints_target = Path(hass.config.path("blueprints"))

        for blueprint in BLUEPRINTS:
            domain = blueprint["domain"]
            filename = blueprint["filename"]

            source_file = blueprints_source / domain / filename
            target_dir = blueprints_target / domain / DOMAIN
            target_file = target_dir / filename

            if not source_file.exists():
                _LOGGER.warning("Blueprint not found: %s", source_file)
                continue

            # Create target directory if it doesn't exist (run in executor to avoid blocking)
            await hass.async_add_executor_job(
                lambda: target_dir.mkdir(parents=True, exist_ok=True)
            )

            # Copy blueprint if it doesn't exist or is outdated
            should_copy = False
            if not target_file.exists():
                should_copy = True
                _LOGGER.info("Installing blueprint: %s", filename)
            else:
                # Check if source is newer
                source_mtime = source_file.stat().st_mtime
                target_mtime = target_file.stat().st_mtime
                if source_mtime > target_mtime:
                    should_copy = True
                    _LOGGER.info("Updating blueprint: %s", filename)

            if should_copy:
                # Run blocking file copy in executor
                await hass.async_add_executor_job(
                    shutil.copy2, source_file, target_file
                )
                _LOGGER.info("Blueprint installed: %s -> %s", filename, target_file)

    except Exception as e:
        _LOGGER.warning("Failed to import blueprints: %s", e)


# Service schemas
SERVICE_ANALYZE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CAMERA): cv.string,
        vol.Optional(ATTR_DURATION, default=3): vol.All(vol.Coerce(int), vol.Range(min=1, max=10)),
        vol.Optional(ATTR_USER_QUERY, default=""): cv.string,
        vol.Optional(ATTR_FACIAL_RECOGNITION, default=False): cv.boolean,
        # Separate frame position for facial recognition (default 50% = middle of video)
        vol.Optional(ATTR_FACIAL_RECOGNITION_FRAME_POSITION): vol.All(vol.Coerce(int), vol.Range(min=0, max=100)),
        vol.Optional(ATTR_REMEMBER, default=False): cv.boolean,
        # Frame position for notification image (0=first, 50=middle, 100=last)
        vol.Optional(ATTR_FRAME_POSITION): vol.All(vol.Coerce(int), vol.Range(min=0, max=100)),
    }
)

SERVICE_RECORD_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CAMERA): cv.string,
        vol.Optional(ATTR_DURATION, default=3): vol.All(vol.Coerce(int), vol.Range(min=1, max=10)),
    }
)

SERVICE_IDENTIFY_FACES_SCHEMA = vol.Schema(
    {
        vol.Required("image_path"): cv.string,
    }
)


async def async_setup(hass: HomeAssistant, config: dict[str, Any]) -> bool:
    """Set up the HA Video Vision component."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry to new version."""
    _LOGGER.info("Migrating HA Video Vision config entry from version %s", config_entry.version)

    if config_entry.version < 4:
        new_data = {**config_entry.data}
        new_options = {**config_entry.options}
        
        # Migrate to auto-discovery: convert old camera config to selected_cameras
        if CONF_SELECTED_CAMERAS not in new_options and CONF_SELECTED_CAMERAS not in new_data:
            new_options[CONF_SELECTED_CAMERAS] = []
        
        hass.config_entries.async_update_entry(
            config_entry,
            data=new_data,
            options=new_options,
            version=4,
        )
        _LOGGER.info("Migration to version 4 (auto-discovery) successful")

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up HA Video Vision from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    # Import bundled blueprints
    await async_import_blueprints(hass)

    # Merge data and options
    config = {**entry.data, **entry.options}
    
    # Create the video analyzer instance
    analyzer = VideoAnalyzer(hass, config)
    hass.data[DOMAIN][entry.entry_id] = {
        "config": config,
        "analyzer": analyzer,
    }
    
    # Register services
    async def handle_analyze_camera(call: ServiceCall) -> dict[str, Any]:
        """Handle analyze_camera service call."""
        camera = call.data[ATTR_CAMERA]
        duration = call.data.get(ATTR_DURATION, 3)
        user_query = call.data.get(ATTR_USER_QUERY, "")
        do_facial_recognition = call.data.get(ATTR_FACIAL_RECOGNITION, False)
        do_remember = call.data.get(ATTR_REMEMBER, False)
        # Frame position for notification (None = use config default)
        frame_position = call.data.get(ATTR_FRAME_POSITION)
        # Separate frame position for facial recognition (allows capturing face at different time)
        facial_recognition_frame_position = call.data.get(ATTR_FACIAL_RECOGNITION_FRAME_POSITION)

        result = await analyzer.analyze_camera(
            camera, duration, user_query, frame_position, facial_recognition_frame_position
        )

        # Run LLM-based facial recognition if enabled
        face_rec_path = result.get("face_rec_snapshot_path") or result.get("snapshot_path")
        facial_rec_enabled = config.get(CONF_FACIAL_RECOGNITION_ENABLED, DEFAULT_FACIAL_RECOGNITION_ENABLED)
        if do_facial_recognition and facial_rec_enabled and result.get("success") and face_rec_path:
            face_result = await analyzer.identify_faces(face_rec_path)
            result["face_recognition"] = face_result

        # Save to timeline if remember is enabled
        # IMPORTANT: Wrap in try/except to ensure Timeline errors don't break notifications
        if do_remember and result.get("success"):
            timeline = hass.data[DOMAIN][entry.entry_id].get("timeline")
            if timeline:
                try:
                    await timeline.async_add_event(
                        camera_entity=result.get("camera", ""),
                        camera_name=result.get("friendly_name", ""),
                        description=result.get("description", ""),
                        snapshot_path=result.get("snapshot_path"),
                        person_detected=result.get("person_detected", False),
                        provider=result.get("provider_used"),
                    )
                    _LOGGER.debug("Saved analysis to timeline for %s", result.get("friendly_name"))
                except Exception as e:
                    _LOGGER.warning(
                        "Failed to save to timeline for %s (notifications will still be sent): %s",
                        result.get("friendly_name"), e
                    )

        return result

    async def handle_record_clip(call: ServiceCall) -> dict[str, Any]:
        """Handle record_clip service call."""
        camera = call.data[ATTR_CAMERA]
        duration = call.data.get(ATTR_DURATION, 3)

        return await analyzer.record_clip(camera, duration)

    async def handle_identify_faces(call: ServiceCall) -> dict[str, Any]:
        """Handle identify_faces service call (LLM-based facial recognition)."""
        image_path = call.data["image_path"]
        return await analyzer.identify_faces(image_path)

    # Register services with response support
    hass.services.async_register(
        DOMAIN,
        SERVICE_ANALYZE_CAMERA,
        handle_analyze_camera,
        schema=SERVICE_ANALYZE_SCHEMA,
        supports_response=True,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_RECORD_CLIP,
        handle_record_clip,
        schema=SERVICE_RECORD_SCHEMA,
        supports_response=True,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_IDENTIFY_FACES,
        handle_identify_faces,
        schema=SERVICE_IDENTIFY_FACES_SCHEMA,
        supports_response=True,
    )

    # Listen for option updates
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    # Set up calendar platform for timeline (if enabled)
    if config.get(CONF_TIMELINE_ENABLED, True):
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    _LOGGER.info("HA Video Vision (Auto-Discovery) setup complete with %d cameras",
                 len(config.get(CONF_SELECTED_CAMERAS, [])))
    return True


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    config = {**entry.data, **entry.options}
    hass.data[DOMAIN][entry.entry_id]["config"] = config
    hass.data[DOMAIN][entry.entry_id]["analyzer"].update_config(config)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # Unload platforms if they were set up
    config = {**entry.data, **entry.options}
    if config.get(CONF_TIMELINE_ENABLED, True):
        await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    # Remove services
    hass.services.async_remove(DOMAIN, SERVICE_ANALYZE_CAMERA)
    hass.services.async_remove(DOMAIN, SERVICE_RECORD_CLIP)
    hass.services.async_remove(DOMAIN, SERVICE_IDENTIFY_FACES)

    hass.data[DOMAIN].pop(entry.entry_id, None)
    return True


class VideoAnalyzer:
    """Class to handle video analysis with auto-discovered cameras."""

    CAMERA_CACHE_TTL = 300  # 5 minutes
    STREAM_URL_CACHE_TTL = 60  # 1 minute cache for stream URLs

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]) -> None:
        """Initialize the analyzer."""
        self.hass = hass
        self._session = async_get_clientsession(hass)
        self._camera_cache: tuple[list[dict], float] | None = None
        self._stream_url_cache: dict[str, tuple[str, float]] = {}  # entity_id -> (url, timestamp)
        self.update_config(config)

    def update_config(self, config: dict[str, Any]) -> None:
        """Update configuration."""
        # Provider settings - use CONF_DEFAULT_PROVIDER first, fallback to CONF_PROVIDER for legacy
        self.provider = config.get(CONF_DEFAULT_PROVIDER, config.get(CONF_PROVIDER, DEFAULT_PROVIDER))
        self.provider_configs = config.get(CONF_PROVIDER_CONFIGS, {})

        # Get config for the active/default provider
        active_config = self.provider_configs.get(self.provider, {})

        if active_config:
            # Use provider-specific config from provider_configs
            self.api_key = active_config.get("api_key", "")
            self.vllm_model = active_config.get("model", PROVIDER_DEFAULT_MODELS.get(self.provider, ""))
            self.base_url = active_config.get("base_url", PROVIDER_BASE_URLS.get(self.provider, ""))
        else:
            # Fall back to top-level config (legacy/migration support)
            self.api_key = config.get(CONF_API_KEY, "")
            self.vllm_model = config.get(CONF_VLLM_MODEL, PROVIDER_DEFAULT_MODELS.get(self.provider, DEFAULT_VLLM_MODEL))

            if self.provider == PROVIDER_LOCAL:
                self.base_url = config.get(CONF_VLLM_URL, DEFAULT_VLLM_URL)
            else:
                self.base_url = PROVIDER_BASE_URLS.get(self.provider, DEFAULT_VLLM_URL)

        # AI settings
        self.vllm_max_tokens = config.get(CONF_VLLM_MAX_TOKENS, DEFAULT_VLLM_MAX_TOKENS)
        self.vllm_temperature = config.get(CONF_VLLM_TEMPERATURE, DEFAULT_VLLM_TEMPERATURE)

        # Auto-discovered cameras (list of entity_ids)
        self.selected_cameras = config.get(CONF_SELECTED_CAMERAS, DEFAULT_SELECTED_CAMERAS)

        # Voice aliases for easy voice commands
        self.camera_aliases = config.get(CONF_CAMERA_ALIASES, DEFAULT_CAMERA_ALIASES)

        # Camera contexts for natural responses
        self.camera_contexts = config.get(CONF_CAMERA_CONTEXTS, DEFAULT_CAMERA_CONTEXTS)

        # Video settings (ensure int type for arithmetic operations)
        self.video_duration = int(config.get(CONF_VIDEO_DURATION, DEFAULT_VIDEO_DURATION))
        self.video_width = int(config.get(CONF_VIDEO_WIDTH, DEFAULT_VIDEO_WIDTH))
        self.video_fps_percent = int(config.get(CONF_VIDEO_FPS_PERCENT, DEFAULT_VIDEO_FPS_PERCENT))
        # Frame position for notification image (0=first, 50=middle, 100=last)
        self.notification_frame_position = config.get(
            CONF_NOTIFICATION_FRAME_POSITION, DEFAULT_NOTIFICATION_FRAME_POSITION
        )

        # Snapshot settings (ensure int type for quality calculations)
        self.snapshot_dir = config.get(CONF_SNAPSHOT_DIR, DEFAULT_SNAPSHOT_DIR)
        self.snapshot_quality = int(config.get(CONF_SNAPSHOT_QUALITY, DEFAULT_SNAPSHOT_QUALITY))

        # Facial recognition settings (LLM-based)
        self.facial_recognition_enabled = config.get(CONF_FACIAL_RECOGNITION_ENABLED, DEFAULT_FACIAL_RECOGNITION_ENABLED)
        self.facial_recognition_directory = config.get(CONF_FACIAL_RECOGNITION_DIRECTORY, DEFAULT_FACIAL_RECOGNITION_DIRECTORY)
        self.facial_recognition_resolution = int(config.get(CONF_FACIAL_RECOGNITION_RESOLUTION, DEFAULT_FACIAL_RECOGNITION_RESOLUTION))

        _LOGGER.info(
            "HA Video Vision config - Provider: %s, Cameras: %d, Resolution: %dp, FPS: %d%%",
            self.provider, len(self.selected_cameras), self.video_width, self.video_fps_percent
        )
        # Log configured providers
        if self.provider_configs:
            configured = [p for p, c in self.provider_configs.items() if c.get("api_key") or p == PROVIDER_LOCAL]
            _LOGGER.info("Configured providers: %s", configured)

    def _get_effective_provider(self) -> tuple[str, str, str]:
        """Get the effective provider.

        Returns: (provider, model, api_key)
        """
        return (self.provider, self.vllm_model, self.api_key)

    def _get_ffmpeg_quality(self) -> str:
        """Convert snapshot_quality (50-100) to FFmpeg JPEG quality (31-1).

        FFmpeg -q:v scale: 1 = best quality, 31 = worst quality
        Our scale: 50 = medium, 100 = best
        """
        # Linear mapping: 100 -> 1, 50 -> 31
        ffmpeg_q = max(1, min(31, int((100 - self.snapshot_quality) * 0.6 + 1)))
        return str(ffmpeg_q)

    def _normalize_name(self, name: str) -> str:
        """Normalize a name for comparison (lowercase, remove special chars)."""
        # Lowercase, replace underscores/hyphens with spaces, remove extra spaces
        normalized = name.lower().strip()
        normalized = re.sub(r'[_\-]+', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def _get_camera_matches(self) -> list[dict]:
        """Get list of all cameras with searchable names (cached for 30s)."""
        current_time = time.time()

        # Check cache first
        if self._camera_cache is not None:
            cached_list, cached_time = self._camera_cache
            if current_time - cached_time < self.CAMERA_CACHE_TTL:
                return cached_list

        # Build fresh camera list
        camera_matches = []

        # First check selected cameras
        for entity_id in self.selected_cameras:
            state = self.hass.states.get(entity_id)
            if not state:
                continue

            friendly_name = state.attributes.get("friendly_name", "")
            entity_suffix = entity_id.replace("camera.", "")

            camera_matches.append({
                "entity_id": entity_id,
                "friendly_name": friendly_name,
                "friendly_norm": self._normalize_name(friendly_name),
                "entity_suffix": entity_suffix,
                "entity_norm": self._normalize_name(entity_suffix),
            })

        # Also check all cameras (for flexibility)
        for state in self.hass.states.async_all("camera"):
            entity_id = state.entity_id
            if entity_id in self.selected_cameras:
                continue  # Already added

            friendly_name = state.attributes.get("friendly_name", "")
            entity_suffix = entity_id.replace("camera.", "")

            camera_matches.append({
                "entity_id": entity_id,
                "friendly_name": friendly_name,
                "friendly_norm": self._normalize_name(friendly_name),
                "entity_suffix": entity_suffix,
                "entity_norm": self._normalize_name(entity_suffix),
            })

        # Cache the result
        self._camera_cache = (camera_matches, current_time)
        return camera_matches

    def _find_camera_entity(self, camera_input: str) -> str | None:
        """Find camera entity ID by alias, name, entity_id, or friendly name."""
        camera_input_norm = self._normalize_name(camera_input)
        camera_input_lower = camera_input.lower().strip()

        # PRIORITY 0: Check voice aliases FIRST (no caching needed - direct dict lookup)
        for alias, entity_id in self.camera_aliases.items():
            alias_norm = self._normalize_name(alias)
            # Exact alias match
            if alias_norm == camera_input_norm:
                return entity_id
            # Alias contained in input (e.g., "backyard" in "check the backyard camera")
            if alias_norm in camera_input_norm:
                return entity_id
            # Input contained in alias
            if camera_input_norm in alias_norm:
                return entity_id

        # Get cached camera list (saves ~0.1-0.2s on repeated lookups)
        camera_matches = self._get_camera_matches()

        # Priority 1: Exact match on entity_id
        if camera_input_lower.startswith("camera."):
            for cam in camera_matches:
                if cam["entity_id"].lower() == camera_input_lower:
                    return cam["entity_id"]

        # Priority 2: Exact match on friendly name (normalized)
        for cam in camera_matches:
            if cam["friendly_norm"] == camera_input_norm:
                return cam["entity_id"]

        # Priority 3: Exact match on entity suffix (normalized)
        for cam in camera_matches:
            if cam["entity_norm"] == camera_input_norm:
                return cam["entity_id"]

        # Priority 4: Friendly name contains input OR input contains friendly name
        for cam in camera_matches:
            if camera_input_norm in cam["friendly_norm"] or cam["friendly_norm"] in camera_input_norm:
                return cam["entity_id"]

        # Priority 5: Entity suffix contains input
        for cam in camera_matches:
            if camera_input_norm in cam["entity_norm"] or cam["entity_norm"] in camera_input_norm:
                return cam["entity_id"]

        # Priority 6: Any word match (e.g., "porch" matches "Front Porch")
        input_words = set(camera_input_norm.split())
        for cam in camera_matches:
            friendly_words = set(cam["friendly_norm"].split())
            entity_words = set(cam["entity_norm"].split())

            if input_words & friendly_words:  # Any common words
                return cam["entity_id"]
            if input_words & entity_words:
                return cam["entity_id"]

        return None

    async def _get_camera_snapshot(
        self, entity_id: str, retries: int = 3, delay: float = 0.3, is_cloud_camera: bool = False
    ) -> bytes | None:
        """Get camera snapshot using HA's camera component with retry logic.

        For cloud-based cameras (Ring, Nest, etc.), the first snapshot may be stale.
        This method uses a "wake-up" request to prime the camera before capturing.

        Args:
            entity_id: Camera entity ID
            retries: Number of retry attempts
            delay: Base delay between retries (seconds)
            is_cloud_camera: If True, uses optimized cloud camera strategy
        """
        last_image = None
        last_error = None

        # For cloud cameras (Ring/Nest), send a "wake-up" request first
        # This primes the camera to start capturing fresh content
        if is_cloud_camera:
            try:
                _LOGGER.debug("Sending wake-up snapshot request to cloud camera %s", entity_id)
                await async_get_image(self.hass, entity_id)
                # Minimal pause to let the camera start processing (optimized for speed)
                await asyncio.sleep(0.2)
            except Exception as e:
                _LOGGER.debug("Wake-up request for %s failed (non-critical): %s", entity_id, e)

        for attempt in range(retries):
            try:
                if attempt > 0:
                    _LOGGER.debug(
                        "Snapshot retry %d/%d for %s (waiting %.1fs)",
                        attempt + 1, retries, entity_id, delay
                    )
                    await asyncio.sleep(delay)
                    # Linear backoff for cloud cameras (faster), exponential for local
                    if is_cloud_camera:
                        delay = min(delay + 0.5, 3.0)  # Max 3s delay for cloud
                    else:
                        delay = min(delay * 1.5, 5.0)

                image = await async_get_image(self.hass, entity_id)
                if image and image.content:
                    # Got a valid image
                    if last_image is not None:
                        if image.content != last_image:
                            # Different image from previous - this indicates fresh content!
                            _LOGGER.debug(
                                "Got fresh snapshot from %s on attempt %d (%d bytes) - content changed",
                                entity_id, attempt + 1, len(image.content)
                            )
                            return image.content
                        else:
                            # Same image as before - camera returning cached/stale image
                            _LOGGER.debug(
                                "Snapshot from %s unchanged on attempt %d, retrying...",
                                entity_id, attempt + 1
                            )
                    else:
                        # First image - for cloud cameras after wake-up, likely fresher
                        # For local cameras, verify freshness with one more request
                        if is_cloud_camera and attempt == 0:
                            _LOGGER.debug(
                                "Got snapshot from cloud camera %s (%d bytes) after wake-up",
                                entity_id, len(image.content)
                            )
                            # Accept first image after wake-up for speed
                            return image.content
                        else:
                            _LOGGER.debug(
                                "Got initial snapshot from %s (%d bytes), verifying freshness...",
                                entity_id, len(image.content)
                            )

                    # Save the image for comparison
                    last_image = image.content

            except Exception as e:
                last_error = e
                _LOGGER.debug(
                    "Snapshot attempt %d failed for %s: %s",
                    attempt + 1, entity_id, e
                )

        # All retries exhausted
        if last_error:
            _LOGGER.warning(
                "Failed to get snapshot from %s after %d attempts: %s",
                entity_id, retries, last_error
            )
        elif last_image:
            _LOGGER.debug(
                "Returning snapshot from %s (content unchanged across attempts)",
                entity_id
            )
            return last_image
        else:
            _LOGGER.warning(
                "No snapshot available from %s after %d attempts",
                entity_id, retries
            )

        return last_image

    async def _get_stream_url(self, entity_id: str) -> str | None:
        """Get RTSP/stream URL from camera entity.

        For cloud cameras (Ring, Nest, etc.), this may require activating the stream first.
        Tries multiple methods to obtain a valid stream URL.
        Uses caching to minimize latency on repeated calls.
        """
        # Check cache first for minimal latency
        import time
        now = time.time()
        if entity_id in self._stream_url_cache:
            cached_url, cached_time = self._stream_url_cache[entity_id]
            if now - cached_time < self.STREAM_URL_CACHE_TTL:
                _LOGGER.debug("Using cached stream URL for %s", entity_id)
                return cached_url

        def cache_and_return(url: str) -> str:
            """Cache the stream URL and return it."""
            self._stream_url_cache[entity_id] = (url, now)
            return url

        # Method 1: Standard stream source retrieval
        try:
            stream_url = await async_get_stream_source(self.hass, entity_id)
            if stream_url:
                # Mask credentials for logging
                masked_url = re.sub(r'://[^:]+:[^@]+@', '://****:****@', stream_url)
                _LOGGER.debug("Got stream URL for %s via async_get_stream_source: %s", entity_id, masked_url)
                return cache_and_return(stream_url)
        except Exception as e:
            _LOGGER.debug("async_get_stream_source failed for %s: %s", entity_id, e)

        # Method 2: Try to get stream URL directly from camera entity
        # Some cloud cameras expose stream_source as an attribute after activation
        try:
            state = self.hass.states.get(entity_id)
            if state and state.attributes:
                # Check for stream source in attributes (various naming conventions)
                for attr_name in ["stream_source", "rtsp_stream", "rtsp_url", "video_url"]:
                    stream_source = state.attributes.get(attr_name)
                    if stream_source and stream_source.startswith("rtsp://"):
                        _LOGGER.debug("Got stream URL for %s from %s attribute", entity_id, attr_name)
                        return cache_and_return(stream_source)

                # Check for frontend_stream_type - indicates streaming capability
                stream_type = state.attributes.get("frontend_stream_type")
                if stream_type:
                    _LOGGER.debug(
                        "Camera %s supports streaming (type: %s) but no direct URL available",
                        entity_id, stream_type
                    )
        except Exception as e:
            _LOGGER.debug("Failed to check entity attributes for %s: %s", entity_id, e)

        # Method 3: Check for ring-mqtt info sensor (stores RTSP URL separately)
        # ring-mqtt creates sensor.{name}_info with stream_source attribute
        try:
            # Extract camera name from entity_id (e.g., camera.front_door -> front_door)
            camera_name = entity_id.replace("camera.", "")

            # Try common ring-mqtt info sensor naming patterns
            info_sensor_patterns = [
                f"sensor.{camera_name}_info",
                f"sensor.{camera_name}_stream_source",
                f"sensor.ring_{camera_name}_info",
            ]

            for sensor_id in info_sensor_patterns:
                sensor_state = self.hass.states.get(sensor_id)
                if sensor_state and sensor_state.attributes:
                    stream_source = sensor_state.attributes.get("stream_source")
                    if stream_source and stream_source.startswith("rtsp://"):
                        _LOGGER.info(
                            "Got stream URL for %s from ring-mqtt info sensor %s",
                            entity_id, sensor_id
                        )
                        return cache_and_return(stream_source)
        except Exception as e:
            _LOGGER.debug("ring-mqtt info sensor check failed for %s: %s", entity_id, e)

        # Method 4: Try triggering a stream via camera.turn_on service
        # Some cloud cameras need to be "woken up" before streaming
        try:
            camera_domain = entity_id.split(".")[0]
            if camera_domain == "camera":
                # Check if camera supports turn_on
                state = self.hass.states.get(entity_id)
                if state and state.state == "idle":
                    _LOGGER.debug("Attempting to activate camera %s before stream retrieval", entity_id)
                    await self.hass.services.async_call(
                        "camera",
                        "turn_on",
                        {"entity_id": entity_id},
                        blocking=True,
                    )
                    # Brief wait for stream to initialize (optimized for speed)
                    await asyncio.sleep(0.5)
                    # Try to get stream URL again
                    stream_url = await async_get_stream_source(self.hass, entity_id)
                    if stream_url:
                        _LOGGER.debug("Got stream URL for %s after activation", entity_id)
                        return cache_and_return(stream_url)
        except Exception as e:
            _LOGGER.debug("Camera activation attempt failed for %s: %s", entity_id, e)

        _LOGGER.warning(
            "No stream URL available for %s - this is likely a cloud camera (Ring, Nest, etc.). "
            "For Ring cameras: Install ring-mqtt add-on, enable livestream, and verify the "
            "'stream_source' attribute exists on the camera's Info sensor (sensor.%s_info). "
            "See docs/RING_MQTT_SETUP.md for setup instructions.",
            entity_id, entity_id.replace("camera.", "")
        )
        return None

    async def _build_ffmpeg_cmd(self, stream_url: str, duration: int, output_path: str) -> list[str]:
        """Build ffmpeg command with ZERO-latency optimizations - every millisecond counts."""
        cmd = ["ffmpeg", "-y"]

        # ZERO LATENCY FLAGS - absolute minimum startup time
        cmd.extend([
            "-fflags", "+nobuffer+discardcorrupt+genpts",  # No buffer, start mid-stream, fix timestamps
            "-flags", "low_delay",                          # Low latency decoding
            "-probesize", "4096",                           # 4KB - bare minimum probe
            "-analyzeduration", "0",                        # Skip analysis - trust the stream
            "-max_delay", "0",                              # Zero packet delay
            "-reorder_queue_size", "0",                     # No packet reordering queue
            "-thread_queue_size", "8",                      # Minimal thread queue
        ])

        if stream_url.startswith("rtsp://"):
            # TCP is more reliable than UDP (no packet loss)
            cmd.extend(["-rtsp_transport", "tcp"])

        cmd.extend(["-i", stream_url, "-t", str(duration)])

        # Explicitly map video stream (fixes Reolink cameras that don't report pixel format)
        cmd.extend(["-map", "0:v:0"])

        # Force pixel format for streams that don't report it properly
        cmd.extend(["-pix_fmt", "yuv420p"])

        # Always re-encode for reliability (stream copy can fail on keyframe issues)
        # ultrafast + zerolatency is nearly as fast as copy but more reliable
        if self.video_fps_percent < 100:
            target_fps = max(8, int(30 * self.video_fps_percent / 100))
            cmd.extend(["-vf", f"fps={target_fps},scale={self.video_width}:-2"])
        else:
            cmd.extend(["-vf", f"scale={self.video_width}:-2"])

        cmd.extend([
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-crf", "28",
        ])

        cmd.extend(["-an", output_path])

        return cmd

    def _build_ffmpeg_frame_cmd(self, stream_url: str, output_path: str) -> list[str]:
        """Build ffmpeg command to extract a single frame."""
        cmd = ["ffmpeg", "-y"]

        if stream_url.startswith("rtsp://"):
            cmd.extend(["-rtsp_transport", "tcp"])

        cmd.extend([
            "-i", stream_url,
            "-frames:v", "1",
            "-vf", f"scale={self.video_width}:-2",
            "-q:v", self._get_ffmpeg_quality(),
            output_path
        ])

        return cmd

    async def record_clip(self, camera_input: str, duration: int = None) -> dict[str, Any]:
        """Record a video clip from camera."""
        duration = duration if duration is not None else self.video_duration
        
        entity_id = self._find_camera_entity(camera_input)
        if not entity_id:
            available = ", ".join(self.selected_cameras) if self.selected_cameras else "None configured"
            return {
                "success": False, 
                "error": f"Camera '{camera_input}' not found. Available: {available}"
            }
        
        stream_url = await self._get_stream_url(entity_id)
        if not stream_url:
            return {
                "success": False, 
                "error": f"Could not get stream URL for {entity_id}. Camera may not support streaming."
            }
        
        os.makedirs(self.snapshot_dir, exist_ok=True)
        video_path = None
        
        state = self.hass.states.get(entity_id)
        friendly_name = state.attributes.get("friendly_name", entity_id) if state else entity_id
        safe_name = entity_id.replace("camera.", "").replace(".", "_")
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=self.snapshot_dir) as vf:
                video_path = vf.name

            # Simple recording - no FPS probing
            cmd = await self._build_ffmpeg_cmd(stream_url, duration, video_path)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=duration + 10)
            
            if proc.returncode != 0:
                _LOGGER.error("FFmpeg error: %s", stderr.decode() if stderr else "Unknown")
                return {"success": False, "error": "Failed to record video"}
            
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                return {"success": False, "error": "Video file empty"}
            
            final_path = os.path.join(self.snapshot_dir, f"{safe_name}_clip.mp4")
            os.rename(video_path, final_path)
            
            return {
                "success": True,
                "camera": entity_id,
                "friendly_name": friendly_name,
                "video_path": final_path,
                "duration": duration,
            }
            
        except asyncio.TimeoutError:
            return {"success": False, "error": "Recording timed out"}
        except Exception as e:
            _LOGGER.error("Error recording clip: %s", e)
            return {"success": False, "error": str(e)}
        finally:
            if video_path and os.path.exists(video_path) and "clip.mp4" not in video_path:
                try:
                    os.remove(video_path)
                except Exception:
                    pass

    async def _record_video_and_extract_frames(
        self, entity_id: str, duration: int, frame_position: int | None = None,
        facial_recognition_frame_position: int | None = None, stream_url: str | None = None
    ) -> tuple[bytes | None, bytes | None, bytes | None]:
        """Record video and extract frames. Unified method for all video capture.

        Args:
            entity_id: Camera entity ID
            duration: Recording duration in seconds
            frame_position: Position in video for notification frame (0-100%)
            facial_recognition_frame_position: Separate position for face rec frame
            stream_url: Pre-fetched stream URL (if None, will be fetched)

        Returns:
            Tuple of (video_bytes, frame_bytes, face_rec_frame_bytes)
        """
        # Fetch stream URL if not provided
        if stream_url is None:
            stream_url = await self._get_stream_url(entity_id)

        video_bytes = None
        frame_bytes = None
        face_rec_frame_bytes = None

        # Use configured default if not specified
        if frame_position is None:
            frame_position = self.notification_frame_position

        if not stream_url:
            # No stream URL - VIDEO ONLY mode means this camera won't work
            _LOGGER.error(
                "No RTSP stream available for %s. VIDEO ONLY mode requires a streaming URL. "
                "For cloud cameras (Ring/Nest), install ring-mqtt or similar add-on for RTSP streaming.",
                entity_id
            )
            raise RuntimeError(f"No RTSP stream available for {entity_id} - video only mode requires streaming")

        video_path = None
        frame_path = None
        face_rec_frame_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as vf:
                video_path = vf.name
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as ff:
                frame_path = ff.name

            # Build and execute ffmpeg recording command - with retry for camera wake-up
            video_cmd = await self._build_ffmpeg_cmd(stream_url, duration, video_path)

            # Log the FFmpeg command (mask credentials in URL)
            masked_cmd = [re.sub(r'://[^:]+:[^@]+@', '://****:****@', str(c)) for c in video_cmd]
            _LOGGER.debug("FFmpeg command for %s: %s", entity_id, ' '.join(masked_cmd))

            # Retry logic for cameras that need to wake up from standby
            max_retries = 2
            last_error = None
            for attempt in range(max_retries + 1):
                video_proc = await asyncio.create_subprocess_exec(
                    *video_cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE
                )

                try:
                    # Timeout: recording duration + 10s buffer for connection
                    _, stderr = await asyncio.wait_for(video_proc.communicate(), timeout=duration + 10)

                    if video_proc.returncode == 0:
                        # Success - break out of retry loop
                        break
                    else:
                        stderr_text = stderr.decode() if stderr else "No error output"
                        last_error = f"Recording failed: {stderr_text[-200:]}"
                        if attempt < max_retries:
                            _LOGGER.warning("FFmpeg attempt %d failed for %s, retrying in 1s...", attempt + 1, entity_id)
                            await asyncio.sleep(1)
                        else:
                            _LOGGER.error("FFmpeg failed (code %d) for %s: %s", video_proc.returncode, entity_id, stderr_text[-500:])
                            raise RuntimeError(last_error)

                except asyncio.TimeoutError:
                    # Kill hung process
                    try:
                        video_proc.kill()
                        await video_proc.wait()
                    except Exception:
                        pass

                    last_error = "Recording timeout - camera stream not responding"
                    if attempt < max_retries:
                        _LOGGER.warning("Recording timeout for %s (attempt %d), retrying in 1s to wake camera...", entity_id, attempt + 1)
                        await asyncio.sleep(1)
                    else:
                        raise RuntimeError(last_error)

            # Verify video was actually recorded
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise RuntimeError("Recording produced empty video - no data received from camera stream")

            # Read video and extract frames
            video_size = os.path.getsize(video_path)
            _LOGGER.info("Recorded %d second video: %.1f KB", duration, video_size/1024)

            if os.path.exists(video_path) and video_size > 0:
                async with aiofiles.open(video_path, 'rb') as f:
                    video_bytes = await f.read()

                # Extract frames in parallel for speed
                frame_time = duration * (frame_position / 100)
                frame_cmd = [
                    "ffmpeg", "-y", "-ss", str(frame_time), "-i", video_path,
                    "-frames:v", "1", "-q:v", self._get_ffmpeg_quality(), "-f", "mjpeg", frame_path
                ]
                frame_proc = await asyncio.create_subprocess_exec(
                    *frame_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
                )

                # ALWAYS extract face rec frame from video (ensures sync with AI analysis)
                # Separate extraction even if same position as notification frame
                face_rec_proc = None
                effective_face_rec_position = facial_recognition_frame_position if facial_recognition_frame_position is not None else frame_position
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as frf:
                    face_rec_frame_path = frf.name
                face_rec_frame_time = duration * (effective_face_rec_position / 100)
                face_rec_cmd = [
                    "ffmpeg", "-y", "-ss", str(face_rec_frame_time), "-i", video_path,
                    "-frames:v", "1", "-q:v", self._get_ffmpeg_quality(), "-f", "mjpeg", face_rec_frame_path
                ]
                face_rec_proc = await asyncio.create_subprocess_exec(
                    *face_rec_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
                )

                # Wait for extractions (both always run now)
                await asyncio.wait_for(frame_proc.wait(), timeout=10)
                await asyncio.wait_for(face_rec_proc.wait(), timeout=10)

                # Read extracted frames (both always extracted now)
                if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                    async with aiofiles.open(frame_path, 'rb') as f:
                        frame_bytes = await f.read()
                if os.path.exists(face_rec_frame_path) and os.path.getsize(face_rec_frame_path) > 0:
                    async with aiofiles.open(face_rec_frame_path, 'rb') as f:
                        face_rec_frame_bytes = await f.read()

            return video_bytes, frame_bytes, face_rec_frame_bytes

        except Exception as e:
            _LOGGER.error("Error recording video from %s: %s", entity_id, e)
            raise  # No image fallback - video or nothing
        finally:
            for path in [video_path, frame_path, face_rec_frame_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

    async def _save_snapshot_async(self, frame_bytes: bytes, safe_name: str) -> str | None:
        """Save snapshot to disk asynchronously. Returns snapshot path or None."""
        try:
            os.makedirs(self.snapshot_dir, exist_ok=True)
            snapshot_path = os.path.join(self.snapshot_dir, f"{safe_name}_latest.jpg")
            async with aiofiles.open(snapshot_path, 'wb') as f:
                await f.write(frame_bytes)
            return snapshot_path
        except Exception as e:
            _LOGGER.error("Failed to save snapshot: %s", e)
            return None

    async def _save_video_async(self, video_bytes: bytes, safe_name: str) -> str | None:
        """Save video to disk asynchronously. Returns video path or None.

        Saves the last analyzed video as {camera}_latest.mp4 for verification/debugging.
        """
        try:
            os.makedirs(self.snapshot_dir, exist_ok=True)
            video_path = os.path.join(self.snapshot_dir, f"{safe_name}_latest.mp4")
            async with aiofiles.open(video_path, 'wb') as f:
                await f.write(video_bytes)
            video_size_kb = len(video_bytes) / 1024
            _LOGGER.info("Saved video: %s (%.1f KB)", video_path, video_size_kb)
            return video_path
        except Exception as e:
            _LOGGER.error("Failed to save video: %s", e)
            return None

    async def analyze_camera(
        self, camera_input: str, duration: int = None, user_query: str = "",
        frame_position: int | None = None, facial_recognition_frame_position: int | None = None
    ) -> dict[str, Any]:
        """Analyze camera using VIDEO and AI vision.

        Args:
            camera_input: Camera name or entity ID
            duration: Recording duration in seconds
            user_query: Custom prompt for AI analysis
            frame_position: Position in video to extract notification frame (0-100%).
                          0 = first frame, 50 = middle, 100 = last frame.
                          None = use configured default.
            facial_recognition_frame_position: Separate frame position for facial recognition.
                          If different from notification frame_position, extracts a separate frame.
                          None = use same frame as notification.
        """
        duration = duration if duration is not None else self.video_duration

        # SPEED: Find entity first (fast sync lookup)
        entity_id = self._find_camera_entity(camera_input)
        if not entity_id:
            available = ", ".join(self.selected_cameras) if self.selected_cameras else "None configured"
            return {
                "success": False,
                "error": f"Camera '{camera_input}' not found. Available: {available}"
            }

        # Get stream URL first to determine camera type
        _LOGGER.debug("Getting stream URL for %s", entity_id)
        stream_url = await self._get_stream_url(entity_id)

        # Get metadata (fast sync operations)
        state = self.hass.states.get(entity_id)
        friendly_name = state.attributes.get("friendly_name", entity_id) if state else entity_id
        safe_name = entity_id.replace("camera.", "").replace(".", "_")

        # Record video and extract frames - VIDEO ONLY mode, no snapshot fallback
        # Extracts notification frame + face_rec frame from the recorded video
        video_bytes, video_frame_bytes, face_rec_frame_bytes = await self._record_video_and_extract_frames(
            entity_id, duration, frame_position, facial_recognition_frame_position, stream_url
        )

        # Save notification frame from video
        # Using video frame ensures sync with AI analysis content
        snapshot_path = None
        if video_frame_bytes:
            snapshot_path = await self._save_snapshot_async(video_frame_bytes, safe_name)

        # Save video file for verification/debugging (overwrites previous)
        video_save_path = None
        if video_bytes:
            video_save_path = await self._save_video_async(video_bytes, safe_name)

        # Frame for any image-based operations
        frame_bytes = video_frame_bytes

        # Prepare prompt
        if user_query:
            prompt = user_query
        else:
            prompt = (
                "CAREFULLY scan the ENTIRE frame including all edges, corners, and background areas. "
                "Report ANY people, animals, or pets visible - even if small, distant, partially obscured, or at the edges. "
                "Also report moving vehicles. For people, always describe their appearance, location, and what they are doing. "
                "IMPORTANT: When multiple vehicles are present, distinguish them by COLOR first (e.g., 'blue SUV' vs 'silver SUV'). "
                "Be precise about which specific vehicle a person is interacting with - identify it by its color and position in frame. "
                "For animals, identify the type (dog, cat, etc.) and what they are doing. "
                "Be concise (2-3 sentences)."
            )

        # Save facial recognition frame (extracted from video - VIDEO ONLY mode)
        face_rec_snapshot_task = None
        if face_rec_frame_bytes:
            face_rec_snapshot_task = asyncio.create_task(
                self._save_snapshot_async(face_rec_frame_bytes, f"{safe_name}_face_rec")
            )
        else:
            _LOGGER.warning("No facial recognition frame extracted from video for %s", entity_id)

        # Send video to AI provider for analysis
        # The AI analyzes the entire video - no separate snapshot needed
        description, provider_used = await self._analyze_with_provider(
            video_bytes, frame_bytes, prompt, entity_id
        )

        _LOGGER.debug("Analysis complete for %s", friendly_name)

        # Wait for face_rec snapshot to complete if needed
        face_rec_snapshot_path = None
        if face_rec_snapshot_task:
            face_rec_snapshot_path = await face_rec_snapshot_task

        # Check for person/animal-related words in AI description
        description_lower = (description or "").lower()
        person_detected = any(word in description_lower for word in PERSON_KEYWORDS)
        animal_detected = any(word in description_lower for word in ANIMAL_KEYWORDS)

        return {
            "success": True,
            "camera": entity_id,
            "friendly_name": friendly_name,
            "description": description,
            "person_detected": person_detected,
            "animal_detected": animal_detected,
            "snapshot_path": snapshot_path,
            "snapshot_url": f"/media/local/ha_video_vision/{safe_name}_latest.jpg" if snapshot_path else None,
            # Video saved for verification/debugging
            "video_path": video_save_path,
            "video_url": f"/media/local/ha_video_vision/{safe_name}_latest.mp4" if video_save_path else None,
            # Facial recognition frame: always extracted from video (no snapshot fallback)
            "face_rec_snapshot_path": face_rec_snapshot_path,
            "provider_used": provider_used,
            "default_provider": self.provider,
        }

    def _build_system_prompt(self, entity_id: str) -> str:
        """Build a context-aware system prompt for the given camera.

        If camera context is configured, include it in the system prompt
        to enable more natural, personalized responses.
        """
        # Get camera context if available
        camera_context = self.camera_contexts.get(entity_id, "")

        if camera_context:
            # Context-aware prompt that uses the provided information
            return (
                "You are a helpful home security assistant analyzing camera footage. "
                "Use the following context about this camera's view to provide natural, "
                "personalized descriptions. You may use the names and descriptions provided.\n\n"
                f"CAMERA CONTEXT:\n{camera_context}\n\n"
                "Based on this context, describe what you see naturally. For example, if you see "
                "someone matching a described neighbor near their described vehicle, you can say "
                "'the neighbor is by their truck' instead of 'a person is near a vehicle'. "
                "Be accurate - only use context when it clearly matches what you observe. "
                "If unsure, describe what you see without assuming identities. "
                "IMPORTANT: Be PRECISE about which vehicle a person is interacting with - "
                "always identify vehicles by their distinct COLOR to avoid confusion."
            )
        else:
            # Default prompt - modeled after LLM Vision's effective approach
            return (
                "Analyze the image and give a concise, objective event summary. "
                "Focus on people, pets, and vehicles - scan the ENTIRE frame including background, "
                "edges, and distant areas. Report ANY people visible no matter how small. "
                "Track movement and activity. Describe physical characteristics only - "
                "never assume identities. Be accurate and factual. "
                "When describing positions, be PRECISE about spatial relationships - "
                "identify objects by their COLOR and position in frame (left/right/center). "
                "If someone is near a vehicle, clearly identify WHICH vehicle by its distinct color."
            )

    # Shared error message for cloud cameras without video streams
    _NO_VIDEO_ERROR = (
        "No video stream available for this camera. "
        "For Ring/Nest cloud cameras: Install ring-mqtt add-on and enable livestream, "
        "then verify the 'stream_source' attribute exists on the camera's Info sensor. "
        "See https://github.com/LosCV29/ha-video-vision/blob/main/docs/RING_MQTT_SETUP.md for setup guide."
    )

    def _extract_openai_response(self, result: dict, provider_name: str) -> str:
        """Extract text content from OpenAI-compatible API response.

        Used by OpenRouter and Local providers which use the same response format.
        """
        choices = result.get("choices", [])
        if not choices:
            _LOGGER.warning("%s returned empty choices: %s", provider_name, result)
            return "No response from AI (empty choices)"
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not content:
            _LOGGER.warning("%s returned empty content", provider_name)
            return "No description available from AI"
        return content

    async def _make_ai_request(
        self, url: str, payload: dict, headers: dict | None = None,
        timeout: int = 60, provider_name: str = "AI"
    ) -> tuple[dict | None, str | None]:
        """Make HTTP request to AI provider with unified error handling.

        Returns: (response_json, error_message) - one will be None
        """
        try:
            async with asyncio.timeout(timeout):
                async with self._session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        return await response.json(), None
                    else:
                        error = await response.text()
                        _LOGGER.error("%s error: %s", provider_name, error[:500])
                        return None, f"Analysis failed: {response.status}"
        except Exception as e:
            _LOGGER.error("%s analysis error: %s", provider_name, e)
            return None, f"Analysis error: {str(e)}"

    async def _analyze_with_provider(
        self, video_bytes: bytes | None, frame_bytes: bytes | None, prompt: str,
        entity_id: str = ""
    ) -> tuple[str, str]:
        """Send video/image to the configured AI provider.

        Args:
            video_bytes: Video data to analyze
            frame_bytes: Fallback frame for providers that don't support video
            prompt: Analysis prompt
            entity_id: Camera entity ID for context

        Returns: (description, provider_used)
        """
        effective_provider, effective_model, effective_api_key = self._get_effective_provider()
        system_prompt = self._build_system_prompt(entity_id)

        video_size = len(video_bytes) if video_bytes else 0
        _LOGGER.info("Sending VIDEO to %s: %d bytes (%.1f KB)", effective_provider, video_size, video_size/1024)

        if effective_provider == PROVIDER_GOOGLE:
            result = await self._analyze_google(
                video_bytes, prompt, effective_model, effective_api_key, system_prompt
            )
        elif effective_provider == PROVIDER_OPENROUTER:
            result = await self._analyze_openrouter(
                video_bytes, prompt, effective_model, effective_api_key, system_prompt
            )
        elif effective_provider == PROVIDER_LOCAL:
            result = await self._analyze_local(
                video_bytes, frame_bytes, prompt, system_prompt
            )
        else:
            result = "Unknown provider configured"

        return result, effective_provider

    async def _analyze_google(
        self, video_bytes: bytes | None, prompt: str,
        model: str, api_key: str, system_prompt: str
    ) -> str:
        """Analyze using Google Gemini - VIDEO ONLY."""
        if not video_bytes:
            return self._NO_VIDEO_ERROR

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        video_b64 = base64.b64encode(video_bytes).decode()

        payload = {
            "contents": [{"parts": [
                {"inline_data": {"mime_type": "video/mp4", "data": video_b64}},
                {"text": prompt}
            ]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "temperature": self.vllm_temperature,
                "maxOutputTokens": self.vllm_max_tokens,
            }
        }

        result, error = await self._make_ai_request(url, payload, provider_name="Gemini")
        if error:
            return error

        # Parse Gemini-specific response structure
        candidates = result.get("candidates", [])
        if not candidates:
            prompt_feedback = result.get("promptFeedback", {})
            block_reason = prompt_feedback.get("blockReason")
            if block_reason:
                _LOGGER.warning("Gemini blocked request: %s", block_reason)
                return f"Content blocked by safety filters: {block_reason}"
            return "No response from Gemini (empty candidates)"

        candidate = candidates[0]
        if candidate.get("finishReason") == "SAFETY":
            _LOGGER.warning("Gemini safety block: %s", candidate.get("safetyRatings", []))
            return "Content blocked by safety filters"

        parts = candidate.get("content", {}).get("parts", [])
        if not parts:
            _LOGGER.warning("Gemini returned empty parts. Full response: %s", result)
            return "No text in Gemini response"

        text_parts = [p.get("text", "") for p in parts if "text" in p]
        return "".join(text_parts) if text_parts else "No text in response"

    async def _analyze_openrouter(
        self, video_bytes: bytes | None, prompt: str,
        model: str, api_key: str, system_prompt: str
    ) -> str:
        """Analyze using OpenRouter - VIDEO ONLY."""
        if not video_bytes:
            return self._NO_VIDEO_ERROR

        # Warn about free models not supporting video
        if model and ":free" in model.lower():
            _LOGGER.warning(
                "Free models on OpenRouter (%s) do NOT support video input. "
                "Use Google Gemini (free tier) for video analysis instead.", model
            )

        video_b64 = base64.b64encode(video_bytes).decode()
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
                    {"type": "text", "text": prompt}
                ]}
            ],
            "max_tokens": self.vllm_max_tokens,
            "temperature": self.vllm_temperature,
            "provider": {"only": ["Google Vertex"]}  # Required for base64 video support
        }

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        result, error = await self._make_ai_request(
            "https://openrouter.ai/api/v1/chat/completions", payload, headers, provider_name="OpenRouter"
        )
        return error if error else self._extract_openai_response(result, "OpenRouter")

    async def _analyze_local(
        self, video_bytes: bytes | None, frame_bytes: bytes | None,
        prompt: str, system_prompt: str
    ) -> str:
        """Analyze using local vLLM endpoint - VIDEO only, no image fallback."""
        if not video_bytes:
            return "No video available for analysis - image fallback disabled"

        # Build content with video only
        content = []
        video_b64 = base64.b64encode(video_bytes).decode()
        content.append({"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}})
        content.append({"type": "text", "text": prompt})

        payload = {
            "model": self.vllm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            "max_tokens": self.vllm_max_tokens,
            "temperature": self.vllm_temperature,
        }

        result, error = await self._make_ai_request(
            f"{self.base_url}/chat/completions", payload, timeout=120, provider_name="Local vLLM"
        )
        return error if error else self._extract_openai_response(result, "Local vLLM")

    def _scan_reference_directories(self, faces_dir: Path, max_photos: int = 3) -> dict[str, list[Path]]:
        """Scan reference photo directories synchronously (for use with asyncio.to_thread).

        Returns: {"PersonName": [Path(...), ...], ...}
        """
        result: dict[str, list[Path]] = {}
        image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

        for person_dir in faces_dir.iterdir():
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            image_files = sorted([
                f for f in person_dir.iterdir()
                if f.suffix.lower() in image_extensions
            ])[:max_photos]

            if image_files:
                result[person_name] = image_files

        return result

    async def _load_reference_photos(self, max_photos_per_person: int = 3) -> dict[str, list[bytes]]:
        """Load reference photos from the facial recognition directory.

        Directory structure: /config/camera_faces/PersonName/*.jpg
        Returns: {"PersonName": [photo_bytes, ...], ...}
        """
        reference_photos: dict[str, list[bytes]] = {}

        if not self.facial_recognition_directory:
            return reference_photos

        faces_dir = Path(self.facial_recognition_directory)
        if not faces_dir.exists():
            _LOGGER.warning("Facial recognition directory not found: %s", faces_dir)
            return reference_photos

        try:
            # Scan directories in a thread to avoid blocking the event loop
            person_files = await asyncio.to_thread(
                self._scan_reference_directories, faces_dir, max_photos_per_person
            )

            # Load photo bytes asynchronously
            for person_name, image_files in person_files.items():
                photos = []
                for img_path in image_files:
                    try:
                        async with aiofiles.open(img_path, 'rb') as f:
                            photo_bytes = await f.read()
                            photos.append(photo_bytes)
                    except Exception as e:
                        _LOGGER.debug("Failed to load reference photo %s: %s", img_path, e)

                if photos:
                    reference_photos[person_name] = photos
                    _LOGGER.debug("Loaded %d reference photos for %s", len(photos), person_name)

        except Exception as e:
            _LOGGER.error("Error loading reference photos: %s", e)

        return reference_photos

    async def identify_faces(self, image_path: str) -> dict[str, Any]:
        """Identify faces using LLM-based comparison with reference photos.

        Uses the same LLM provider configured for video analysis to compare
        the camera frame against reference photos stored in the faces directory.
        """
        _LOGGER.debug("LLM Face rec: %s", image_path)

        try:
            # Check if facial recognition is enabled
            if not self.facial_recognition_enabled:
                return {
                    "success": False,
                    "error": "Facial recognition is disabled",
                    "faces_detected": 0,
                    "identified_people": [],
                    "summary": "",
                }

            # Read camera image
            if not os.path.exists(image_path):
                _LOGGER.error("Image file NOT FOUND: %s", image_path)
                return {
                    "success": False,
                    "error": f"Image not found: {image_path}",
                    "faces_detected": 0,
                    "identified_people": [],
                    "summary": "",
                }

            async with aiofiles.open(image_path, 'rb') as f:
                camera_image = await f.read()

            # Load reference photos (up to 3 per person for all providers)
            reference_photos = await self._load_reference_photos(max_photos_per_person=3)
            if not reference_photos:
                _LOGGER.warning("No reference photos found in %s", self.facial_recognition_directory)
                return {
                    "success": True,
                    "faces_detected": 0,
                    "identified_people": [],
                    "summary": "No reference photos configured",
                }

            # Build the LLM prompt with reference photos
            people_names = list(reference_photos.keys())
            _LOGGER.debug("Comparing against %d people: %s", len(people_names), people_names)

            # Use the LLM to identify faces
            result = await self._identify_faces_with_llm(camera_image, reference_photos)

            return result

        except Exception as e:
            _LOGGER.error("LLM Face rec error: %s", e)
            return {"success": False, "error": str(e), "faces_detected": 0, "identified_people": [], "summary": ""}

    async def _identify_faces_with_llm(
        self, camera_image: bytes, reference_photos: dict[str, list[bytes]]
    ) -> dict[str, Any]:
        """Use the LLM to identify faces by comparing camera image to reference photos."""
        effective_provider, effective_model, effective_api_key = self._get_effective_provider()

        # Build prompt parts with reference photos
        people_names = list(reference_photos.keys())

        # System prompt for facial recognition
        system_prompt = (
            "You are a facial recognition assistant. You will be shown reference photos of known people, "
            "followed by a camera image. Your task is to identify if any person in the camera image "
            "matches any of the known people from the reference photos.\n\n"
            "IMPORTANT RULES:\n"
            "1. Compare facial features carefully: face shape, eyes, nose, mouth, hair, skin tone\n"
            "2. Only identify someone if you are reasonably confident (40%+ certainty)\n"
            "3. Consider lighting, angle, and image quality differences\n"
            "4. If you cannot see faces clearly or no one matches, say 'No known faces'\n\n"
            "RESPONSE FORMAT (strictly follow this):\n"
            "- If you identify someone: 'PersonName XX%' where XX is your confidence (0-100)\n"
            "- For multiple people: 'PersonName1 XX%, PersonName2 YY%'\n"
            "- If no match or no faces visible: 'No known faces'\n\n"
            "Only respond with the identification result, nothing else."
        )

        # Build the user prompt describing the reference photos
        user_prompt = f"I have reference photos for {len(people_names)} people: {', '.join(people_names)}.\n\n"
        user_prompt += "First, I'll show you the reference photos for each person, then the camera image to analyze.\n\n"

        if effective_provider == PROVIDER_GOOGLE:
            result = await self._identify_faces_google(
                camera_image, reference_photos, effective_model, effective_api_key, system_prompt, user_prompt
            )
        elif effective_provider == PROVIDER_OPENROUTER:
            result = await self._identify_faces_openrouter(
                camera_image, reference_photos, effective_model, effective_api_key, system_prompt, user_prompt
            )
        else:
            # Local provider
            result = await self._identify_faces_local(
                camera_image, reference_photos, system_prompt, user_prompt
            )

        return result

    async def _identify_faces_google(
        self, camera_image: bytes, reference_photos: dict[str, list[bytes]],
        model: str, api_key: str, system_prompt: str, user_prompt: str
    ) -> dict[str, Any]:
        """Identify faces using Google Gemini."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        # Use configured resolution (0 = original, no resize)
        res = self.facial_recognition_resolution
        quality = 90  # High quality for cloud providers

        # Build parts array with reference photos and camera image
        parts = []

        # Add reference photos with labels (optionally resized based on config)
        for person_name, photos in reference_photos.items():
            parts.append({"text": f"Reference photos of {person_name}:"})
            for photo in photos:
                processed = await asyncio.to_thread(self._resize_reference_image, photo, res, quality)
                photo_b64 = base64.b64encode(processed).decode()
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": photo_b64}})

        # Add the camera image to analyze
        parts.append({"text": "\nNow analyze this camera image and identify any matching people:"})
        processed_camera = await asyncio.to_thread(self._resize_reference_image, camera_image, res, quality)
        camera_b64 = base64.b64encode(processed_camera).decode()
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": camera_b64}})

        payload = {
            "contents": [{"parts": parts}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "temperature": 0.1,  # Low temperature for consistent identification
                "maxOutputTokens": 100,
            }
        }

        result, error = await self._make_ai_request(url, payload, provider_name="Gemini Face Rec")
        if error:
            return {"success": False, "error": error, "faces_detected": 0, "identified_people": [], "summary": ""}

        # Parse Gemini response
        candidates = result.get("candidates", [])
        if not candidates:
            return {"success": True, "faces_detected": 0, "identified_people": [], "summary": "No known faces"}

        response_text = ""
        parts = candidates[0].get("content", {}).get("parts", [])
        for p in parts:
            if "text" in p:
                response_text += p["text"]

        return self._parse_face_recognition_response(response_text, list(reference_photos.keys()))

    async def _identify_faces_openrouter(
        self, camera_image: bytes, reference_photos: dict[str, list[bytes]],
        model: str, api_key: str, system_prompt: str, user_prompt: str
    ) -> dict[str, Any]:
        """Identify faces using OpenRouter."""
        # Use configured resolution (0 = original, no resize)
        res = self.facial_recognition_resolution
        quality = 90  # High quality for cloud providers

        # Build content array with reference photos and camera image
        content = []

        # Add reference photos with labels (optionally resized based on config)
        for person_name, photos in reference_photos.items():
            content.append({"type": "text", "text": f"Reference photos of {person_name}:"})
            for photo in photos:
                processed = await asyncio.to_thread(self._resize_reference_image, photo, res, quality)
                photo_b64 = base64.b64encode(processed).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{photo_b64}"}})

        # Add the camera image to analyze
        content.append({"type": "text", "text": "\nNow analyze this camera image and identify any matching people:"})
        processed_camera = await asyncio.to_thread(self._resize_reference_image, camera_image, res, quality)
        camera_b64 = base64.b64encode(processed_camera).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{camera_b64}"}})

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            "max_tokens": 100,
            "temperature": 0.1,
        }

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        result, error = await self._make_ai_request(
            "https://openrouter.ai/api/v1/chat/completions", payload, headers, provider_name="OpenRouter Face Rec"
        )

        if error:
            return {"success": False, "error": error, "faces_detected": 0, "identified_people": [], "summary": ""}

        response_text = self._extract_openai_response(result, "OpenRouter")
        return self._parse_face_recognition_response(response_text, list(reference_photos.keys()))

    def _resize_reference_image(self, image_bytes: bytes, max_size: int = 768, quality: int = 90) -> bytes:
        """Resize reference image for facial recognition.

        Args:
            image_bytes: Original image bytes
            max_size: Maximum dimension (0 = no resize, keep original)
            quality: JPEG quality (1-100, higher = sharper)

        Returns:
            Resized image bytes, or original if max_size is 0
        """
        # If max_size is 0, return original image (sharpest possible)
        if max_size == 0:
            return image_bytes

        try:
            img = Image.open(BytesIO(image_bytes))
            # Convert to RGB if necessary (handles RGBA, P modes, etc.)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')

            # Only resize if image is larger than max_size
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Save as JPEG with configured quality
            output = BytesIO()
            img.save(output, format='JPEG', quality=quality)
            return output.getvalue()
        except Exception as e:
            _LOGGER.debug("Failed to resize image: %s, using original", e)
            return image_bytes

    async def _identify_faces_local(
        self, camera_image: bytes, reference_photos: dict[str, list[bytes]],
        system_prompt: str, user_prompt: str
    ) -> dict[str, Any]:
        """Identify faces using local vLLM endpoint."""
        # Use configured resolution (0 = original, no resize) - full quality for local
        res = self.facial_recognition_resolution
        quality = 90  # High quality matching cloud providers

        # Build content array with reference photos and camera image
        content = []

        # Add reference photos with labels (optionally resized based on config)
        for person_name, photos in reference_photos.items():
            content.append({"type": "text", "text": f"Reference photos of {person_name}:"})
            for photo in photos:
                processed = await asyncio.to_thread(self._resize_reference_image, photo, res, quality)
                photo_b64 = base64.b64encode(processed).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{photo_b64}"}})

        # Add the camera image to analyze
        content.append({"type": "text", "text": "\nNow analyze this camera image and identify any matching people:"})
        processed_camera = await asyncio.to_thread(self._resize_reference_image, camera_image, res, quality)
        camera_b64 = base64.b64encode(processed_camera).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{camera_b64}"}})

        payload = {
            "model": self.vllm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            "max_tokens": 100,
            "temperature": 0.1,
        }

        result, error = await self._make_ai_request(
            f"{self.base_url}/chat/completions", payload, timeout=120, provider_name="Local vLLM Face Rec"
        )

        if error:
            return {"success": False, "error": error, "faces_detected": 0, "identified_people": [], "summary": ""}

        response_text = self._extract_openai_response(result, "Local vLLM")
        return self._parse_face_recognition_response(response_text, list(reference_photos.keys()))

    def _parse_face_recognition_response(
        self, response_text: str, known_people: list[str]
    ) -> dict[str, Any]:
        """Parse LLM response to extract identified people and confidence.

        Expected format: "PersonName XX%" or "PersonName1 XX%, PersonName2 YY%"
        """
        response_text = response_text.strip()
        _LOGGER.debug("Face rec response: %s", response_text)

        # Check for "no faces" type responses
        no_face_phrases = ["no known faces", "no faces", "no match", "cannot identify", "unable to identify"]
        if any(phrase in response_text.lower() for phrase in no_face_phrases):
            return {
                "success": True,
                "faces_detected": 0,
                "identified_people": [],
                "summary": "No known faces",
            }

        identified_people = []

        # Parse "Name XX%" patterns
        # Match patterns like "Carlos 65%", "Carlos: 65%", "Carlos (65%)"
        pattern = r'(\w+)\s*[:(\s]*(\d+)\s*%'
        matches = re.findall(pattern, response_text, re.IGNORECASE)

        for name, confidence in matches:
            # Verify the name matches one of our known people (case-insensitive)
            matched_name = None
            for known_name in known_people:
                if name.lower() == known_name.lower():
                    matched_name = known_name
                    break

            if matched_name:
                identified_people.append({
                    "name": matched_name,
                    "confidence": int(confidence),
                })

        # Build summary in "Name XX%" format
        if identified_people:
            summary = ", ".join([
                f"{p['name'].title()} {p['confidence']}%"
                for p in identified_people
            ])
            faces_detected = len(identified_people)
        else:
            summary = "No known faces"
            faces_detected = 0

        return {
            "success": True,
            "faces_detected": faces_detected,
            "identified_people": identified_people,
            "summary": summary,
        }
