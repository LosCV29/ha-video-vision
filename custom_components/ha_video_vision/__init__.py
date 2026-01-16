"""HA Video Vision - AI Camera Analysis with Auto-Discovery."""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import aiofiles
import aiohttp
import voluptuous as vol

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
    # Facial Recognition
    CONF_FACIAL_RECOGNITION_URL,
    CONF_FACIAL_RECOGNITION_CONFIDENCE,
    DEFAULT_FACIAL_RECOGNITION_URL,
    DEFAULT_FACIAL_RECOGNITION_CONFIDENCE,
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
        vol.Optional("server_url"): cv.string,
        vol.Optional("min_confidence"): vol.All(vol.Coerce(int), vol.Range(min=0, max=100)),
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

        # Run facial recognition IN PARALLEL - don't block on AI first!
        face_rec_path = result.get("face_rec_snapshot_path") or result.get("snapshot_path")
        if do_facial_recognition and result.get("success") and face_rec_path:
            server_url = config.get(CONF_FACIAL_RECOGNITION_URL, DEFAULT_FACIAL_RECOGNITION_URL)
            min_confidence = config.get(CONF_FACIAL_RECOGNITION_CONFIDENCE, DEFAULT_FACIAL_RECOGNITION_CONFIDENCE)
            face_result = await analyzer.identify_faces(face_rec_path, server_url, min_confidence)
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
        """Handle identify_faces service call."""
        image_path = call.data["image_path"]
        server_url = call.data.get("server_url") or config.get(CONF_FACIAL_RECOGNITION_URL, DEFAULT_FACIAL_RECOGNITION_URL)
        min_confidence = call.data.get("min_confidence") or config.get(CONF_FACIAL_RECOGNITION_CONFIDENCE, DEFAULT_FACIAL_RECOGNITION_CONFIDENCE)
        return await analyzer.identify_faces(image_path, server_url, min_confidence)

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

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]) -> None:
        """Initialize the analyzer."""
        self.hass = hass
        self._session = async_get_clientsession(hass)
        self._camera_cache: tuple[list[dict], float] | None = None
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

        # Video settings
        self.video_duration = config.get(CONF_VIDEO_DURATION, DEFAULT_VIDEO_DURATION)
        self.video_width = config.get(CONF_VIDEO_WIDTH, DEFAULT_VIDEO_WIDTH)
        self.video_fps_percent = config.get(CONF_VIDEO_FPS_PERCENT, DEFAULT_VIDEO_FPS_PERCENT)
        # Frame position for notification image (0=first, 50=middle, 100=last)
        self.notification_frame_position = config.get(
            CONF_NOTIFICATION_FRAME_POSITION, DEFAULT_NOTIFICATION_FRAME_POSITION
        )

        # Snapshot settings
        self.snapshot_dir = config.get(CONF_SNAPSHOT_DIR, DEFAULT_SNAPSHOT_DIR)
        self.snapshot_quality = config.get(CONF_SNAPSHOT_QUALITY, DEFAULT_SNAPSHOT_QUALITY)

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

    def _normalize_name(self, name: str) -> str:
        """Normalize a name for comparison (lowercase, remove special chars)."""
        import re
        # Lowercase, replace underscores/hyphens with spaces, remove extra spaces
        normalized = name.lower().strip()
        normalized = re.sub(r'[_\-]+', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def _get_camera_matches(self) -> list[dict]:
        """Get list of all cameras with searchable names (cached for 30s)."""
        import time
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
        self, entity_id: str, retries: int = 3, delay: float = 1.0, is_cloud_camera: bool = False
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
                # Brief pause to let the camera start processing
                await asyncio.sleep(0.5)
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
        """
        # Method 1: Standard stream source retrieval
        try:
            stream_url = await async_get_stream_source(self.hass, entity_id)
            if stream_url:
                _LOGGER.debug("Got stream URL for %s via async_get_stream_source", entity_id)
                return stream_url
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
                        return stream_source

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
                        return stream_source
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
                    # Wait a moment for stream to initialize
                    await asyncio.sleep(2)
                    # Try to get stream URL again
                    stream_url = await async_get_stream_source(self.hass, entity_id)
                    if stream_url:
                        _LOGGER.debug("Got stream URL for %s after activation", entity_id)
                        return stream_url
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
        """Build ffmpeg command with low-latency optimizations for instant recording."""
        cmd = ["ffmpeg", "-y"]

        # LOW LATENCY FLAGS - minimize time before recording starts
        # These reduce the ~1-2 second delay FFmpeg normally takes to probe the stream
        cmd.extend([
            "-fflags", "nobuffer",          # Don't buffer input - start immediately
            "-flags", "low_delay",          # Low latency decoding mode
            "-probesize", "32",             # Minimal probe size (bytes) - faster stream detection
            "-analyzeduration", "0",        # Skip duration analysis - start recording NOW
        ])

        if stream_url.startswith("rtsp://"):
            cmd.extend(["-rtsp_transport", "tcp"])

        cmd.extend([
            "-i", stream_url,
            "-t", str(duration),
            "-vf", f"scale={self.video_width}:-2",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-an",
            output_path
        ])

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
            "-q:v", "2",
            output_path
        ])

        return cmd

    async def record_clip(self, camera_input: str, duration: int = None) -> dict[str, Any]:
        """Record a video clip from camera."""
        duration = duration or self.video_duration
        
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

    async def _record_video_and_frames(
        self, entity_id: str, duration: int, frame_position: int | None = None,
        facial_recognition_frame_position: int | None = None
    ) -> tuple[bytes | None, bytes | None, bytes | None]:
        """Record video and extract frame. Simple and fast."""
        stream_url = await self._get_stream_url(entity_id)

        video_bytes = None
        frame_bytes = None
        face_rec_frame_bytes = None

        # Use configured default if not specified
        if frame_position is None:
            frame_position = self.notification_frame_position

        if not stream_url:
            # No stream URL (cloud camera like Ring/Nest) - use snapshot only
            # Use optimized cloud camera strategy with wake-up request
            _LOGGER.info(
                "Cloud camera detected: %s - using optimized snapshot mode. "
                "For video analysis, consider using ring-mqtt add-on for RTSP streaming.",
                entity_id
            )
            frame_bytes = await self._get_camera_snapshot(
                entity_id, retries=3, delay=1.0, is_cloud_camera=True
            )
            return video_bytes, frame_bytes, None

        video_path = None
        frame_path = None
        face_rec_frame_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as vf:
                video_path = vf.name
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as ff:
                frame_path = ff.name

            # Simple ffmpeg recording - no FPS probing, no complexity
            video_cmd = await self._build_ffmpeg_cmd(stream_url, duration, video_path)

            video_proc = await asyncio.create_subprocess_exec(
                *video_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )

            _, stderr = await asyncio.wait_for(video_proc.communicate(), timeout=duration + 10)

            # Check if ffmpeg succeeded
            if video_proc.returncode != 0:
                stderr_text = stderr.decode() if stderr else "No error output"
                _LOGGER.error("FFmpeg failed (code %d) for %s: %s",
                            video_proc.returncode, entity_id, stderr_text[:500])
                raise RuntimeError(f"FFmpeg failed: {stderr_text[:200]}")

            # Read the video file
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                async with aiofiles.open(video_path, 'rb') as f:
                    video_bytes = await f.read()

                # PARALLEL frame extraction for SPEED
                frame_time = duration * (frame_position / 100)

                # Build notification frame command
                frame_cmd = [
                    "ffmpeg", "-y", "-ss", str(frame_time), "-i", video_path,
                    "-frames:v", "1", "-q:v", "2", "-f", "mjpeg", frame_path
                ]

                # Start notification frame extraction
                frame_proc = await asyncio.create_subprocess_exec(
                    *frame_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
                )

                # Start face rec frame extraction IN PARALLEL if different position
                face_rec_proc = None
                if (facial_recognition_frame_position is not None and
                    facial_recognition_frame_position != frame_position):
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as frf:
                        face_rec_frame_path = frf.name
                    face_rec_frame_time = duration * (facial_recognition_frame_position / 100)
                    face_rec_cmd = [
                        "ffmpeg", "-y", "-ss", str(face_rec_frame_time), "-i", video_path,
                        "-frames:v", "1", "-q:v", "2", "-f", "mjpeg", face_rec_frame_path
                    ]
                    face_rec_proc = await asyncio.create_subprocess_exec(
                        *face_rec_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
                    )

                # Wait for BOTH extractions in parallel
                await asyncio.wait_for(frame_proc.wait(), timeout=10)
                if face_rec_proc:
                    await asyncio.wait_for(face_rec_proc.wait(), timeout=10)

                # Read frames
                if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                    async with aiofiles.open(frame_path, 'rb') as f:
                        frame_bytes = await f.read()
                if face_rec_frame_path and os.path.exists(face_rec_frame_path) and os.path.getsize(face_rec_frame_path) > 0:
                    async with aiofiles.open(face_rec_frame_path, 'rb') as f:
                        face_rec_frame_bytes = await f.read()

            return video_bytes, frame_bytes, face_rec_frame_bytes

        except Exception as e:
            _LOGGER.error("Error recording video from %s: %s", entity_id, e)
            # Try to get a snapshot as fallback
            fallback_frame = await self._get_camera_snapshot(entity_id)
            return None, fallback_frame, None
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
        duration = duration or self.video_duration

        _LOGGER.debug("Analyzing %s with %s", camera_input, self.provider)

        entity_id = self._find_camera_entity(camera_input)
        if not entity_id:
            available = ", ".join(self.selected_cameras) if self.selected_cameras else "None configured"
            return {
                "success": False,
                "error": f"Camera '{camera_input}' not found. Available: {available}"
            }

        state = self.hass.states.get(entity_id)
        friendly_name = state.attributes.get("friendly_name", entity_id) if state else entity_id
        safe_name = entity_id.replace("camera.", "").replace(".", "_")

        # CRITICAL: Start video recording IMMEDIATELY - this is the first thing we do
        # The recording captures what triggered the motion detection
        _LOGGER.debug("Starting IMMEDIATE video recording for %s", friendly_name)
        video_task = asyncio.create_task(
            self._record_video_and_frames(
                entity_id, duration, frame_position, facial_recognition_frame_position
            )
        )

        # Take snapshot IN PARALLEL with video recording (for notification image)
        # This doesn't delay the video - both happen simultaneously
        snapshot_task = asyncio.create_task(self._get_camera_snapshot(entity_id))

        # Wait for both to complete
        video_bytes, video_frame_bytes, face_rec_frame_bytes = await video_task
        immediate_snapshot = await snapshot_task

        # Save the immediate snapshot for notification
        snapshot_path = None
        if immediate_snapshot:
            snapshot_path = await self._save_snapshot_async(immediate_snapshot, safe_name)

        # Use immediate snapshot for notification display
        frame_bytes = immediate_snapshot or video_frame_bytes

        # Prepare prompt
        if user_query:
            prompt = user_query
        else:
            prompt = (
                "CAREFULLY scan the ENTIRE frame including all edges, corners, and background areas. "
                "Report ANY people, animals, or pets visible - even if small, distant, partially obscured, or at the edges. "
                "Also report moving vehicles. For people, describe their location and actions. "
                "For animals, identify the type (dog, cat, etc.) and what they are doing. "
                "Be concise (2-3 sentences). Say 'no activity' only if absolutely nothing is present."
            )

        # Save facial recognition frame separately if it exists and differs from notification frame
        face_rec_snapshot_task = None
        if face_rec_frame_bytes:
            face_rec_snapshot_task = asyncio.create_task(
                self._save_snapshot_async(face_rec_frame_bytes, f"{safe_name}_face_rec")
            )

        # Send to AI provider (face_rec snapshot saves in background)
        # Note: snapshot_path already saved from immediate snapshot above
        description, provider_used = await self._analyze_with_provider(video_bytes, frame_bytes, prompt)

        _LOGGER.debug("Analysis complete for %s", friendly_name)

        # Wait for face_rec snapshot to complete if needed
        face_rec_snapshot_path = None
        if face_rec_snapshot_task:
            face_rec_snapshot_path = await face_rec_snapshot_task

        # Check for person-related words in AI description
        # Expanded list to catch more variations of how the AI might describe people
        description_text = description or ""
        person_keywords = [
            "person", "people", "someone", "man", "woman", "child",
            "individual", "adult", "figure", "pedestrian", "walker",
            "visitor", "delivery", "carrier", "walking", "standing",
            "approaching", "leaving", "human", "resident", "guest"
        ]
        person_detected = any(
            word in description_text.lower()
            for word in person_keywords
        )

        # Check for animal-related words in AI description
        animal_keywords = [
            "dog", "cat", "pet", "animal", "puppy", "kitten",
            "canine", "feline", "bird", "squirrel", "rabbit",
            "deer", "raccoon", "fox", "coyote", "wildlife",
            "creature", "critter", "hound", "pup", "kitty"
        ]
        animal_detected = any(
            word in description_text.lower()
            for word in animal_keywords
        )

        return {
            "success": True,
            "camera": entity_id,
            "friendly_name": friendly_name,
            "description": description,
            "person_detected": person_detected,
            "animal_detected": animal_detected,
            "snapshot_path": snapshot_path,
            "snapshot_url": f"/media/local/ha_video_vision/{safe_name}_latest.jpg" if snapshot_path else None,
            # Facial recognition uses separate frame if available, otherwise falls back to notification frame
            "face_rec_snapshot_path": face_rec_snapshot_path or snapshot_path,
            "provider_used": provider_used,
            "default_provider": self.provider,
        }

    async def _analyze_with_provider(
        self, video_bytes: bytes | None, frame_bytes: bytes | None, prompt: str
    ) -> tuple[str, str]:
        """Send video/image to the configured AI provider.

        Returns: (description, provider_used)
        """
        # Get provider settings
        effective_provider, effective_model, effective_api_key = self._get_effective_provider()

        _LOGGER.debug("Sending to AI: %s", effective_provider)

        if effective_provider == PROVIDER_GOOGLE:
            result = await self._analyze_google(video_bytes, frame_bytes, prompt, effective_model, effective_api_key)
        elif effective_provider == PROVIDER_OPENROUTER:
            result = await self._analyze_openrouter(video_bytes, frame_bytes, prompt, effective_model, effective_api_key)
        elif effective_provider == PROVIDER_LOCAL:
            result = await self._analyze_local(video_bytes, frame_bytes, prompt)
        else:
            result = "Unknown provider configured"

        return result, effective_provider

    async def _analyze_google(
        self, video_bytes: bytes | None, frame_bytes: bytes | None, prompt: str,
        model: str = None, api_key: str = None
    ) -> str:
        """Analyze using Google Gemini - VIDEO ONLY."""
        # VIDEO ONLY - This integration focuses on video analysis
        if not video_bytes:
            return (
                "No video stream available for this camera. "
                "For Ring/Nest cloud cameras: Install ring-mqtt add-on and enable livestream, "
                "then verify the 'stream_source' attribute exists on the camera's Info sensor. "
                "See https://github.com/LosCV29/ha-video-vision/blob/main/docs/RING_MQTT_SETUP.md for setup guide."
            )

        # Use provided overrides or fall back to config
        model = model or self.vllm_model
        api_key = api_key or self.api_key

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

            parts = [{"text": prompt}]

            # VIDEO ONLY - no image analysis
            video_b64 = base64.b64encode(video_bytes).decode()
            parts.insert(0, {
                "inline_data": {
                    "mime_type": "video/mp4",
                    "data": video_b64
                }
            })
            
            # System instruction to prevent hallucination of identities
            system_instruction = (
                "You are a security camera analyst. Describe ONLY what you can actually see. "
                "NEVER identify or name specific people. NEVER guess identities. "
                "Only describe physical characteristics like 'a person in a red shirt' or 'an adult'. "
                "Do not make up names, do not say 'the homeowner', do not assume who anyone is."
            )

            payload = {
                "contents": [{"parts": parts}],
                "systemInstruction": {"parts": [{"text": system_instruction}]},
                "generationConfig": {
                    "temperature": self.vllm_temperature,
                    "maxOutputTokens": self.vllm_max_tokens,
                }
            }
            
            async with asyncio.timeout(60):
                async with self._session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()

                        # Handle various Gemini response structures
                        candidates = result.get("candidates", [])
                        if not candidates:
                            # Check for prompt feedback (safety blocking)
                            prompt_feedback = result.get("promptFeedback", {})
                            block_reason = prompt_feedback.get("blockReason")
                            if block_reason:
                                _LOGGER.warning("Gemini blocked request: %s", block_reason)
                                return f"Content blocked by safety filters: {block_reason}"
                            return "No response from Gemini (empty candidates)"

                        candidate = candidates[0]

                        # Check finish reason
                        finish_reason = candidate.get("finishReason", "")
                        if finish_reason == "SAFETY":
                            safety_ratings = candidate.get("safetyRatings", [])
                            _LOGGER.warning("Gemini safety block: %s", safety_ratings)
                            return "Content blocked by safety filters"

                        # Get content
                        content = candidate.get("content", {})
                        parts = content.get("parts", [])

                        if not parts:
                            _LOGGER.warning("Gemini returned empty parts. Full response: %s", result)
                            return "No text in Gemini response"

                        # Extract text from parts
                        text_parts = [p.get("text", "") for p in parts if "text" in p]
                        return "".join(text_parts) if text_parts else "No text in response"
                    else:
                        error = await response.text()
                        _LOGGER.error("Gemini error: %s", error[:500])
                        return f"Analysis failed: {response.status}"

        except Exception as e:
            _LOGGER.error("Gemini analysis error: %s", e)
            return f"Analysis error: {str(e)}"

    async def _analyze_openrouter(
        self, video_bytes: bytes | None, frame_bytes: bytes | None, prompt: str,
        model: str = None, api_key: str = None
    ) -> str:
        """Analyze using OpenRouter - VIDEO ONLY."""
        # VIDEO ONLY - This integration focuses on video analysis
        if not video_bytes:
            return (
                "No video stream available for this camera. "
                "For Ring/Nest cloud cameras: Install ring-mqtt add-on and enable livestream, "
                "then verify the 'stream_source' attribute exists on the camera's Info sensor. "
                "See https://github.com/LosCV29/ha-video-vision/blob/main/docs/RING_MQTT_SETUP.md for setup guide."
            )

        # Use provided overrides or fall back to config
        model = model or self.vllm_model
        api_key = api_key or self.api_key

        # Warn about free models not supporting video
        is_free_model = model and ":free" in model.lower()
        if is_free_model:
            _LOGGER.warning(
                "Free models on OpenRouter (%s) do NOT support video input. "
                "Use Google Gemini (free tier) for video analysis instead.",
                model
            )

        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            content = []

            # VIDEO ONLY - no image analysis
            video_b64 = base64.b64encode(video_bytes).decode()
            content.append({
                "type": "video_url",
                "video_url": {
                    "url": f"data:video/mp4;base64,{video_b64}"
                }
            })

            content.append({"type": "text", "text": prompt})

            # System message to prevent hallucination of identities
            system_message = (
                "You are a security camera analyst. Describe ONLY what you can actually see. "
                "NEVER identify or name specific people. NEVER guess identities. "
                "Only describe physical characteristics like 'a person in a red shirt' or 'an adult'. "
                "Do not make up names, do not say 'the homeowner', do not assume who anyone is."
            )

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": content}
                ],
                "max_tokens": self.vllm_max_tokens,
                "temperature": self.vllm_temperature,
                # Force routing through Google Vertex for base64 video support
                # AI Studio only supports YouTube links, Vertex supports base64
                "provider": {
                    "only": ["Google Vertex"]
                }
            }
            
            async with asyncio.timeout(60):
                async with self._session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        # Safely extract content from response
                        choices = result.get("choices", [])
                        if not choices:
                            _LOGGER.warning("OpenRouter returned empty choices: %s", result)
                            return "No response from AI (empty choices)"
                        message = choices[0].get("message", {})
                        content = message.get("content", "")
                        if not content:
                            _LOGGER.warning("OpenRouter returned empty content")
                            return "No description available from AI"
                        return content
                    else:
                        error = await response.text()
                        _LOGGER.error("OpenRouter error: %s", error[:500])
                        return f"Analysis failed: {response.status}"
                        
        except Exception as e:
            _LOGGER.error("OpenRouter analysis error: %s", e)
            return f"Analysis error: {str(e)}"

    async def _analyze_local(self, video_bytes: bytes | None, frame_bytes: bytes | None, prompt: str) -> str:
        """Analyze using local vLLM endpoint - VIDEO preferred, image fallback."""
        if not video_bytes and not frame_bytes:
            return "No video or image available for analysis"

        try:
            url = f"{self.base_url}/chat/completions"

            content = []

            # VIDEO ONLY - prefer video, fall back to image for local models that don't support video
            if video_bytes:
                video_b64 = base64.b64encode(video_bytes).decode()
                content.append({
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}
                })
            elif frame_bytes:
                # Image fallback for local models that don't support video
                image_b64 = base64.b64encode(frame_bytes).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                })

            content.append({"type": "text", "text": prompt})

            # System message to prevent hallucination - CRITICAL for accurate responses
            system_message = (
                "You are a security camera analyst. Describe ONLY what you can actually see in the video/image. "
                "NEVER identify or name specific people. NEVER guess identities. "
                "Only describe physical characteristics like 'a person in a red shirt' or 'an adult'. "
                "Do not make up names, do not say 'the homeowner', do not assume who anyone is. "
                "If you don't see any people, say so clearly. Do NOT hallucinate or imagine people who aren't there. "
                "Be accurate and conservative - only report what is clearly visible."
            )

            payload = {
                "model": self.vllm_model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": content}
                ],
                "max_tokens": self.vllm_max_tokens,
                "temperature": self.vllm_temperature,
            }
            
            async with asyncio.timeout(120):
                async with self._session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        # Safely extract content from response
                        choices = result.get("choices", [])
                        if not choices:
                            _LOGGER.warning("Local vLLM returned empty choices: %s", result)
                            return "No response from AI (empty choices)"
                        message = choices[0].get("message", {})
                        content = message.get("content", "")
                        if not content:
                            _LOGGER.warning("Local vLLM returned empty content")
                            return "No description available from AI"
                        return content
                    else:
                        error = await response.text()
                        _LOGGER.error("Local vLLM error: %s", error[:500])
                        return f"Analysis failed: {response.status}"

        except Exception as e:
            _LOGGER.error("Local vLLM error: %s", e)
            return f"Analysis error: {str(e)}"

    async def identify_faces(
        self, image_path: str, server_url: str, min_confidence: int = 50
    ) -> dict[str, Any]:
        """Identify faces using the facial recognition add-on."""
        _LOGGER.debug("Face rec: %s", image_path)

        try:
            # Read image file
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
                image_bytes = await f.read()

            image_b64 = base64.b64encode(image_bytes).decode()
            url = f"{server_url.rstrip('/')}/identify"

            async with asyncio.timeout(10):  # Reduced timeout for speed
                async with self._session.post(url, json={"image_base64": image_b64}) as response:
                    if response.status == 200:
                        result = await response.json()
                        all_people = result.get("people", [])
                        identified_people = [
                            p for p in all_people
                            if p.get("name") != "Unknown" and p.get("confidence", 0) >= min_confidence
                        ]
                        return {
                            "success": True,
                            "faces_detected": result.get("faces_detected", 0),
                            "identified_people": identified_people,
                            "summary": ", ".join([
                                f"{p['name']} ({p['confidence']}%)"
                                for p in identified_people
                            ]) if identified_people else "No known faces",
                        }
                    else:
                        error = await response.text()
                        _LOGGER.warning("Facial recognition error (%d): %s", response.status, error[:200])
                        return {
                            "success": False,
                            "error": f"Server error: {response.status}",
                            "faces_detected": 0,
                            "identified_people": [],
                            "summary": "",
                        }

        except asyncio.TimeoutError:
            _LOGGER.warning("Face rec timeout")
            return {"success": False, "error": "Timeout", "faces_detected": 0, "identified_people": [], "summary": ""}
        except Exception as e:
            _LOGGER.error("Face rec error: %s", e)
            return {"success": False, "error": str(e), "faces_detected": 0, "identified_people": [], "summary": ""}
