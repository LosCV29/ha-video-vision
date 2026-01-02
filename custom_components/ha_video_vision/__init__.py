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
    # Gaming Mode
    CONF_GAMING_MODE_ENTITY,
    CONF_CLOUD_FALLBACK_PROVIDER,
    DEFAULT_GAMING_MODE_ENTITY,
    DEFAULT_CLOUD_FALLBACK_PROVIDER,
    # AI Settings
    CONF_VLLM_URL,
    CONF_VLLM_MODEL,
    CONF_VLLM_MAX_TOKENS,
    CONF_VLLM_TEMPERATURE,
    DEFAULT_VLLM_URL,
    DEFAULT_VLLM_MODEL,
    DEFAULT_VLLM_MAX_TOKENS,
    DEFAULT_VLLM_TEMPERATURE,
    # Facial Recognition
    CONF_FACIAL_REC_URL,
    CONF_FACIAL_REC_ENABLED,
    CONF_FACIAL_REC_CONFIDENCE,
    DEFAULT_FACIAL_REC_URL,
    DEFAULT_FACIAL_REC_ENABLED,
    DEFAULT_FACIAL_REC_CONFIDENCE,
    # Cameras - Auto-Discovery
    CONF_SELECTED_CAMERAS,
    DEFAULT_SELECTED_CAMERAS,
    CONF_CAMERA_ALIASES,
    DEFAULT_CAMERA_ALIASES,
    # Video
    CONF_VIDEO_DURATION,
    CONF_VIDEO_WIDTH,
    DEFAULT_VIDEO_DURATION,
    DEFAULT_VIDEO_WIDTH,
    # Snapshot
    CONF_SNAPSHOT_DIR,
    CONF_SNAPSHOT_QUALITY,
    DEFAULT_SNAPSHOT_DIR,
    DEFAULT_SNAPSHOT_QUALITY,
    # Services
    SERVICE_ANALYZE_CAMERA,
    SERVICE_RECORD_CLIP,
    SERVICE_IDENTIFY_FACES,
    # Attributes
    ATTR_CAMERA,
    ATTR_DURATION,
    ATTR_USER_QUERY,
    ATTR_IMAGE_PATH,
)

_LOGGER = logging.getLogger(__name__)

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
    }
)

SERVICE_RECORD_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CAMERA): cv.string,
        vol.Optional(ATTR_DURATION, default=3): vol.All(vol.Coerce(int), vol.Range(min=1, max=10)),
    }
)

SERVICE_IDENTIFY_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_IMAGE_PATH): cv.string,
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

        return await analyzer.analyze_camera(camera, duration, user_query)

    async def handle_record_clip(call: ServiceCall) -> dict[str, Any]:
        """Handle record_clip service call."""
        camera = call.data[ATTR_CAMERA]
        duration = call.data.get(ATTR_DURATION, 3)
        
        return await analyzer.record_clip(camera, duration)

    async def handle_identify_faces(call: ServiceCall) -> dict[str, Any]:
        """Handle identify_faces service call."""
        image_path = call.data[ATTR_IMAGE_PATH]
        
        return await analyzer.identify_faces_from_file(image_path)

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
        schema=SERVICE_IDENTIFY_SCHEMA,
        supports_response=True,
    )

    # Listen for option updates
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

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
    # Remove services
    hass.services.async_remove(DOMAIN, SERVICE_ANALYZE_CAMERA)
    hass.services.async_remove(DOMAIN, SERVICE_RECORD_CLIP)
    hass.services.async_remove(DOMAIN, SERVICE_IDENTIFY_FACES)
    
    hass.data[DOMAIN].pop(entry.entry_id, None)
    return True


class VideoAnalyzer:
    """Class to handle video analysis with auto-discovered cameras."""

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]) -> None:
        """Initialize the analyzer."""
        self.hass = hass
        self._session = async_get_clientsession(hass)
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

        # Facial recognition
        self.facial_rec_url = config.get(CONF_FACIAL_REC_URL, DEFAULT_FACIAL_REC_URL)
        self.facial_rec_enabled = config.get(CONF_FACIAL_REC_ENABLED, DEFAULT_FACIAL_REC_ENABLED)
        self.facial_rec_confidence = config.get(CONF_FACIAL_REC_CONFIDENCE, DEFAULT_FACIAL_REC_CONFIDENCE)

        # Auto-discovered cameras (list of entity_ids)
        self.selected_cameras = config.get(CONF_SELECTED_CAMERAS, DEFAULT_SELECTED_CAMERAS)

        # Voice aliases for easy voice commands
        self.camera_aliases = config.get(CONF_CAMERA_ALIASES, DEFAULT_CAMERA_ALIASES)

        # Video settings
        self.video_duration = config.get(CONF_VIDEO_DURATION, DEFAULT_VIDEO_DURATION)
        self.video_width = config.get(CONF_VIDEO_WIDTH, DEFAULT_VIDEO_WIDTH)

        # Snapshot settings
        self.snapshot_dir = config.get(CONF_SNAPSHOT_DIR, DEFAULT_SNAPSHOT_DIR)
        self.snapshot_quality = config.get(CONF_SNAPSHOT_QUALITY, DEFAULT_SNAPSHOT_QUALITY)

        # Gaming mode settings
        self.gaming_mode_entity = config.get(CONF_GAMING_MODE_ENTITY, DEFAULT_GAMING_MODE_ENTITY)
        self.cloud_fallback_provider = config.get(CONF_CLOUD_FALLBACK_PROVIDER, DEFAULT_CLOUD_FALLBACK_PROVIDER)

        _LOGGER.warning(
            "HA Video Vision config - Provider: %s, Cameras: %d, Resolution: %dp",
            self.provider, len(self.selected_cameras), self.video_width
        )
        _LOGGER.warning(
            "Gaming mode config - Entity: %s, Fallback: %s (NOTE: Only works when default provider is LOCAL)",
            self.gaming_mode_entity, self.cloud_fallback_provider
        )
        # Log configured providers
        if self.provider_configs:
            configured = [p for p, c in self.provider_configs.items() if c.get("api_key") or p == PROVIDER_LOCAL]
            _LOGGER.warning("Configured providers: %s", configured)

    def _is_gaming_mode_active(self) -> bool:
        """Check if gaming mode is active (local AI should be bypassed)."""
        if not self.gaming_mode_entity:
            _LOGGER.debug("Gaming mode entity not configured")
            return False

        state = self.hass.states.get(self.gaming_mode_entity)
        if state is None:
            # Entity doesn't exist - gaming mode not configured
            _LOGGER.warning(
                "Gaming mode entity '%s' not found - create input_boolean.gaming_mode helper",
                self.gaming_mode_entity
            )
            return False

        is_active = state.state == "on"
        _LOGGER.warning(
            "Gaming mode check: entity=%s, state=%s, active=%s",
            self.gaming_mode_entity, state.state, is_active
        )
        return is_active

    def _get_effective_provider(self) -> tuple[str, str, str]:
        """Get the effective provider, considering gaming mode.

        Returns: (provider, model, api_key)
        """
        gaming_active = self._is_gaming_mode_active()

        _LOGGER.warning(
            "Provider selection - Default: %s, Gaming mode: %s, Fallback: %s",
            self.provider, gaming_active, self.cloud_fallback_provider
        )

        # Check if we need to switch from local to cloud
        if self.provider == PROVIDER_LOCAL and gaming_active:
            fallback = self.cloud_fallback_provider
            fallback_config = self.provider_configs.get(fallback, {})

            fallback_model = fallback_config.get("model", PROVIDER_DEFAULT_MODELS.get(fallback, ""))
            fallback_api_key = fallback_config.get("api_key", "")

            if not fallback_api_key:
                _LOGGER.error(
                    "Gaming mode active but fallback provider '%s' has no API key configured! "
                    "Configure %s in Options first.",
                    fallback, fallback
                )
                # Fall back to local anyway since cloud isn't configured
                return (self.provider, self.vllm_model, self.api_key)

            _LOGGER.warning(
                "GAMING MODE ACTIVE - Switching from LOCAL to %s (model: %s)",
                fallback, fallback_model
            )

            return (fallback, fallback_model, fallback_api_key)

        # Return current provider settings
        _LOGGER.warning(
            "Using default provider: %s, model: %s",
            self.provider, self.vllm_model
        )
        return (self.provider, self.vllm_model, self.api_key)

    def _normalize_name(self, name: str) -> str:
        """Normalize a name for comparison (lowercase, remove special chars)."""
        import re
        # Lowercase, replace underscores/hyphens with spaces, remove extra spaces
        normalized = name.lower().strip()
        normalized = re.sub(r'[_\-]+', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def _find_camera_entity(self, camera_input: str) -> str | None:
        """Find camera entity ID by alias, name, entity_id, or friendly name."""
        camera_input_norm = self._normalize_name(camera_input)
        camera_input_lower = camera_input.lower().strip()
        
        # PRIORITY 0: Check voice aliases FIRST
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
        
        # Build a list of all cameras with their searchable names
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

    async def _get_camera_snapshot(self, entity_id: str, retries: int = 3, delay: float = 1.0) -> bytes | None:
        """Get camera snapshot using HA's camera component with retry logic.

        For cloud-based cameras (Ring, Nest, etc.), the first snapshot may be stale.
        Retry with delays to allow the camera to process new events.
        """
        last_image = None
        last_error = None

        for attempt in range(retries):
            try:
                if attempt > 0:
                    _LOGGER.debug(
                        "Snapshot retry %d/%d for %s (waiting %.1fs)",
                        attempt + 1, retries, entity_id, delay
                    )
                    await asyncio.sleep(delay)
                    # Increase delay for next attempt (exponential backoff)
                    delay = min(delay * 1.5, 5.0)

                image = await async_get_image(self.hass, entity_id)
                if image and image.content:
                    # Got a valid image
                    if last_image and image.content == last_image:
                        # Same image as before - camera may be returning cached/stale image
                        _LOGGER.debug(
                            "Snapshot from %s unchanged on attempt %d, retrying...",
                            entity_id, attempt + 1
                        )
                        continue

                    _LOGGER.debug(
                        "Got snapshot from %s on attempt %d (%d bytes)",
                        entity_id, attempt + 1, len(image.content)
                    )
                    return image.content

                last_image = image.content if image else None

            except Exception as e:
                last_error = e
                _LOGGER.debug(
                    "Snapshot attempt %d failed for %s: %s",
                    attempt + 1, entity_id, e
                )

        # All retries exhausted
        if last_error:
            _LOGGER.warning(
                "Failed to get fresh snapshot from %s after %d attempts: %s",
                entity_id, retries, last_error
            )
        elif last_image:
            _LOGGER.debug(
                "Returning possibly stale snapshot from %s (unchanged across retries)",
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
        """Get RTSP/stream URL from camera entity."""
        try:
            stream_url = await async_get_stream_source(self.hass, entity_id)
            return stream_url
        except Exception as e:
            _LOGGER.debug("Could not get stream URL for %s: %s", entity_id, e)
            return None

    def _build_ffmpeg_cmd(self, stream_url: str, duration: int, output_path: str) -> list[str]:
        """Build ffmpeg command based on stream type (RTSP vs HLS/HTTP)."""
        # Base command
        cmd = ["ffmpeg", "-y"]

        # Add protocol-specific options
        if stream_url.startswith("rtsp://"):
            # RTSP stream - use TCP transport for reliability
            cmd.extend(["-rtsp_transport", "tcp"])
        # For HLS/HTTP streams, no special transport needed

        # Input
        cmd.extend(["-i", stream_url])

        # Duration and encoding
        cmd.extend([
            "-t", str(duration),
            "-vf", f"scale={self.video_width}:-2",
            "-r", "10",
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
        
        friendly_name = self.hass.states.get(entity_id).attributes.get("friendly_name", entity_id)
        safe_name = entity_id.replace("camera.", "").replace(".", "_")
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=self.snapshot_dir) as vf:
                video_path = vf.name
            
            # Build command based on stream type (RTSP vs HLS/HTTP)
            cmd = self._build_ffmpeg_cmd(stream_url, duration, video_path)
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=duration + 15)
            
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

    async def _record_video_and_frames(self, entity_id: str, duration: int) -> tuple[bytes | None, bytes | None, bytes | None]:
        """Record video and extract frames from camera entity."""
        stream_url = await self._get_stream_url(entity_id)

        video_bytes = None
        frame_bytes = None
        facial_frame_bytes = None

        if not stream_url:
            # No stream URL (cloud camera like Ring/Nest) - use snapshot only
            # Use more retries and longer delays for cloud cameras
            _LOGGER.warning(
                "No stream URL for %s - using snapshot mode (cloud camera). "
                "If images are stale, increase 'Capture Delay' in the blueprint.",
                entity_id
            )
            facial_frame_bytes = await self._get_camera_snapshot(
                entity_id, retries=4, delay=1.5
            )
            frame_bytes = facial_frame_bytes
            return video_bytes, frame_bytes, facial_frame_bytes

        # Has stream URL - get snapshot first for facial recognition
        facial_frame_bytes = await self._get_camera_snapshot(entity_id)
        
        video_path = None
        frame_path = None
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as vf:
                video_path = vf.name
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as ff:
                frame_path = ff.name
            
            # Build commands based on stream type (RTSP vs HLS/HTTP)
            video_cmd = self._build_ffmpeg_cmd(stream_url, duration, video_path)
            frame_cmd = self._build_ffmpeg_frame_cmd(stream_url, frame_path)
            
            video_proc = await asyncio.create_subprocess_exec(
                *video_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            frame_proc = await asyncio.create_subprocess_exec(
                *frame_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            
            await asyncio.wait_for(video_proc.communicate(), timeout=duration + 15)
            await asyncio.wait_for(frame_proc.wait(), timeout=10)
            
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                async with aiofiles.open(video_path, 'rb') as f:
                    video_bytes = await f.read()
            
            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                async with aiofiles.open(frame_path, 'rb') as f:
                    frame_bytes = await f.read()
            
            # Use HA snapshot for facial if we don't have one yet
            if not facial_frame_bytes:
                facial_frame_bytes = frame_bytes
            
            return video_bytes, frame_bytes, facial_frame_bytes
            
        except Exception as e:
            _LOGGER.error("Error recording video from %s: %s", entity_id, e)
            return None, facial_frame_bytes, facial_frame_bytes
        finally:
            for path in [video_path, frame_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

    async def analyze_camera(
        self, camera_input: str, duration: int = None, user_query: str = ""
    ) -> dict[str, Any]:
        """Analyze camera using video and optional facial recognition."""
        duration = duration or self.video_duration

        _LOGGER.warning(
            "Camera analysis requested - Input: '%s', Provider: %s, Model: %s",
            camera_input, self.provider, self.vllm_model
        )

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
        
        # Record video and get frames
        video_bytes, frame_bytes, facial_frame_bytes = await self._record_video_and_frames(entity_id, duration)
        
        # Run facial recognition if enabled
        identified_people = []
        if facial_frame_bytes and self.facial_rec_enabled:
            identified_people = await self._identify_faces(facial_frame_bytes)
        
        # Prepare prompt
        if user_query:
            prompt = user_query
        else:
            prompt = (
                "Describe what you see in this camera feed. "
                "Focus on: people present, their actions, any notable events. "
                "Be concise (2-3 sentences)."
            )
        
        # NOTE: We intentionally do NOT inject facial recognition names into the prompt.
        # Telling the AI "these people are in frame" causes it to hallucinate/confirm
        # their presence even when they're not visible. Instead, facial recognition
        # results are returned separately in the response for the caller to use.

        # Send to AI provider (returns description and effective provider used)
        description, provider_used = await self._analyze_with_provider(video_bytes, frame_bytes, prompt)

        _LOGGER.warning(
            "Analysis complete for %s (%s) - Provider: %s, Response length: %d chars",
            friendly_name, entity_id, provider_used, len(description) if description else 0
        )

        # Save snapshot
        snapshot_path = None
        if frame_bytes:
            os.makedirs(self.snapshot_dir, exist_ok=True)
            snapshot_path = os.path.join(self.snapshot_dir, f"{safe_name}_latest.jpg")
            try:
                async with aiofiles.open(snapshot_path, 'wb') as f:
                    await f.write(frame_bytes)
            except Exception as e:
                _LOGGER.error("Failed to save snapshot: %s", e)

        # Safely check for person-related words in description
        description_text = description or ""
        person_detected = bool(identified_people) or any(
            word in description_text.lower()
            for word in ["person", "people", "someone", "man", "woman", "child"]
        )

        # Include gaming mode debug info
        gaming_mode_active = self._is_gaming_mode_active()

        return {
            "success": True,
            "camera": entity_id,
            "friendly_name": friendly_name,
            "description": description,
            "identified_people": identified_people,
            "person_detected": person_detected,
            "snapshot_path": snapshot_path,
            "snapshot_url": f"/media/local/ha_video_vision/{safe_name}_latest.jpg" if snapshot_path else None,
            "provider_used": provider_used,
            "default_provider": self.provider,
            "gaming_mode_active": gaming_mode_active,
            "gaming_mode_entity": self.gaming_mode_entity,
        }

    async def _analyze_with_provider(
        self, video_bytes: bytes | None, frame_bytes: bytes | None, prompt: str
    ) -> tuple[str, str]:
        """Send video/image to the configured AI provider.

        Returns: (description, provider_used)
        """
        # Get effective provider (may switch if gaming mode is active)
        effective_provider, effective_model, effective_api_key = self._get_effective_provider()

        media_type = "video" if video_bytes else ("image" if frame_bytes else "none")
        _LOGGER.warning(
            "Sending %s to AI - Provider: %s, Model: %s, Base URL: %s",
            media_type, effective_provider, effective_model,
            self.base_url if effective_provider == PROVIDER_LOCAL else "default"
        )

        # Add anti-hallucination instructions for cloud providers
        # Small cloud models (especially free ones) tend to guess/hallucinate identities
        if effective_provider in (PROVIDER_GOOGLE, PROVIDER_OPENROUTER):
            anti_hallucination = (
                "\n\nIMPORTANT: Do NOT identify or name specific people. "
                "Do NOT guess anyone's identity. Only describe physical characteristics "
                "(e.g., 'a person in a red shirt') without naming individuals. "
                "Never say specific names like 'John' or 'the homeowner'."
            )
            prompt = prompt + anti_hallucination

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
        """Analyze using Google Gemini."""
        if not video_bytes and not frame_bytes:
            return "No video or image available for analysis"

        # Use provided overrides or fall back to config
        model = model or self.vllm_model
        api_key = api_key or self.api_key

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            
            parts = [{"text": prompt}]
            
            if video_bytes:
                video_b64 = base64.b64encode(video_bytes).decode()
                parts.insert(0, {
                    "inline_data": {
                        "mime_type": "video/mp4",
                        "data": video_b64
                    }
                })
            elif frame_bytes:
                image_b64 = base64.b64encode(frame_bytes).decode()
                parts.insert(0, {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_b64
                    }
                })
            
            payload = {
                "contents": [{"parts": parts}],
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
        """Analyze using OpenRouter with video support."""
        if not video_bytes and not frame_bytes:
            return "No video or image available for analysis"

        # Use provided overrides or fall back to config
        model = model or self.vllm_model
        api_key = api_key or self.api_key

        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            content = []

            if video_bytes:
                video_b64 = base64.b64encode(video_bytes).decode()
                content.append({
                    "type": "video_url",
                    "video_url": {
                        "url": f"data:video/mp4;base64,{video_b64}"
                    }
                })
            elif frame_bytes:
                image_b64 = base64.b64encode(frame_bytes).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                })

            content.append({"type": "text", "text": prompt})

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": self.vllm_max_tokens,
                "temperature": self.vllm_temperature,
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
        """Analyze using local vLLM endpoint."""
        if not video_bytes and not frame_bytes:
            return "No video or image available for analysis"
        
        try:
            url = f"{self.base_url}/chat/completions"
            
            content = []
            
            if video_bytes:
                video_b64 = base64.b64encode(video_bytes).decode()
                content.append({
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}
                })
            elif frame_bytes:
                image_b64 = base64.b64encode(frame_bytes).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                })
            
            content.append({"type": "text", "text": prompt})
            
            payload = {
                "model": self.vllm_model,
                "messages": [{"role": "user", "content": content}],
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

    async def _identify_faces(self, image_bytes: bytes) -> list[dict]:
        """Send image to facial recognition server."""
        if not self.facial_rec_enabled or not self.facial_rec_url:
            return []
        
        try:
            image_b64 = base64.b64encode(image_bytes).decode()
            
            async with asyncio.timeout(15):
                async with self._session.post(
                    f"{self.facial_rec_url}/identify",
                    json={"image_base64": image_b64},
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        people = result.get("people", [])
                        
                        return [
                            {"name": p["name"], "confidence": p["confidence"]}
                            for p in people
                            if p.get("name") != "Unknown" and p.get("confidence", 0) >= self.facial_rec_confidence
                        ]
                    return []
                    
        except Exception as e:
            _LOGGER.warning("Facial recognition error: %s", e)
            return []

    async def identify_faces_from_file(self, image_path: str) -> dict[str, Any]:
        """Identify faces from an image file."""
        if not os.path.exists(image_path):
            return {"success": False, "error": f"Image not found: {image_path}"}

        try:
            async with aiofiles.open(image_path, 'rb') as f:
                image_bytes = await f.read()

            people = await self._identify_faces(image_bytes)

            return {
                "success": True,
                "faces_detected": len(people),
                "people": people,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
