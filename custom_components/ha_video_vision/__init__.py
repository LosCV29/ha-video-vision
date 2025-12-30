"""HA Video Vision - AI Camera Analysis with Auto-Discovery."""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile
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
    PROVIDER_LOCAL,
    PROVIDER_GOOGLE,
    PROVIDER_OPENROUTER,
    PROVIDER_BASE_URLS,
    PROVIDER_DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    # vLLM
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
    # Cameras - NEW Auto-Discovery
    CONF_SELECTED_CAMERAS,
    DEFAULT_SELECTED_CAMERAS,
    # Video
    CONF_VIDEO_DURATION,
    CONF_VIDEO_WIDTH,
    CONF_VIDEO_CRF,
    CONF_FRAME_FOR_FACIAL,
    DEFAULT_VIDEO_DURATION,
    DEFAULT_VIDEO_WIDTH,
    DEFAULT_VIDEO_CRF,
    DEFAULT_FRAME_FOR_FACIAL,
    # Snapshot
    CONF_SNAPSHOT_DIR,
    DEFAULT_SNAPSHOT_DIR,
    # Notifications
    CONF_NOTIFY_SERVICES,
    CONF_IOS_DEVICES,
    CONF_COOLDOWN_SECONDS,
    CONF_CRITICAL_ALERTS,
    DEFAULT_NOTIFY_SERVICES,
    DEFAULT_IOS_DEVICES,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_CRITICAL_ALERTS,
    # Services
    SERVICE_ANALYZE_CAMERA,
    SERVICE_RECORD_CLIP,
    SERVICE_IDENTIFY_FACES,
    # Attributes
    ATTR_CAMERA,
    ATTR_DURATION,
    ATTR_USER_QUERY,
    ATTR_NOTIFY,
    ATTR_IMAGE_PATH,
)

_LOGGER = logging.getLogger(__name__)

# Service schemas
SERVICE_ANALYZE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CAMERA): cv.string,
        vol.Optional(ATTR_DURATION, default=3): vol.All(vol.Coerce(int), vol.Range(min=1, max=10)),
        vol.Optional(ATTR_USER_QUERY, default=""): cv.string,
        vol.Optional(ATTR_NOTIFY, default=False): cv.boolean,
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
        notify = call.data.get(ATTR_NOTIFY, False)
        
        result = await analyzer.analyze_camera(camera, duration, user_query)
        
        if notify and result.get("success"):
            await analyzer.send_notification(result)
        
        return result

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
        # Provider settings
        self.provider = config.get(CONF_PROVIDER, DEFAULT_PROVIDER)
        self.provider_configs = config.get(CONF_PROVIDER_CONFIGS, {})
        
        active_config = self.provider_configs.get(self.provider, {})
        
        if active_config:
            self.api_key = active_config.get("api_key", "")
            self.vllm_model = active_config.get("model", PROVIDER_DEFAULT_MODELS.get(self.provider, ""))
            self.base_url = active_config.get("base_url", PROVIDER_BASE_URLS.get(self.provider, ""))
        else:
            self.api_key = config.get(CONF_API_KEY, "")
            self.vllm_model = config.get(CONF_VLLM_MODEL, PROVIDER_DEFAULT_MODELS.get(self.provider, DEFAULT_VLLM_MODEL))
            
            if self.provider == PROVIDER_LOCAL:
                self.base_url = config.get(CONF_VLLM_URL, DEFAULT_VLLM_URL)
            else:
                self.base_url = PROVIDER_BASE_URLS.get(self.provider, DEFAULT_VLLM_URL)
        
        self.vllm_max_tokens = config.get(CONF_VLLM_MAX_TOKENS, DEFAULT_VLLM_MAX_TOKENS)
        self.vllm_temperature = config.get(CONF_VLLM_TEMPERATURE, DEFAULT_VLLM_TEMPERATURE)
        
        # Facial recognition
        self.facial_rec_url = config.get(CONF_FACIAL_REC_URL, DEFAULT_FACIAL_REC_URL)
        self.facial_rec_enabled = config.get(CONF_FACIAL_REC_ENABLED, DEFAULT_FACIAL_REC_ENABLED)
        self.facial_rec_confidence = config.get(CONF_FACIAL_REC_CONFIDENCE, DEFAULT_FACIAL_REC_CONFIDENCE)
        
        # Auto-discovered cameras (list of entity_ids)
        self.selected_cameras = config.get(CONF_SELECTED_CAMERAS, DEFAULT_SELECTED_CAMERAS)
        
        # Video settings
        self.video_duration = config.get(CONF_VIDEO_DURATION, DEFAULT_VIDEO_DURATION)
        self.video_width = config.get(CONF_VIDEO_WIDTH, DEFAULT_VIDEO_WIDTH)
        self.video_crf = config.get(CONF_VIDEO_CRF, DEFAULT_VIDEO_CRF)
        self.frame_for_facial = config.get(CONF_FRAME_FOR_FACIAL, DEFAULT_FRAME_FOR_FACIAL)
        
        # Snapshot settings
        self.snapshot_dir = config.get(CONF_SNAPSHOT_DIR, DEFAULT_SNAPSHOT_DIR)
        
        # Notifications
        notify_services = config.get(CONF_NOTIFY_SERVICES, DEFAULT_NOTIFY_SERVICES)
        if isinstance(notify_services, str):
            self.notify_services = [s.strip() for s in notify_services.split(",") if s.strip()]
        else:
            self.notify_services = notify_services or []
        
        ios_devices = config.get(CONF_IOS_DEVICES, DEFAULT_IOS_DEVICES)
        if isinstance(ios_devices, str):
            self.ios_devices = [s.strip() for s in ios_devices.split(",") if s.strip()]
        else:
            self.ios_devices = ios_devices or []
        
        self.cooldown_seconds = config.get(CONF_COOLDOWN_SECONDS, DEFAULT_COOLDOWN_SECONDS)
        self.critical_alerts = config.get(CONF_CRITICAL_ALERTS, DEFAULT_CRITICAL_ALERTS)
        
        _LOGGER.info(
            "HA Video Vision config updated - Provider: %s, Cameras: %d",
            self.provider, len(self.selected_cameras)
        )

    def _find_camera_entity(self, camera_input: str) -> str | None:
        """Find camera entity ID by name, entity_id, or friendly name."""
        camera_input_lower = camera_input.lower().strip()
        
        # Direct match with entity_id
        if camera_input_lower.startswith("camera."):
            if camera_input_lower in [c.lower() for c in self.selected_cameras]:
                return camera_input
            # Check if it exists even if not in selected
            state = self.hass.states.get(camera_input)
            if state:
                return camera_input
        
        # Search through selected cameras
        for entity_id in self.selected_cameras:
            state = self.hass.states.get(entity_id)
            if not state:
                continue
            
            # Match by entity_id (without camera. prefix)
            if entity_id.lower() == f"camera.{camera_input_lower}":
                return entity_id
            
            # Match by entity_id suffix
            if entity_id.lower().endswith(camera_input_lower):
                return entity_id
            
            # Match by friendly name
            friendly_name = state.attributes.get("friendly_name", "").lower()
            if friendly_name == camera_input_lower:
                return entity_id
            
            # Partial match on friendly name
            if camera_input_lower in friendly_name or friendly_name in camera_input_lower:
                return entity_id
        
        # Search ALL cameras (for flexibility)
        for state in self.hass.states.async_all("camera"):
            entity_id = state.entity_id
            friendly_name = state.attributes.get("friendly_name", "").lower()
            
            if camera_input_lower in entity_id.lower() or camera_input_lower in friendly_name:
                return entity_id
        
        return None

    async def _get_camera_snapshot(self, entity_id: str) -> bytes | None:
        """Get camera snapshot using HA's camera component."""
        try:
            image = await async_get_image(self.hass, entity_id)
            return image.content
        except Exception as e:
            _LOGGER.error("Failed to get snapshot from %s: %s", entity_id, e)
            return None

    async def _get_stream_url(self, entity_id: str) -> str | None:
        """Get RTSP/stream URL from camera entity."""
        try:
            stream_url = await async_get_stream_source(self.hass, entity_id)
            return stream_url
        except Exception as e:
            _LOGGER.debug("Could not get stream URL for %s: %s", entity_id, e)
            return None

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
            
            cmd = [
                "ffmpeg", "-y", "-rtsp_transport", "tcp",
                "-i", stream_url,
                "-t", str(duration),
                "-vf", f"scale={self.video_width}:-2",
                "-r", "10",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", str(self.video_crf),
                "-an",
                video_path
            ]
            
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
        
        # Always try to get a snapshot for facial recognition (high quality)
        facial_frame_bytes = await self._get_camera_snapshot(entity_id)
        
        if not stream_url:
            # No stream URL - use snapshot only
            _LOGGER.info("No stream URL for %s, using snapshot only", entity_id)
            frame_bytes = facial_frame_bytes
            return video_bytes, frame_bytes, facial_frame_bytes
        
        video_path = None
        frame_path = None
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as vf:
                video_path = vf.name
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as ff:
                frame_path = ff.name
            
            # Record video
            video_cmd = [
                "ffmpeg", "-y", "-rtsp_transport", "tcp",
                "-i", stream_url,
                "-t", str(duration),
                "-vf", f"scale={self.video_width}:-2",
                "-r", "10",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", str(self.video_crf),
                "-an",
                video_path
            ]
            
            # Extract frame from stream
            frame_cmd = [
                "ffmpeg", "-y", "-rtsp_transport", "tcp",
                "-i", stream_url,
                "-frames:v", "1",
                "-vf", f"scale={self.video_width}:-2",
                "-q:v", "2",
                frame_path
            ]
            
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
        
        if identified_people:
            names = [p["name"] for p in identified_people]
            prompt += f"\n\nIdentified people in frame: {', '.join(names)}"
        
        # Send to AI provider
        description = await self._analyze_with_provider(video_bytes, frame_bytes, prompt)
        
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
        
        person_detected = bool(identified_people) or any(
            word in description.lower() 
            for word in ["person", "people", "someone", "man", "woman", "child"]
        )
        
        return {
            "success": True,
            "camera": entity_id,
            "friendly_name": friendly_name,
            "description": description,
            "identified_people": identified_people,
            "person_detected": person_detected,
            "snapshot_path": snapshot_path,
            "snapshot_url": f"/media/local/ha_video_vision/{safe_name}_latest.jpg" if snapshot_path else None,
            "provider_used": self.provider,
        }

    async def _analyze_with_provider(
        self, video_bytes: bytes | None, frame_bytes: bytes | None, prompt: str
    ) -> str:
        """Send video/image to the configured AI provider."""
        
        if self.provider == PROVIDER_GOOGLE:
            return await self._analyze_google(video_bytes, frame_bytes, prompt)
        elif self.provider == PROVIDER_OPENROUTER:
            return await self._analyze_openrouter(video_bytes, frame_bytes, prompt)
        elif self.provider == PROVIDER_LOCAL:
            return await self._analyze_local(video_bytes, frame_bytes, prompt)
        else:
            return "Unknown provider configured"

    async def _analyze_google(self, video_bytes: bytes | None, frame_bytes: bytes | None, prompt: str) -> str:
        """Analyze using Google Gemini."""
        if not video_bytes and not frame_bytes:
            return "No video or image available for analysis"
        
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.vllm_model}:generateContent?key={self.api_key}"
            
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
                        return result["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        error = await response.text()
                        _LOGGER.error("Gemini error: %s", error[:500])
                        return f"Analysis failed: {response.status}"
                        
        except Exception as e:
            _LOGGER.error("Gemini analysis error: %s", e)
            return f"Analysis error: {str(e)}"

    async def _analyze_openrouter(self, video_bytes: bytes | None, frame_bytes: bytes | None, prompt: str) -> str:
        """Analyze using OpenRouter with video support."""
        if not video_bytes and not frame_bytes:
            return "No video or image available for analysis"
        
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
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
                "model": self.vllm_model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": self.vllm_max_tokens,
                "temperature": self.vllm_temperature,
            }
            
            async with asyncio.timeout(60):
                async with self._session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
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
                        return result["choices"][0]["message"]["content"]
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

    async def send_notification(self, analysis_result: dict[str, Any]) -> None:
        """Send notification with analysis results."""
        if not self.notify_services:
            return
        
        description = analysis_result.get("description", "Camera checked")
        friendly_name = analysis_result.get("friendly_name", "Camera")
        snapshot_url = analysis_result.get("snapshot_url")
        identified = analysis_result.get("identified_people", [])
        
        title = f"📹 {friendly_name}"
        
        if identified:
            names = [p["name"] for p in identified]
            message = f"{', '.join(names)} detected. {description}"
        else:
            message = description
        
        for service in self.notify_services:
            try:
                service_domain, service_name = service.split(".", 1)
                
                data = {
                    "title": title,
                    "message": message,
                }
                
                # Add image for mobile notifications
                if snapshot_url:
                    if service in self.ios_devices:
                        data["data"] = {
                            "attachment": {"url": snapshot_url},
                            "push": {"sound": "default"},
                        }
                        if self.critical_alerts:
                            data["data"]["push"]["interruption-level"] = "critical"
                    else:
                        data["data"] = {"image": snapshot_url}
                
                await self.hass.services.async_call(
                    service_domain, service_name.replace("mobile_app_", ""), data
                )
                
            except Exception as e:
                _LOGGER.error("Notification error for %s: %s", service, e)
