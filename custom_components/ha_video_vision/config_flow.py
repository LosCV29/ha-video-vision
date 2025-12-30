"""Config flow for HA Video Vision - Auto-Discovery."""
from __future__ import annotations

import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector
from homeassistant.helpers.entity_registry import async_get as async_get_entity_registry
import homeassistant.helpers.config_validation as cv

from .const import (
    DOMAIN,
    # Provider
    CONF_PROVIDER,
    CONF_API_KEY,
    PROVIDER_LOCAL,
    PROVIDER_GOOGLE,
    PROVIDER_OPENROUTER,
    ALL_PROVIDERS,
    PROVIDER_NAMES,
    PROVIDER_BASE_URLS,
    PROVIDER_DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    # vLLM
    CONF_VLLM_URL,
    CONF_VLLM_MODEL,
    DEFAULT_VLLM_URL,
    DEFAULT_VLLM_MODEL,
    # Cameras - NEW
    CONF_SELECTED_CAMERAS,
    DEFAULT_SELECTED_CAMERAS,
    # Facial Recognition
    CONF_FACIAL_REC_URL,
    CONF_FACIAL_REC_ENABLED,
    CONF_FACIAL_REC_CONFIDENCE,
    DEFAULT_FACIAL_REC_URL,
    DEFAULT_FACIAL_REC_ENABLED,
    DEFAULT_FACIAL_REC_CONFIDENCE,
    # Video - FULL SETTINGS
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
)

_LOGGER = logging.getLogger(__name__)


class VideoVisionConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for HA Video Vision."""

    VERSION = 4

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._data: dict[str, Any] = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step - Provider selection."""
        if user_input is not None:
            self._data[CONF_PROVIDER] = user_input[CONF_PROVIDER]
            return await self.async_step_credentials()

        # Build provider options
        provider_options = [
            selector.SelectOptionDict(value=p, label=PROVIDER_NAMES[p])
            for p in ALL_PROVIDERS
        ]

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required(CONF_PROVIDER, default=PROVIDER_OPENROUTER): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=provider_options,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
            }),
        )

    async def async_step_credentials(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle credentials step based on provider."""
        errors = {}
        provider = self._data.get(CONF_PROVIDER, PROVIDER_OPENROUTER)

        if user_input is not None:
            valid = await self._test_provider_connection(provider, user_input)
            if valid:
                self._data.update(user_input)
                return await self.async_step_cameras()
            else:
                errors["base"] = "cannot_connect"

        # Build schema based on provider
        if provider == PROVIDER_LOCAL:
            schema = vol.Schema({
                vol.Required(CONF_VLLM_URL, default=DEFAULT_VLLM_URL): str,
                vol.Required(CONF_VLLM_MODEL, default=PROVIDER_DEFAULT_MODELS[provider]): str,
            })
        else:
            schema = vol.Schema({
                vol.Required(CONF_API_KEY): str,
                vol.Optional(CONF_VLLM_MODEL, default=PROVIDER_DEFAULT_MODELS[provider]): str,
            })

        return self.async_show_form(
            step_id="credentials",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "provider_name": PROVIDER_NAMES[provider],
                "default_model": PROVIDER_DEFAULT_MODELS[provider],
            },
        )

    async def _test_provider_connection(
        self, provider: str, config: dict[str, Any]
    ) -> bool:
        """Test connection to the selected provider."""
        try:
            async with aiohttp.ClientSession() as session:
                if provider == PROVIDER_LOCAL:
                    url = f"{config[CONF_VLLM_URL]}/models"
                    headers = {}
                elif provider == PROVIDER_GOOGLE:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={config[CONF_API_KEY]}"
                    headers = {}
                elif provider == PROVIDER_OPENROUTER:
                    url = "https://openrouter.ai/api/v1/models"
                    headers = {"Authorization": f"Bearer {config[CONF_API_KEY]}"}
                else:
                    return False

                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    return response.status == 200
        except Exception as e:
            _LOGGER.warning("Provider connection test failed: %s", e)
            return False

    async def async_step_cameras(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle camera selection step - Auto-discovered!"""
        if user_input is not None:
            self._data[CONF_SELECTED_CAMERAS] = user_input.get(CONF_SELECTED_CAMERAS, [])
            
            # Create the config entry
            return self.async_create_entry(
                title="HA Video Vision",
                data=self._data,
            )

        # Auto-discover all camera entities
        camera_entities = []
        for state in self.hass.states.async_all("camera"):
            friendly_name = state.attributes.get("friendly_name", state.entity_id)
            camera_entities.append(
                selector.SelectOptionDict(
                    value=state.entity_id,
                    label=f"{friendly_name} ({state.entity_id})"
                )
            )

        if not camera_entities:
            # No cameras found - show message
            return self.async_show_form(
                step_id="cameras",
                data_schema=vol.Schema({
                    vol.Optional(CONF_SELECTED_CAMERAS, default=[]): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[],
                            multiple=True,
                            mode=selector.SelectSelectorMode.LIST,
                        )
                    ),
                }),
                description_placeholders={
                    "camera_count": "0",
                    "camera_hint": "No cameras found. Add camera integrations first.",
                },
            )

        return self.async_show_form(
            step_id="cameras",
            data_schema=vol.Schema({
                vol.Required(CONF_SELECTED_CAMERAS, default=[]): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=camera_entities,
                        multiple=True,
                        mode=selector.SelectSelectorMode.LIST,
                    )
                ),
            }),
            description_placeholders={
                "camera_count": str(len(camera_entities)),
                "camera_hint": "Select which cameras to enable for AI analysis.",
            },
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return VideoVisionOptionsFlow(config_entry)


class VideoVisionOptionsFlow(config_entries.OptionsFlow):
    """Handle options for HA Video Vision."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self._entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        return self.async_show_menu(
            step_id="init",
            menu_options=[
                "provider",
                "cameras",
                "facial_rec",
                "video",
                "notifications",
            ],
        )

    async def async_step_provider(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle provider settings."""
        errors = {}
        current = {**self._entry.data, **self._entry.options}

        if user_input is not None:
            provider = user_input.get(CONF_PROVIDER, current.get(CONF_PROVIDER))
            valid = await self._test_provider_connection(provider, user_input)
            if valid:
                new_options = {**self._entry.options, **user_input}
                return self.async_create_entry(title="", data=new_options)
            else:
                errors["base"] = "cannot_connect"

        provider = current.get(CONF_PROVIDER, DEFAULT_PROVIDER)
        provider_options = [
            selector.SelectOptionDict(value=p, label=PROVIDER_NAMES[p])
            for p in ALL_PROVIDERS
        ]

        if provider == PROVIDER_LOCAL:
            schema = vol.Schema({
                vol.Required(CONF_PROVIDER, default=provider): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=provider_options,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(CONF_VLLM_URL, default=current.get(CONF_VLLM_URL, DEFAULT_VLLM_URL)): str,
                vol.Required(CONF_VLLM_MODEL, default=current.get(CONF_VLLM_MODEL, DEFAULT_VLLM_MODEL)): str,
            })
        else:
            schema = vol.Schema({
                vol.Required(CONF_PROVIDER, default=provider): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=provider_options,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(CONF_API_KEY, default=current.get(CONF_API_KEY, "")): str,
                vol.Optional(CONF_VLLM_MODEL, default=current.get(CONF_VLLM_MODEL, PROVIDER_DEFAULT_MODELS.get(provider, ""))): str,
            })

        return self.async_show_form(
            step_id="provider",
            data_schema=schema,
            errors=errors,
        )

    async def _test_provider_connection(
        self, provider: str, config: dict[str, Any]
    ) -> bool:
        """Test connection to the selected provider."""
        try:
            async with aiohttp.ClientSession() as session:
                if provider == PROVIDER_LOCAL:
                    url = f"{config.get(CONF_VLLM_URL, DEFAULT_VLLM_URL)}/models"
                    headers = {}
                elif provider == PROVIDER_GOOGLE:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={config.get(CONF_API_KEY, '')}"
                    headers = {}
                elif provider == PROVIDER_OPENROUTER:
                    url = "https://openrouter.ai/api/v1/models"
                    headers = {"Authorization": f"Bearer {config.get(CONF_API_KEY, '')}"}
                else:
                    return False

                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    return response.status == 200
        except Exception:
            return False

    async def async_step_cameras(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle camera selection - Auto-discovered!"""
        if user_input is not None:
            new_options = {**self._entry.options}
            new_options[CONF_SELECTED_CAMERAS] = user_input.get(CONF_SELECTED_CAMERAS, [])
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}
        selected = current.get(CONF_SELECTED_CAMERAS, [])

        # Auto-discover all camera entities
        camera_entities = []
        for state in self.hass.states.async_all("camera"):
            friendly_name = state.attributes.get("friendly_name", state.entity_id)
            camera_entities.append(
                selector.SelectOptionDict(
                    value=state.entity_id,
                    label=f"{friendly_name} ({state.entity_id})"
                )
            )

        return self.async_show_form(
            step_id="cameras",
            data_schema=vol.Schema({
                vol.Required(CONF_SELECTED_CAMERAS, default=selected): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=camera_entities,
                        multiple=True,
                        mode=selector.SelectSelectorMode.LIST,
                    )
                ),
            }),
            description_placeholders={
                "camera_count": str(len(camera_entities)),
            },
        )

    async def async_step_facial_rec(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle facial recognition settings."""
        if user_input is not None:
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="facial_rec",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_FACIAL_REC_ENABLED,
                        default=current.get(CONF_FACIAL_REC_ENABLED, DEFAULT_FACIAL_REC_ENABLED),
                    ): bool,
                    vol.Required(
                        CONF_FACIAL_REC_URL,
                        default=current.get(CONF_FACIAL_REC_URL, DEFAULT_FACIAL_REC_URL),
                    ): str,
                    vol.Required(
                        CONF_FACIAL_REC_CONFIDENCE,
                        default=current.get(CONF_FACIAL_REC_CONFIDENCE, DEFAULT_FACIAL_REC_CONFIDENCE),
                    ): vol.All(vol.Coerce(int), vol.Range(min=0, max=100)),
                }
            ),
        )

    async def async_step_video(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle video recording settings."""
        if user_input is not None:
            # Convert string values to integers
            if CONF_VIDEO_WIDTH in user_input:
                user_input[CONF_VIDEO_WIDTH] = int(user_input[CONF_VIDEO_WIDTH])
            if CONF_VIDEO_CRF in user_input:
                user_input[CONF_VIDEO_CRF] = int(user_input[CONF_VIDEO_CRF])
            
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="video",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_VIDEO_DURATION,
                        default=current.get(CONF_VIDEO_DURATION, DEFAULT_VIDEO_DURATION),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=10)),
                    vol.Optional(
                        CONF_VIDEO_WIDTH,
                        default=str(current.get(CONF_VIDEO_WIDTH, DEFAULT_VIDEO_WIDTH)),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                {"label": "320p (Fastest, Smallest)", "value": "320"},
                                {"label": "480p (Fast, Small)", "value": "480"},
                                {"label": "640p (Balanced)", "value": "640"},
                                {"label": "720p (Good Quality)", "value": "720"},
                                {"label": "1080p (Best Quality, Largest)", "value": "1080"},
                            ],
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_VIDEO_CRF,
                        default=str(current.get(CONF_VIDEO_CRF, DEFAULT_VIDEO_CRF)),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                {"label": "18 - Best Quality (Larger File)", "value": "18"},
                                {"label": "23 - High Quality", "value": "23"},
                                {"label": "28 - Balanced (Default)", "value": "28"},
                                {"label": "32 - Smaller File", "value": "32"},
                                {"label": "35 - Smallest File (Lower Quality)", "value": "35"},
                            ],
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_FRAME_FOR_FACIAL,
                        default=current.get(CONF_FRAME_FOR_FACIAL, DEFAULT_FRAME_FOR_FACIAL),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=90)),
                    vol.Optional(
                        CONF_SNAPSHOT_DIR,
                        default=current.get(CONF_SNAPSHOT_DIR, DEFAULT_SNAPSHOT_DIR),
                    ): str,
                }
            ),
        )

    async def async_step_notifications(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle notification settings."""
        if user_input is not None:
            # Parse notify services
            services = [s.strip() for s in user_input.get("notify_services_text", "").split(",") if s.strip()]
            user_input[CONF_NOTIFY_SERVICES] = services
            del user_input["notify_services_text"]
            
            # Parse iOS devices
            ios_devices = [s.strip() for s in user_input.get("ios_devices_text", "").split(",") if s.strip()]
            user_input[CONF_IOS_DEVICES] = ios_devices
            del user_input["ios_devices_text"]
            
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}
        
        notify_services = current.get(CONF_NOTIFY_SERVICES, DEFAULT_NOTIFY_SERVICES)
        notify_text = ", ".join(notify_services) if notify_services else ""
        
        ios_devices = current.get(CONF_IOS_DEVICES, DEFAULT_IOS_DEVICES)
        ios_text = ", ".join(ios_devices) if ios_devices else ""

        return self.async_show_form(
            step_id="notifications",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        "notify_services_text",
                        default=notify_text,
                    ): str,
                    vol.Optional(
                        "ios_devices_text",
                        default=ios_text,
                    ): str,
                    vol.Optional(
                        CONF_COOLDOWN_SECONDS,
                        default=current.get(CONF_COOLDOWN_SECONDS, DEFAULT_COOLDOWN_SECONDS),
                    ): vol.All(vol.Coerce(int), vol.Range(min=0, max=3600)),
                    vol.Optional(
                        CONF_CRITICAL_ALERTS,
                        default=current.get(CONF_CRITICAL_ALERTS, DEFAULT_CRITICAL_ALERTS),
                    ): bool,
                }
            ),
        )
