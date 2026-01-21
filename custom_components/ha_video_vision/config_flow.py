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
    ALL_PROVIDERS,
    PROVIDER_NAMES,
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
    # Cameras
    CONF_SELECTED_CAMERAS,
    DEFAULT_SELECTED_CAMERAS,
    CONF_CAMERA_ALIASES,
    DEFAULT_CAMERA_ALIASES,
    CONF_CAMERA_CONTEXTS,
    DEFAULT_CAMERA_CONTEXTS,
    # Video Settings
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
    # Facial Recognition (LLM-based)
    CONF_FACIAL_RECOGNITION_ENABLED,
    CONF_FACIAL_RECOGNITION_DIRECTORY,
    CONF_FACIAL_RECOGNITION_RESOLUTION,
    DEFAULT_FACIAL_RECOGNITION_ENABLED,
    DEFAULT_FACIAL_RECOGNITION_DIRECTORY,
    DEFAULT_FACIAL_RECOGNITION_RESOLUTION,
    # Timeline
    CONF_TIMELINE_ENABLED,
    CONF_TIMELINE_RETENTION_DAYS,
    DEFAULT_TIMELINE_ENABLED,
    DEFAULT_TIMELINE_RETENTION_DAYS,
)

_LOGGER = logging.getLogger(__name__)


# =============================================================================
# SHARED HELPER FUNCTIONS (used by both ConfigFlow and OptionsFlow)
# =============================================================================

async def fetch_provider_models(provider: str, config: dict[str, Any]) -> list[dict]:
    """Fetch available models from provider API.

    Args:
        provider: Provider identifier (local, google, openrouter)
        config: Configuration dict containing credentials

    Returns:
        List of model dicts with 'id' and 'name' keys
    """
    try:
        async with aiohttp.ClientSession() as session:
            headers = {}
            if provider == PROVIDER_LOCAL:
                url = f"{config.get(CONF_VLLM_URL, DEFAULT_VLLM_URL)}/models"
            elif provider == PROVIDER_GOOGLE:
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={config[CONF_API_KEY]}"
            elif provider == PROVIDER_OPENROUTER:
                url = "https://openrouter.ai/api/v1/models"
                headers = {"Authorization": f"Bearer {config[CONF_API_KEY]}"}
            else:
                return []

            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status != 200:
                    return []
                data = await response.json()
                return parse_provider_models(provider, data)
    except Exception as e:
        _LOGGER.warning("Failed to fetch models: %s", e)
        return []


def parse_provider_models(provider: str, data: dict) -> list[dict]:
    """Parse models response based on provider - VIDEO CAPABLE ONLY.

    STRICT filtering: Only models confirmed to support base64 video input.
    OpenRouter only supports video via Google Vertex, so only Gemini models work.
    """
    models = []
    if provider == PROVIDER_GOOGLE:
        for model in data.get("models", []):
            name = model.get("name", "")
            # Gemini models support video - filter for gemini-2.0, gemini-1.5, gemini-3
            if any(x in name.lower() for x in ["gemini-2.0", "gemini-1.5", "gemini-exp", "gemini-3"]):
                model_id = name.replace("models/", "")
                display_name = model.get("displayName", model_id)
                models.append({"id": model_id, "name": display_name})
    elif provider == PROVIDER_OPENROUTER:
        for model in data.get("data", []):
            model_id = model.get("id", "")
            # STRICT: Only Google Gemini models support base64 video via Vertex
            is_gemini = model_id.startswith("google/gemini-")
            is_free = ":free" in model_id.lower()
            if is_gemini and not is_free:
                name = model.get("name", model_id)
                models.append({"id": model_id, "name": name})
    elif provider == PROVIDER_LOCAL:
        # Local models - show all, user knows their setup
        for model in data.get("data", []):
            model_id = model.get("id", "")
            models.append({"id": model_id, "name": model_id})
    return models


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
        # Check if already configured - single instance only
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

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
                vol.Required(CONF_PROVIDER, default=DEFAULT_PROVIDER): selector.SelectSelector(
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
        provider = self._data.get(CONF_PROVIDER, DEFAULT_PROVIDER)

        if user_input is not None:
            # Test connection and fetch models
            models = await fetch_provider_models(provider, user_input)
            if models:
                self._data.update(user_input)
                self._data["_available_models"] = models
                return await self.async_step_select_model()
            else:
                errors["base"] = "cannot_connect"

        # Build schema based on provider
        if provider == PROVIDER_LOCAL:
            schema = vol.Schema({
                vol.Required(CONF_VLLM_URL, default=DEFAULT_VLLM_URL): str,
            })
        else:
            schema = vol.Schema({
                vol.Required(CONF_API_KEY): str,
            })

        return self.async_show_form(
            step_id="credentials",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "provider_name": PROVIDER_NAMES[provider],
            },
        )

    async def async_step_select_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Select model from available models."""
        provider = self._data.get(CONF_PROVIDER, DEFAULT_PROVIDER)
        models = self._data.get("_available_models", [])

        if user_input is not None:
            self._data[CONF_VLLM_MODEL] = user_input[CONF_VLLM_MODEL]
            # Clean up temp data
            self._data.pop("_available_models", None)
            return await self.async_step_cameras()

        # Build model options
        model_options = [
            selector.SelectOptionDict(value=m["id"], label=m["name"])
            for m in models[:50]  # Limit to 50 models
        ]

        default_model = PROVIDER_DEFAULT_MODELS.get(provider, models[0]["id"] if models else "")

        return self.async_show_form(
            step_id="select_model",
            data_schema=vol.Schema({
                vol.Required(CONF_VLLM_MODEL, default=default_model): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=model_options,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
            }),
            description_placeholders={
                "provider_name": PROVIDER_NAMES[provider],
                "model_count": str(len(models)),
            },
        )

    async def async_step_cameras(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle camera selection step - Auto-discovered!"""
        if user_input is not None:
            self._data[CONF_SELECTED_CAMERAS] = user_input.get(CONF_SELECTED_CAMERAS, [])

            # Build provider_configs structure
            provider = self._data.get(CONF_PROVIDER, DEFAULT_PROVIDER)
            provider_config = {
                "api_key": self._data.get(CONF_API_KEY, ""),
                "model": self._data.get(CONF_VLLM_MODEL, PROVIDER_DEFAULT_MODELS.get(provider, "")),
            }
            if provider == PROVIDER_LOCAL:
                provider_config["base_url"] = self._data.get(CONF_VLLM_URL, DEFAULT_VLLM_URL)

            self._data[CONF_PROVIDER_CONFIGS] = {provider: provider_config}
            self._data[CONF_DEFAULT_PROVIDER] = provider

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
        self._provider_models: dict[str, list] = {}
        self._temp_api_key: str = ""
        self._temp_base_url: str = ""
        self._selected_context_camera: str = ""

    def _get_provider_config(self, provider: str) -> dict[str, Any]:
        """Get configuration for a specific provider."""
        current = {**self._entry.data, **self._entry.options}
        provider_configs = current.get(CONF_PROVIDER_CONFIGS, {})
        return provider_configs.get(provider, {})

    def _save_provider_config(
        self, provider: str, model: str, api_key: str = "", base_url: str = ""
    ) -> FlowResult:
        """Save provider configuration and return entry."""
        current = {**self._entry.data, **self._entry.options}
        provider_configs = current.get(CONF_PROVIDER_CONFIGS, {})
        new_configs = {**provider_configs}

        config = {"model": model}
        if api_key:
            config["api_key"] = api_key
        if base_url:
            config["base_url"] = base_url
        if provider == PROVIDER_LOCAL:
            config["api_key"] = ""  # Local doesn't need API key

        new_configs[provider] = config
        new_options = {**self._entry.options, CONF_PROVIDER_CONFIGS: new_configs}
        return self.async_create_entry(title="", data=new_options)

    def _build_model_selection_form(
        self, step_id: str, provider: str, models: list[dict]
    ) -> FlowResult:
        """Build model selection form for a provider."""
        provider_config = self._get_provider_config(provider)
        model_options = [
            selector.SelectOptionDict(value=m["id"], label=m["name"])
            for m in models[:50]
        ]
        current_model = provider_config.get("model", PROVIDER_DEFAULT_MODELS.get(provider, ""))

        return self.async_show_form(
            step_id=step_id,
            data_schema=vol.Schema({
                vol.Required(CONF_VLLM_MODEL, default=current_model): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=model_options,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
            }),
        )

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        return self.async_show_menu(
            step_id="init",
            menu_options={
                "default_provider": "Select Default Provider",
                "configure_google": "Configure Google Gemini",
                "configure_openrouter": "Configure OpenRouter",
                "configure_local": "Configure Local vLLM",
                "cameras": "Select Cameras",
                "camera_context": "Camera Context",
                "voice_aliases": "Voice Aliases",
                "video_quality": "Video Settings",
                "ai_settings": "AI Settings",
                "facial_recognition": "Facial Recognition",
                "timeline": "Timeline",
            },
        )

    async def async_step_default_provider(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Select the default AI provider to use."""
        current = {**self._entry.data, **self._entry.options}
        provider_configs = current.get(CONF_PROVIDER_CONFIGS, {})
        current_default = current.get(CONF_DEFAULT_PROVIDER, current.get(CONF_PROVIDER, DEFAULT_PROVIDER))

        if user_input is not None:
            new_default = user_input.get(CONF_DEFAULT_PROVIDER, current_default)
            new_options = {**self._entry.options}
            new_options[CONF_DEFAULT_PROVIDER] = new_default
            new_options[CONF_PROVIDER] = new_default
            return self.async_create_entry(title="", data=new_options)

        # Only show configured providers
        configured_providers = []
        for provider_key in provider_configs:
            if provider_key in PROVIDER_NAMES:
                config = provider_configs[provider_key]
                # Check if provider has credentials
                has_creds = config.get("api_key") or config.get("base_url")
                if has_creds:
                    model = config.get("model", "default")
                    configured_providers.append(
                        selector.SelectOptionDict(
                            value=provider_key,
                            label=f"{PROVIDER_NAMES[provider_key]} ({model})"
                        )
                    )

        if not configured_providers:
            # No providers configured yet
            return self.async_abort(reason="no_providers_configured")

        return self.async_show_form(
            step_id="default_provider",
            data_schema=vol.Schema({
                vol.Required(CONF_DEFAULT_PROVIDER, default=current_default): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=configured_providers,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
            }),
        )

    # ==================== GOOGLE GEMINI ====================
    async def async_step_configure_google(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure Google Gemini provider."""
        errors = {}
        current = {**self._entry.data, **self._entry.options}
        provider_configs = current.get(CONF_PROVIDER_CONFIGS, {})
        google_config = provider_configs.get(PROVIDER_GOOGLE, {})

        if user_input is not None:
            api_key = user_input.get(CONF_API_KEY, "")
            if api_key:
                # Fetch models
                models = await fetch_provider_models(PROVIDER_GOOGLE, {CONF_API_KEY: api_key})
                if models:
                    self._provider_models[PROVIDER_GOOGLE] = models
                    self._temp_api_key = api_key
                    return await self.async_step_google_model()
                else:
                    errors["base"] = "cannot_connect"
            else:
                # Remove provider config if empty
                new_configs = {**provider_configs}
                new_configs.pop(PROVIDER_GOOGLE, None)
                new_options = {**self._entry.options, CONF_PROVIDER_CONFIGS: new_configs}
                return self.async_create_entry(title="", data=new_options)

        return self.async_show_form(
            step_id="configure_google",
            data_schema=vol.Schema({
                vol.Optional(CONF_API_KEY, default=google_config.get("api_key", "")): str,
            }),
            errors=errors,
        )

    async def async_step_google_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Select Google model."""
        if user_input is not None:
            google_config = self._get_provider_config(PROVIDER_GOOGLE)
            return self._save_provider_config(
                PROVIDER_GOOGLE,
                user_input.get(CONF_VLLM_MODEL, PROVIDER_DEFAULT_MODELS[PROVIDER_GOOGLE]),
                api_key=self._temp_api_key or google_config.get("api_key", ""),
            )

        models = self._provider_models.get(PROVIDER_GOOGLE, [])
        return self._build_model_selection_form("google_model", PROVIDER_GOOGLE, models)

    # ==================== OPENROUTER ====================
    async def async_step_configure_openrouter(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure OpenRouter provider."""
        errors = {}
        current = {**self._entry.data, **self._entry.options}
        provider_configs = current.get(CONF_PROVIDER_CONFIGS, {})
        or_config = provider_configs.get(PROVIDER_OPENROUTER, {})

        if user_input is not None:
            api_key = user_input.get(CONF_API_KEY, "")
            if api_key:
                models = await fetch_provider_models(PROVIDER_OPENROUTER, {CONF_API_KEY: api_key})
                if models:
                    self._provider_models[PROVIDER_OPENROUTER] = models
                    self._temp_api_key = api_key
                    return await self.async_step_openrouter_model()
                else:
                    errors["base"] = "cannot_connect"
            else:
                new_configs = {**provider_configs}
                new_configs.pop(PROVIDER_OPENROUTER, None)
                new_options = {**self._entry.options, CONF_PROVIDER_CONFIGS: new_configs}
                return self.async_create_entry(title="", data=new_options)

        return self.async_show_form(
            step_id="configure_openrouter",
            data_schema=vol.Schema({
                vol.Optional(CONF_API_KEY, default=or_config.get("api_key", "")): str,
            }),
            errors=errors,
        )

    async def async_step_openrouter_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Select OpenRouter model."""
        if user_input is not None:
            or_config = self._get_provider_config(PROVIDER_OPENROUTER)
            return self._save_provider_config(
                PROVIDER_OPENROUTER,
                user_input.get(CONF_VLLM_MODEL, PROVIDER_DEFAULT_MODELS[PROVIDER_OPENROUTER]),
                api_key=self._temp_api_key or or_config.get("api_key", ""),
            )

        models = self._provider_models.get(PROVIDER_OPENROUTER, [])
        return self._build_model_selection_form("openrouter_model", PROVIDER_OPENROUTER, models)

    # ==================== LOCAL VLLM ====================
    async def async_step_configure_local(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure Local vLLM provider."""
        errors = {}
        current = {**self._entry.data, **self._entry.options}
        provider_configs = current.get(CONF_PROVIDER_CONFIGS, {})
        local_config = provider_configs.get(PROVIDER_LOCAL, {})

        if user_input is not None:
            base_url = user_input.get(CONF_VLLM_URL, "")
            if base_url:
                models = await fetch_provider_models(PROVIDER_LOCAL, {CONF_VLLM_URL: base_url})
                if models:
                    self._provider_models[PROVIDER_LOCAL] = models
                    self._temp_base_url = base_url
                    return await self.async_step_local_model()
                else:
                    errors["base"] = "cannot_connect"
            else:
                new_configs = {**provider_configs}
                new_configs.pop(PROVIDER_LOCAL, None)
                new_options = {**self._entry.options, CONF_PROVIDER_CONFIGS: new_configs}
                return self.async_create_entry(title="", data=new_options)

        return self.async_show_form(
            step_id="configure_local",
            data_schema=vol.Schema({
                vol.Optional(CONF_VLLM_URL, default=local_config.get("base_url", DEFAULT_VLLM_URL)): str,
            }),
            errors=errors,
        )

    async def async_step_local_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Select Local model."""
        if user_input is not None:
            local_config = self._get_provider_config(PROVIDER_LOCAL)
            return self._save_provider_config(
                PROVIDER_LOCAL,
                user_input.get(CONF_VLLM_MODEL, PROVIDER_DEFAULT_MODELS[PROVIDER_LOCAL]),
                base_url=self._temp_base_url or local_config.get("base_url", DEFAULT_VLLM_URL),
            )

        models = self._provider_models.get(PROVIDER_LOCAL, [])
        return self._build_model_selection_form("local_model", PROVIDER_LOCAL, models)

    # ==================== OTHER OPTIONS ====================
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

    async def async_step_voice_aliases(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle voice alias configuration."""
        if user_input is not None:
            aliases = {}
            alias_text = user_input.get("alias_config", "")

            for line in alias_text.strip().split("\n"):
                line = line.strip()
                if not line or ":" not in line:
                    continue
                parts = line.split(":", 1)
                if len(parts) == 2:
                    voice_name = parts[0].strip().lower()
                    camera_id = parts[1].strip()
                    if voice_name and camera_id:
                        aliases[voice_name] = camera_id

            new_options = {**self._entry.options, CONF_CAMERA_ALIASES: aliases}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}
        aliases = current.get(CONF_CAMERA_ALIASES, DEFAULT_CAMERA_ALIASES)
        selected_cameras = current.get(CONF_SELECTED_CAMERAS, [])

        alias_lines = [f"{name}:{camera}" for name, camera in aliases.items()]
        alias_text = "\n".join(alias_lines) if alias_lines else ""

        camera_hints = []
        for entity_id in selected_cameras:
            state = self.hass.states.get(entity_id)
            if state:
                friendly = state.attributes.get("friendly_name", entity_id)
                camera_hints.append(f"{entity_id} ({friendly})")

        return self.async_show_form(
            step_id="voice_aliases",
            data_schema=vol.Schema({
                vol.Optional("alias_config", default=alias_text): selector.TextSelector(
                    selector.TextSelectorConfig(multiline=True)
                ),
            }),
            description_placeholders={
                "available_cameras": "\n".join(camera_hints) if camera_hints else "No cameras selected",
            },
        )

    async def async_step_camera_context(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle camera context selection - dropdown to pick a camera."""
        current = {**self._entry.data, **self._entry.options}
        selected_cameras = current.get(CONF_SELECTED_CAMERAS, [])
        camera_contexts = current.get(CONF_CAMERA_CONTEXTS, DEFAULT_CAMERA_CONTEXTS)

        if not selected_cameras:
            return self.async_abort(reason="no_cameras_selected")

        if user_input is not None:
            selected_camera = user_input.get("selected_camera", "")
            if selected_camera:
                self._selected_context_camera = selected_camera
                return await self.async_step_camera_context_edit()

        # Build camera options with context indicators
        camera_options = []
        cameras_with_context = 0
        for entity_id in selected_cameras:
            state = self.hass.states.get(entity_id)
            friendly_name = state.attributes.get("friendly_name", entity_id) if state else entity_id
            has_context = entity_id in camera_contexts and camera_contexts[entity_id].strip()
            if has_context:
                cameras_with_context += 1
                label = f"{friendly_name} ({entity_id}) [Has Context]"
            else:
                label = f"{friendly_name} ({entity_id})"
            camera_options.append(
                selector.SelectOptionDict(value=entity_id, label=label)
            )

        return self.async_show_form(
            step_id="camera_context",
            data_schema=vol.Schema({
                vol.Required("selected_camera"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=camera_options,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
            }),
            description_placeholders={
                "camera_count": str(len(selected_cameras)),
                "context_count": str(cameras_with_context),
            },
        )

    async def async_step_camera_context_edit(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Edit context for a specific camera."""
        current = {**self._entry.data, **self._entry.options}
        camera_contexts = current.get(CONF_CAMERA_CONTEXTS, DEFAULT_CAMERA_CONTEXTS)
        entity_id = self._selected_context_camera

        # Get friendly name for display
        state = self.hass.states.get(entity_id)
        friendly_name = state.attributes.get("friendly_name", entity_id) if state else entity_id

        if user_input is not None:
            # Handle clear context checkbox
            clear_context = user_input.get("clear_context", False)
            context_text = user_input.get("context_text", "").strip()

            # Update contexts dictionary
            new_contexts = {**camera_contexts}
            if clear_context or not context_text:
                # Remove context for this camera
                new_contexts.pop(entity_id, None)
            else:
                # Save new context
                new_contexts[entity_id] = context_text

            new_options = {**self._entry.options, CONF_CAMERA_CONTEXTS: new_contexts}
            return self.async_create_entry(title="", data=new_options)

        # Get existing context for this camera
        existing_context = camera_contexts.get(entity_id, "")

        return self.async_show_form(
            step_id="camera_context_edit",
            data_schema=vol.Schema({
                vol.Optional("context_text", default=existing_context): selector.TextSelector(
                    selector.TextSelectorConfig(multiline=True)
                ),
                vol.Optional("clear_context", default=False): selector.BooleanSelector(),
            }),
            description_placeholders={
                "camera_name": friendly_name,
                "entity_id": entity_id,
            },
        )

    async def async_step_video_quality(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle video/image quality settings."""
        if user_input is not None:
            if CONF_VIDEO_WIDTH in user_input:
                user_input[CONF_VIDEO_WIDTH] = int(user_input[CONF_VIDEO_WIDTH])
            if CONF_VIDEO_FPS_PERCENT in user_input:
                user_input[CONF_VIDEO_FPS_PERCENT] = int(user_input[CONF_VIDEO_FPS_PERCENT])
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="video_quality",
            data_schema=vol.Schema({
                vol.Required(CONF_VIDEO_DURATION, default=current.get(CONF_VIDEO_DURATION, DEFAULT_VIDEO_DURATION)): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=1, max=10, step=1, unit_of_measurement="seconds", mode=selector.NumberSelectorMode.SLIDER)
                ),
                vol.Required(CONF_VIDEO_WIDTH, default=str(current.get(CONF_VIDEO_WIDTH) or DEFAULT_VIDEO_WIDTH)): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"label": "480 (fast, low detail)", "value": "480"},
                            {"label": "640 (SD)", "value": "640"},
                            {"label": "720 (HD)", "value": "720"},
                            {"label": "1080 (Full HD)", "value": "1080"},
                            {"label": "1280 (recommended)", "value": "1280"},
                            {"label": "1440 (2K)", "value": "1440"},
                            {"label": "1920 (max detail)", "value": "1920"},
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(CONF_VIDEO_FPS_PERCENT, default=str(current.get(CONF_VIDEO_FPS_PERCENT) or DEFAULT_VIDEO_FPS_PERCENT)): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"label": "25% (smaller files)", "value": "25"},
                            {"label": "50%", "value": "50"},
                            {"label": "75%", "value": "75"},
                            {"label": "100% (native)", "value": "100"},
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(CONF_SNAPSHOT_QUALITY, default=current.get(CONF_SNAPSHOT_QUALITY, DEFAULT_SNAPSHOT_QUALITY)): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=50, max=100, step=5, unit_of_measurement="%", mode=selector.NumberSelectorMode.SLIDER)
                ),
                vol.Required(CONF_NOTIFICATION_FRAME_POSITION, default=current.get(CONF_NOTIFICATION_FRAME_POSITION, DEFAULT_NOTIFICATION_FRAME_POSITION)): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0, max=100, step=10, unit_of_measurement="%", mode=selector.NumberSelectorMode.SLIDER)
                ),
                vol.Optional(CONF_SNAPSHOT_DIR, default=current.get(CONF_SNAPSHOT_DIR, DEFAULT_SNAPSHOT_DIR)): str,
            }),
        )

    async def async_step_ai_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle AI response settings."""
        if user_input is not None:
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="ai_settings",
            data_schema=vol.Schema({
                vol.Required(CONF_VLLM_MAX_TOKENS, default=current.get(CONF_VLLM_MAX_TOKENS, DEFAULT_VLLM_MAX_TOKENS)): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=50, max=500, step=10, mode=selector.NumberSelectorMode.SLIDER)
                ),
                vol.Required(CONF_VLLM_TEMPERATURE, default=current.get(CONF_VLLM_TEMPERATURE, DEFAULT_VLLM_TEMPERATURE)): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.1, mode=selector.NumberSelectorMode.SLIDER)
                ),
            }),
            description_placeholders={
                "temp_hint": "Lower = more consistent/factual. Higher = more creative.",
            },
        )

    async def async_step_facial_recognition(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle facial recognition settings (LLM-based with reference photos)."""
        if user_input is not None:
            # Convert resolution to int
            if CONF_FACIAL_RECOGNITION_RESOLUTION in user_input:
                user_input[CONF_FACIAL_RECOGNITION_RESOLUTION] = int(user_input[CONF_FACIAL_RECOGNITION_RESOLUTION])
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="facial_recognition",
            data_schema=vol.Schema({
                vol.Required(
                    CONF_FACIAL_RECOGNITION_ENABLED,
                    default=current.get(CONF_FACIAL_RECOGNITION_ENABLED, DEFAULT_FACIAL_RECOGNITION_ENABLED)
                ): selector.BooleanSelector(),
                vol.Required(
                    CONF_FACIAL_RECOGNITION_DIRECTORY,
                    default=current.get(CONF_FACIAL_RECOGNITION_DIRECTORY, DEFAULT_FACIAL_RECOGNITION_DIRECTORY)
                ): selector.TextSelector(
                    selector.TextSelectorConfig(type=selector.TextSelectorType.TEXT)
                ),
                vol.Required(
                    CONF_FACIAL_RECOGNITION_RESOLUTION,
                    default=str(current.get(CONF_FACIAL_RECOGNITION_RESOLUTION, DEFAULT_FACIAL_RECOGNITION_RESOLUTION))
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"label": "Original (no resize - sharpest)", "value": "0"},
                            {"label": "1024px (high quality)", "value": "1024"},
                            {"label": "768px (recommended)", "value": "768"},
                            {"label": "512px (faster, more people)", "value": "512"},
                            {"label": "384px (local vLLM optimized)", "value": "384"},
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
            }),
            description_placeholders={},
        )

    async def async_step_timeline(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle timeline settings."""
        if user_input is not None:
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="timeline",
            data_schema=vol.Schema({
                vol.Required(
                    CONF_TIMELINE_ENABLED,
                    default=current.get(CONF_TIMELINE_ENABLED, DEFAULT_TIMELINE_ENABLED)
                ): selector.BooleanSelector(),
                vol.Required(
                    CONF_TIMELINE_RETENTION_DAYS,
                    default=current.get(CONF_TIMELINE_RETENTION_DAYS, DEFAULT_TIMELINE_RETENTION_DAYS)
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1, max=365, step=1,
                        unit_of_measurement="days",
                        mode=selector.NumberSelectorMode.BOX
                    )
                ),
            }),
        )
