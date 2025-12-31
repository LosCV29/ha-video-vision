# HA Video Vision Add-ons

Home Assistant add-ons for the [HA Video Vision](https://github.com/LosCV29/ha-video-vision) integration.

## Add-ons in this repository

### Facial Recognition

[![Open your Home Assistant instance and show the add add-on repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2FLosCV29%2Fha-addons)

DeepFace-based facial recognition server that runs locally on your Home Assistant machine.

**Features:**
- One-click installation
- Runs on localhost:8100 (no network config needed)
- Works with HA Video Vision integration
- All processing done locally (privacy-first)
- Supports multiple face detection backends

## Installation

1. Click the button above, or manually add this repository URL to your Home Assistant Add-on Store:
   ```
   https://github.com/LosCV29/ha-addons
   ```

2. Find "Facial Recognition" in the add-on store and click Install

3. Add photos to `/share/faces/PersonName/` folders

4. Start the add-on

5. Configure HA Video Vision to use `http://localhost:8100`

## Requirements

- Home Assistant OS or Supervised installation
- At least 1GB free RAM
- amd64, aarch64, or armv7 architecture

## Support

- [HA Video Vision Documentation](https://github.com/LosCV29/ha-video-vision)
- [Issues](https://github.com/LosCV29/ha-addons/issues)
