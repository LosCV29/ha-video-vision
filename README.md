# 📹 HA Video Vision

[![HACS Custom](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![Home Assistant](https://img.shields.io/badge/Home%20Assistant-2024.1+-blue.svg)](https://www.home-assistant.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**True AI video analysis for Home Assistant** - not just snapshots!

> 🎬 **Send actual video clips to AI for scene understanding. Know WHO is at your door, not just "person detected".**

---

## 🎯 What Makes This Different?

| Feature | Ring/Nest/Others | HA Video Vision |
|---------|------------------|-----------------|
| Detection | "Person detected" | "Carlos is at the door with a package" |
| Analysis | Static snapshots | **Real video clips** |
| **Facial Recognition** | ❌ or $$/month | ✅ FREE (DeepFace) |
| **Privacy** | Cloud required | ✅ 100% Local option |
| **Monthly Cost** | $3-10+/month | **$0** |
| Works Offline | ❌ | ✅ |

---

## ✨ Features

- 🎥 **Native Video Analysis** - Sends actual video clips (not frames) to AI
- 👤 **Facial Recognition** - "Who's at the door?" → "It's Carlos"
- 📱 **Smart Notifications** - AI descriptions + snapshots to iOS/Android
- 🆓 **Free by Default** - OpenRouter Nemotron model is 100% free
- 🏠 **Local-First** - Run everything on your own hardware
- 🔌 **Multi-Provider** - Local vLLM, Google Gemini, or OpenRouter

---

## 🚀 Quick Start

### Installation (HACS)

1. Open HACS → Integrations → ⋮ → Custom Repositories
2. Add: `https://github.com/YOUR_USERNAME/ha-video-vision`
3. Install "HA Video Vision"
4. Restart Home Assistant
5. Settings → Devices & Services → Add Integration → "HA Video Vision"

### Installation (Manual)

```bash
cp -r ha_video_vision /config/custom_components/
```

---

## 🔌 Supported Providers (Video-Capable Only)

| Provider | Model | Cost | Notes |
|----------|-------|------|-------|
| **OpenRouter** | Nemotron 12B VL | **FREE** | ⭐ Recommended |
| **OpenRouter** | Qwen 2.5 VL 72B | ~$0.001/req | Higher quality |
| **Google Gemini** | gemini-2.0-flash | FREE tier | Native video |
| **Local vLLM** | Qwen-VL, LLaVA | FREE | Your GPU |

> **Why video-only?** Image-only providers (OpenAI, Anthropic) require extracting frames, losing motion context. Video providers understand actions and timing.

---

## 📹 How It Works

```
Motion Detected
      ↓
┌─────────────────────────────────────┐
│     HA Video Vision                 │
├─────────────────────────────────────┤
│ 1. Record 3-sec video clip (RTSP)   │
│ 2. Run facial recognition           │
│ 3. Send video to AI for analysis    │
│ 4. Return description + people      │
└─────────────────────────────────────┘
      ↓
"Carlos just arrived, walking up
 the driveway with grocery bags"
      ↓
📱 Notification with snapshot
```

---

## 🛠️ Services

### `ha_video_vision.analyze_camera`

Record and analyze a camera with AI.

```yaml
service: ha_video_vision.analyze_camera
data:
  camera: driveway
  duration: 3
  user_query: "Is anyone there?"
  notify: true
```

**Returns:**
```yaml
success: true
camera: driveway
friendly_name: Driveway
description: "A person is walking up the driveway carrying packages."
identified_people:
  - name: Carlos
    confidence: 87
snapshot_url: /media/local/ha_video_vision/driveway_latest.jpg
provider_used: openrouter
```

### `ha_video_vision.record_clip`

Record a clip without analysis.

```yaml
service: ha_video_vision.record_clip
data:
  camera: porch
  duration: 5
```

### `ha_video_vision.identify_faces`

Run facial recognition on an image.

```yaml
service: ha_video_vision.identify_faces
data:
  image_path: /config/www/test.jpg
```

---

## ⚙️ Configuration

### Provider Setup

**OpenRouter (Recommended)**
1. Get free API key at [openrouter.ai](https://openrouter.ai)
2. Default model: `nvidia/nemotron-nano-12b-v2-vl:free`

**Google Gemini**
1. Get API key at [Google AI Studio](https://aistudio.google.com)
2. Model: `gemini-2.0-flash`

**Local vLLM**
1. Run vLLM with a video-capable model (Qwen-VL, LLaVA-Video)
2. Set base URL: `http://your-server:8000/v1`

### Camera Configuration

Format: `name:channel:friendly_name` (one per line)

```
driveway:05:Driveway
porch:03:Front Porch
backyard:04:Backyard
kitchen:01:Kitchen
```

### RTSP Configuration

Works with most NVRs and IP cameras:

```
Host: 192.168.1.100
Port: 554
Username: admin
Password: your_password
Stream: sub (recommended) or main
```

---

## 👤 Facial Recognition Setup

Facial recognition uses a separate **DeepFace server** running on a machine with a GPU.

### 1. Install DeepFace Server

```bash
pip install fastapi uvicorn deepface pillow numpy python-multipart tf-keras
```

### 2. Create Face Database

```
faces/
├── Carlos/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
└── Elise/
    ├── photo1.jpg
    └── photo2.jpg
```

**Tips for best results:**
- 3-5 clear, front-facing photos per person
- Multiple angles (front, 45° left, 45° right)
- Similar lighting to your cameras

### 3. Run Server

```bash
python server.py  # Runs on port 8100
```

### 4. Configure in HA

- **Facial Recognition URL**: `http://your-server:8100`
- **Enable**: ✅
- **Minimum Confidence**: 50-60%

See [docs/FACIAL_RECOGNITION.md](docs/FACIAL_RECOGNITION.md) for full setup guide.

---

## 🔔 Notifications

Notifications work on both iOS and Android with **image attachments**.

**Configuration:**
```
All Services: notify.mobile_app_carlos_iphone, notify.mobile_app_pixel
iOS Devices: notify.mobile_app_carlos_iphone
```

**iOS-specific features:**
- Critical alerts (bypass Do Not Disturb)
- Rich notifications with thumbnails

---

## 📋 Example Automation

```yaml
automation:
  - alias: "AI Doorbell Alert"
    trigger:
      - platform: state
        entity_id: binary_sensor.doorbell_motion
        to: "on"
    action:
      - service: ha_video_vision.analyze_camera
        data:
          camera: porch
          duration: 3
          user_query: "Describe who is at the door"
          notify: true
        response_variable: result
      - service: tts.speak
        data:
          entity_id: media_player.kitchen
          message: "{{ result.description }}"
```

---

## 🤝 Works Great With

- **[PolyVoice](https://github.com/LosCV29/polyvoice)** - Voice control: "Check the driveway camera"
- **ESPHome Voice** - "Hey Mycroft, who's at the door?"
- **Frigate** - Trigger on person detection
- **Any RTSP Camera** - Reolink, Hikvision, Dahua, etc.

---

## 🎤 Voice Control Setup (with PolyVoice)

HA Video Vision integrates seamlessly with [PolyVoice](https://github.com/LosCV29/polyvoice) for voice-controlled camera analysis.

### Supported Voice Commands

| Voice Pattern | Example |
|---------------|---------|
| "Check the [location] camera" | "Check the garage camera" |
| "Is there anyone in [location]" | "Is there anyone in the backyard" |
| "Who is in the [location]" | "Who is in the nursery" |
| "What's happening in [location]" | "What's happening in the driveway" |

### Setup Steps

1. **Install both integrations**:
   - HA Video Vision (this integration)
   - PolyVoice (voice assistant)

2. **Enable cameras in PolyVoice**:
   - Settings → PolyVoice → Configure → Enable Cameras ✅

3. **Configure Voice Aliases** (recommended):
   - Settings → HA Video Vision → Configure → Voice Aliases
   - Map simple names to camera entity IDs:
   ```
   garage:camera.garage_cam
   nursery:camera.baby_room
   backyard:camera.rear_yard
   kitchen:camera.kitchen_cam
   ```

### How Camera Matching Works

When you say "check the garage camera", the system:

1. Extracts "garage" as the location
2. Searches for a matching camera using this priority:
   - **Voice aliases** (exact match first)
   - **Camera friendly names** (e.g., "Garage Camera")
   - **Entity IDs** (e.g., `camera.garage`)
   - **Partial matches** (e.g., "garage" in "garage_rear")
   - **Word matching** (e.g., "garage" in "Main Garage Cam")

### Tips for Best Results

- Use short, unique voice aliases (e.g., "garage", "baby", "porch")
- Avoid similar names for different cameras
- Test with: `ha_video_vision.analyze_camera` service in Developer Tools

---

## 🔧 Troubleshooting

### "Could not record from camera"
- Check RTSP credentials
- Verify camera is accessible: `ffplay rtsp://user:pass@ip:554/...`
- Try "sub" stream instead of "main"

### "Camera analysis unavailable"
- Check API key for cloud providers
- Verify local vLLM server is running
- Check model supports video input

### Notifications not showing images (iOS)
- Ensure device is listed in "iOS Devices"
- Use Nabu Casa paths: `/media/local/ha_video_vision/...`
- Check snapshot directory permissions

### Facial recognition not working
- Verify DeepFace server is running
- Check URL is accessible from HA
- Add more reference photos
- Lower confidence threshold

---

## 📋 Version History

| Version | Changes |
|---------|---------|
| **3.0.0** | Video-only providers, OpenRouter default, simplified config |
| **2.0.0** | Multi-provider, facial recognition |
| **1.0.0** | Initial release |

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Credits

- **DeepFace** - Facial recognition engine
- **Home Assistant** - The best home automation platform

**Built with ❤️ for the Home Assistant community.**

⭐ **Star this repo if you find it useful!**
