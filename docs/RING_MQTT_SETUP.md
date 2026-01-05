# Ring Camera Setup with ring-mqtt

This guide explains how to configure ring-mqtt to work with HA Video Vision for video analysis of Ring cameras.

## The Problem

Ring cameras are **cloud-based** and don't provide native RTSP streams. The default Ring integration in Home Assistant only provides:
- Snapshot images (not live video)
- Event-based recording (stored in Ring's cloud)

**HA Video Vision requires video streams** to perform AI analysis. This is where ring-mqtt comes in.

## Solution: ring-mqtt Add-on

The [ring-mqtt](https://github.com/tsightler/ring-mqtt) add-on provides an RTSP gateway that converts Ring's cloud streams into local RTSP streams that HA Video Vision can analyze.

## Prerequisites

1. Home Assistant with Supervisor (HAOS, Supervised install)
2. Ring account with cameras
3. MQTT broker (Mosquitto add-on recommended)

## Installation Steps

### 1. Install the ring-mqtt Add-on

1. Go to **Settings → Add-ons → Add-on Store**
2. Click the three dots menu (⋮) → **Repositories**
3. Add: `https://github.com/tsightler/ring-mqtt-ha-addon`
4. Find and install **ring-mqtt**

### 2. Configure ring-mqtt

1. Open the ring-mqtt add-on configuration
2. Set your Ring refresh token (see [ring-mqtt wiki](https://github.com/tsightler/ring-mqtt/wiki/Refresh-Tokens) for how to obtain)
3. **Important**: Enable the livestream feature:

```yaml
livestream_user: ""      # Optional: set username for auth
livestream_pass: ""      # Optional: set password for auth
enable_cameras: true     # Must be true!
enable_modes: true
snapshot_mode: "all"     # or "motion" or "interval"
```

4. Start the add-on

### 3. Verify Stream Source is Available

Once ring-mqtt is running:

1. Go to **Settings → Devices & Services**
2. Find your Ring camera device (added via MQTT)
3. Click on the device to see its entities
4. Find the **Info** sensor (e.g., `sensor.front_door_info`)
5. Click on the sensor and expand **Attributes**
6. Look for `stream_source` - it should contain an RTSP URL like:
   ```
   rtsp://homeassistant.local:8554/abc123def_live
   ```

**If `stream_source` is empty or missing:**
- The livestream feature may not be enabled
- The camera may not support live streaming
- Check the ring-mqtt add-on logs for errors

### 4. Test the Stream

You can test the RTSP stream using VLC or ffplay:

```bash
ffplay rtsp://homeassistant.local:8554/YOUR_CAMERA_ID_live
```

Or create a Generic Camera in Home Assistant to verify it works.

### 5. Configure HA Video Vision

HA Video Vision automatically detects ring-mqtt cameras! It looks for:
- `stream_source` attribute on camera entities
- Info sensors with `stream_source` attribute (e.g., `sensor.front_door_info`)

Simply use the camera in your automations:

```yaml
service: ha_video_vision.analyze_camera
data:
  camera: front_door  # Use the camera name
  duration: 3
  notify: true
```

## How HA Video Vision Finds the Stream

HA Video Vision searches for stream URLs in this order:

1. **Home Assistant's camera component** - `async_get_stream_source()`
2. **Camera entity attributes** - `stream_source`, `rtsp_stream`, `rtsp_url`
3. **ring-mqtt Info sensor** - `sensor.{camera}_info` with `stream_source` attribute

## Troubleshooting

### "No Video Stream Available" Error

**Check 1: Is ring-mqtt running?**
- Go to **Settings → Add-ons → ring-mqtt**
- Verify it shows "Running"
- Check the logs for errors

**Check 2: Is the Info sensor created?**
- Go to **Developer Tools → States**
- Search for `sensor.*_info` (e.g., `sensor.front_door_info`)
- If missing, ring-mqtt may not have discovered your camera

**Check 3: Is stream_source populated?**
- Find your camera's Info sensor
- Check its attributes for `stream_source`
- If empty, the livestream feature isn't working

**Check 4: Can you play the stream?**
```bash
# Test with ffplay
ffplay rtsp://YOUR_HA_IP:8554/CAMERA_ID_live

# Or check with ffprobe
ffprobe rtsp://YOUR_HA_IP:8554/CAMERA_ID_live
```

### Stream Works in VLC but Not in HA Video Vision

- Verify the hostname in `stream_source` is reachable from Home Assistant
- If using Docker, ensure network connectivity between containers
- Try replacing hostname with IP address

### Battery Drain on Ring Cameras

Ring cameras are not designed for continuous streaming:
- Streaming drains battery quickly
- Motion detection pauses during active streams
- Use short analysis durations (2-3 seconds)
- Consider wired Ring cameras for frequent analysis

### "Camera offline" in ring-mqtt

- Ring cameras go to sleep when not in use
- ring-mqtt wakes them on demand
- First analysis may take 3-5 seconds to wake the camera

## Alternative: Snapshot-Only Mode

If you can't get live streaming working, HA Video Vision can fall back to analyzing snapshots. However, this provides limited context compared to video analysis.

The snapshot camera created by ring-mqtt (e.g., `camera.front_door_snapshot`) can still be used, but AI analysis will be based on a single frame rather than motion analysis.

## Links

- [ring-mqtt GitHub](https://github.com/tsightler/ring-mqtt)
- [ring-mqtt Wiki - Video Streaming](https://github.com/tsightler/ring-mqtt/wiki/Video-Streaming)
- [ring-mqtt HA Add-on](https://github.com/tsightler/ring-mqtt-ha-addon)
- [Home Assistant 2024.11+ Streaming Changes](https://github.com/tsightler/ring-mqtt/discussions/927)
