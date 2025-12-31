# Facial Recognition Add-on for Home Assistant

This add-on provides facial recognition capabilities for HA Video Vision using DeepFace.

## Quick Start

1. **Install the add-on** from the repository
2. **Add photos** to `/share/faces/PersonName/` folders
3. **Start the add-on**
4. **Configure HA Video Vision** to use `http://localhost:8100`

## Adding Faces

Create folders in `/share/faces/` for each person you want to recognize:

```
/share/faces/
├── John/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── Jane/
│   ├── front.jpg
│   └── side.jpg
└── Mom/
    └── portrait.jpg
```

### Photo Tips

- Use clear, well-lit photos
- Include multiple angles (front, slight left, slight right)
- 3-5 photos per person is ideal
- Similar lighting to your cameras works best
- Avoid blurry or dark photos

## Accessing the Faces Folder

### Via Samba (File Share)
If you have the Samba add-on installed, access `\\homeassistant\share\faces\`

### Via SSH
```bash
cd /share/faces
mkdir "John"
# Copy photos to the folder
```

### Via File Editor
Use the File Editor add-on to navigate to `/share/faces/`

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `distance_threshold` | 0.45 | Match strictness (lower = stricter) |
| `min_face_confidence` | 0.80 | Minimum detection confidence |
| `min_face_size` | 40 | Minimum face size in pixels |
| `model_name` | Facenet512 | Recognition model |
| `detector_backend` | retinaface | Face detector |

### Distance Threshold

- **0.30** - Very strict, may miss matches
- **0.45** - Balanced (recommended)
- **0.60** - Lenient, may have false positives

### Detector Backends

| Backend | Speed | Accuracy |
|---------|-------|----------|
| opencv | Fastest | Good |
| ssd | Fast | Good |
| mtcnn | Medium | Better |
| retinaface | Slower | Best |
| mediapipe | Fast | Good |

## API Endpoints

The add-on exposes an API on port 8100:

### GET /status
Returns server status and loaded faces.

### POST /identify
Identify faces in an image.

```json
{
  "image_base64": "base64_encoded_image_data"
}
```

### POST /reload
Reload faces from disk (use after adding new photos).

## Using with HA Video Vision

1. Install HA Video Vision integration
2. Go to **Settings → Integrations → HA Video Vision → Configure**
3. Select **Facial Recognition**
4. Set URL to `http://localhost:8100`
5. Enable facial recognition
6. Set confidence threshold (35-50% recommended)

## Troubleshooting

### "No known faces loaded"
- Add photos to `/share/faces/PersonName/` folders
- Each person needs their own folder
- Restart the add-on after adding photos

### Not recognizing faces
- Lower the `distance_threshold` (try 0.50-0.60)
- Lower `min_face_confidence` (try 0.70)
- Add more reference photos from different angles
- Ensure reference photos have good lighting

### False positives
- Raise the `distance_threshold` (try 0.35-0.40)
- Raise `min_face_confidence` (try 0.90)
- Use higher quality reference photos

### Slow detection
- Switch `detector_backend` to `opencv` or `ssd`
- Note: First detection is always slow (model loading)

## Resource Usage

- **RAM**: ~500MB - 1GB
- **CPU**: Moderate during detection
- **Startup**: 30-60 seconds (model download on first run)

## Privacy

All processing happens locally on your Home Assistant machine. No images or data are sent to external servers.
