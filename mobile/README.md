# NatiDex Mobile

Android + iOS app built with Capacitor + onnxruntime-web.
Runs fully on-device — no server required.

## Prerequisites

- Node.js 18+
- Python env with torch + torchvision (existing `.venv`)
- Android Studio (for Android)
- Xcode 15+ on macOS (for iOS)

---

## Step 1 — Export the ONNX model

Run from the **project root** (`e:/ML_projects/NatiDex`):

```bash
.venv/Scripts/python mobile/export_onnx.py
```

This reads `best_model.pth` and writes `mobile/assets/model.onnx` (~13 MB).

---

## Step 2 — Copy data assets

From the **project root**:

```bash
cp class_labels.json  mobile/assets/
cp common_names.json  mobile/assets/
cp -r wiki            mobile/assets/wiki
```

> **Note:** `wiki/img/` (~32 MB of thumbnails) is optional.
> Remove it to reduce app size; the app shows a broken image gracefully.

---

## Step 3 — Install Node dependencies

```bash
cd mobile
npm install
```

---

## Step 4 — Add native platforms

```bash
# Android
npx cap add android

# iOS (macOS only)
npx cap add ios
```

After adding, manually add permissions:

**Android** — edit `android/app/src/main/AndroidManifest.xml`, inside `<manifest>`:
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"
    android:maxSdkVersion="32" />
<uses-feature android:name="android.hardware.camera" android:required="false" />
```

**iOS** — edit `ios/App/App/Info.plist`, inside `<dict>`:
```xml
<key>NSCameraUsageDescription</key>
<string>NatiDex uses your camera to photograph and identify species.</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>NatiDex can identify species from photos in your library.</string>
```

---

## Step 5 — Sync and open

```bash
npx cap sync

# Open in Android Studio
npx cap open android

# Open in Xcode (macOS only)
npx cap open ios
```

From Android Studio / Xcode, select a device or emulator and hit Run.

---

## Testing in a desktop browser (no Capacitor)

Open `mobile/index.html` directly in Chrome/Firefox.
The app falls back to a file picker instead of the camera.
This is the easiest way to verify the ONNX model works before building native.

> You may need a local web server for the fetch() calls to work:
> `npx serve .`  (from inside `mobile/`)

---

## Re-syncing after changes

Whenever you change `index.html` or any asset:

```bash
cd mobile
npx cap sync
```

Then rebuild from Android Studio / Xcode.
