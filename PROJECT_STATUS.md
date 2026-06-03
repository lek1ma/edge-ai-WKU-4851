# Smart Classroom Edge - Current Status

Last updated: 2026-06-02

## Android App

- Project path: `Project/android_app`
- Android Studio / Gradle project is now available.
- Latest debug APK:
  - `Project/android_app/app/build/outputs/apk/debug/app-debug.apk`
- Package:
  - `com.smartclassroom.tablet`

## Current Recognition Pipeline

- Face detection:
  - `assets/model/face_detection.tflite`
  - BlazeFace output is anchor-decoded.
  - Detection boxes are now aligned with faces.
  - Classroom mode runs full-frame detection plus overlapping 3x2 tile detection.
  - Tile detections are merged with global NMS.
  - Scan width is `1280` and maximum displayed faces is `24`.
- Face embedding:
  - `assets/model/facenet.tflite`
  - TensorFlow Lite runs locally on the tablet.
- Face alignment:
  - Uses BlazeFace eye keypoints.
  - Aligns eyes to fixed positions before FaceNet embedding.
  - Falls back to crop/resize if keypoints are unavailable.
- Recognition:
  - Local `SharedPreferences` gallery.
  - Multi-sample matching uses top-2 average per person.
  - Recognition threshold is currently `0.82`.
  - Track fallback threshold is currently `0.78`.
- Stabilization:
  - Short-term face tracks keep the best recognized identity for about 3.5 seconds.
  - Small faces use expanded crop scales.
  - Aligned faces run one FaceNet embedding instead of repeated scale embeddings, improving multi-face performance.

## Roster Import

Recommended batch layout:

```text
Project/android_app/app/src/main/assets/roster/
  Haoran_Guo/
    1.jpg
    2.jpg
    3.jpg
  Alice_Wang/
    1.jpg
    2.jpg
```

Folder names become display names by replacing `_` with spaces.

Current bundled sample:

```text
Project/android_app/app/src/main/assets/roster/Haoran_Guo/1.jpg
```

## Tablet State

- The latest face-alignment APK has been installed and launched on the connected tablet.
- App data was cleared after adding alignment, so old non-aligned enroll samples were removed.
- On first launch, the app rebuilt Haoran Guo's aligned roster gallery.
- Confirmed local gallery:
  - `Haoran Guo`
  - `3` aligned samples

## Recommended Next Steps

1. Re-enroll Haoran Guo on the tablet 1-2 times using the new alignment version.
2. Add 2-5 photos per student under `assets/roster/<Student_Name>/`.
3. Reinstall the APK and clear app data when roster assets change for existing students.
4. Implement attendance state:
   - first seen
   - last seen
   - present / uncertain / absent
   - multi-frame confirmation before marking present
