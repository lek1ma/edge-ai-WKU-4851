# Smart Classroom Edge

Smart Classroom Edge is an Android and Python edge-AI demo for classroom camera scenarios.

## Contents

- `Project/android_app`: Android tablet app with camera preview, front/back camera switching, local face enrollment, multi-face recognition, and automatic attendance.
- `Project/smart_classroom_edge`: Python OpenVINO demo scripts for person counting and attendance-session experiments.

## Android Build

The Android project uses the local build script:

```powershell
cd Project/android_app
./build_apk.ps1
```

The script expects an Android SDK under `Project/tools/android-sdk`, which is intentionally not committed.

The Android project can also be opened directly in Android Studio from `Project/android_app`.
The Gradle debug build is:

```bash
cd Project/android_app
gradle assembleDebug
```

## Android Face Enrollment

The tablet app keeps the face database on the tablet with `SharedPreferences`, so recognition remains local edge-AI work.

1. Run the app on the tablet and make sure exactly one face is visible.
2. Tap `Enroll`.
3. Enter the student's name and save.
4. Repeat enrollment 3-5 times per student under different lighting or angles for better matching.
5. During preview, detected faces are labeled with the saved name and score, or `Unknown` when the local similarity score is below the threshold.

The current Android version uses TensorFlow Lite models on the tablet:

- `face_detection.tflite` for face detection
- `facenet.tflite` for face embeddings
- local `SharedPreferences` gallery for student vectors

## Automatic Attendance

The tablet app builds an attendance roster from the local face gallery. After a known
student is recognized with the existing multi-frame confirmation logic, the student is
marked present and the first recognition time is recorded.

- Attendance records are stored locally on the tablet.
- Records are separated by date and survive app restarts.
- Newly enrolled students are added to the roster automatically.
- The `Reset` button clears the current day's attendance for a new class or test.

Bundled roster photos under `assets/roster/` are converted into local embeddings on first launch, then live camera faces are compared against that on-device gallery.

Recommended batch roster layout:

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

Folder names become display names by replacing `_` with spaces. Add 2-5 photos per student when possible. After changing roster assets, reinstall the APK and clear the app data if the student already exists in the local gallery; otherwise the app keeps the previously generated embeddings.

## Python Demo

```powershell
cd Project/smart_classroom_edge
pip install -r requirements.txt
python run_classroom.py
```

Large third-party SDKs, Open Model Zoo checkouts, generated APKs, and build work directories are ignored by Git.
