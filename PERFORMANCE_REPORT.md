# Smart Classroom Edge Face Recognition System Performance Report

**Test date:** June 6, 2026

**Test version:** `35b7f3a`

**Execution mode:** Local inference on an Android tablet with no cloud-based recognition

## 1. Test Objectives

This test evaluates the performance of face detection, FaceNet feature extraction, and identity recognition in the current Android application. It focuses on:

- Per-frame face detection latency
- Face recognition latency
- End-to-end processing latency and throughput
- First-frame overhead for each session
- Application memory usage
- Performance risks when scaling to classroom-wide multi-person attendance

## 2. Test Environment

| Item | Configuration |
|---|---|
| Tablet manufacturer | HONOR |
| Tablet model | HEY2-W09 |
| Android version | Android 14 |
| SoC platform | parrot |
| Physical display resolution | 1600 x 2560 |
| Face detection model | `face_detection.tflite`, 224 KB |
| Face recognition model | `facenet.tflite`, 90 MB |
| Maximum scan input width | 1280 px |
| Scan request interval | 250 ms |
| Maximum number of detected faces | 24 |
| Inference location | On-device |

## 3. Data and Methodology

Performance data was recorded by the application's built-in `PerformanceLogger`. The log is stored at:

```text
files/performance/performance_log.csv
```

Each record contains the number of detected faces, detection time, recognition time, and total processing time. The test collected **2,970 frames across 13 runtime sessions**. A frame number reset to 1 was treated as the start of a new session, and the first inference of each session was analyzed separately.

Because the historical logs include different camera orientations and debugging versions, the final performance assessment primarily uses the last session from the latest stable version. Historical data is used only to evaluate the range of performance variation.

## 4. Final Stable Version Results

The latest session recorded **300 frames**. Of these, 296 frames contained one detected face, 3 frames contained two detected faces, and 1 frame contained no detected face.

### 4.1 Single-Face Scenario

| Metric | Average | P50 | P90 | P95 | Maximum |
|---|---:|---:|---:|---:|---:|
| Face detection | 334.35 ms | 334.16 ms | 340.77 ms | 343.71 ms | 356.41 ms |
| FaceNet recognition | 617.34 ms | 616.49 ms | 622.54 ms | 625.10 ms | 670.41 ms |
| End-to-end processing | 951.69 ms | 951.65 ms | 960.57 ms | 963.19 ms | 1003.74 ms |

The average end-to-end pipeline throughput in the single-face scenario was approximately **1.05 frames per second**, while the observed continuous sampling rate was approximately **1.00 frame per second**.

### 4.2 Overall Latest Session

| Metric | Average | P50 | P95 | P99 |
|---|---:|---:|---:|---:|
| Face detection | 336.97 ms | 334.16 ms | 344.63 ms | 351.55 ms |
| FaceNet recognition | 615.27 ms | 616.45 ms | 625.09 ms | 631.31 ms |
| End-to-end processing | 952.24 ms | 951.66 ms | 963.48 ms | 971.80 ms |

The recognition stage accounted for approximately **64.6%** of stable-state processing time and was the primary performance bottleneck. The detection stage accounted for approximately **35.4%**.

Although the scan request interval is configured as 250 ms, the application waits for the current recognition task to finish before processing another frame. Therefore, the actual processing rate is determined by the approximately 952 ms end-to-end latency and cannot reach the theoretical rate of 4 frames per second.

## 5. Historical Sessions and First-Frame Overhead

After excluding the first frame of each session, the average end-to-end latency across all historical stable-state samples was **1057.69 ms**, with a P95 latency of **1761.04 ms**. The observed processing rate across historical sessions ranged from approximately **0.61 to 1.02 frames per second**. The variation mainly resulted from camera orientation, rotated-frame detection paths, and whether the image contained a valid face.

The average first-frame latency across the 13 sessions was **3597.37 ms**, with a P95 latency of **4135.34 ms** and a maximum of **4203.76 ms**. In most sessions, the first frame was affected by TensorFlow Lite runtime warm-up and cache initialization.

A warm-up inference that does not affect attendance results is recommended before starting a class. This prevents the first student from experiencing a delay of approximately 3–4 seconds.

## 6. Memory Usage

The Android `dumpsys meminfo` snapshot collected during testing was:

| Memory category | Usage |
|---|---:|
| Application Total PSS | 545,140 KB (approximately 532 MB) |
| Application Total RSS | 695,968 KB (approximately 680 MB) |
| Native Heap PSS | 246,736 KB (approximately 241 MB) |
| Graphics PSS | 141,652 KB (approximately 138 MB) |
| Java Heap PSS | 47,680 KB (approximately 47 MB) |

The current memory usage is relatively high, although the application ran stably on the test tablet. The main sources of memory consumption are native inference memory, camera image bitmaps, and graphics buffers. Long-duration classroom testing should verify that memory usage does not continuously increase.

## 7. Classroom Multi-Person Attendance Assessment

The application supports detecting up to 24 faces per frame. However, only 40 frames in this dataset contained two or three faces, and only 3 frames in the latest stable session contained two faces. The available data demonstrates stable single-face performance, but **it does not prove that the system can maintain the same latency when 24 people appear simultaneously**.

The current FaceNet model is 90 MB, and single-face recognition takes approximately 617 ms. If feature extraction is performed sequentially for multiple faces, recognition latency may increase significantly with the number of students. Before full-classroom deployment, fixed-distance tests with 5, 10, and 20 people should measure:

- Number of detected faces per frame and missed-detection rate
- Time required to identify the entire class for the first time
- Average recognition latency per student
- False recognition rate and proportion of Unknown results
- Temperature, thermal throttling, and memory behavior after 45–90 minutes of continuous operation

## 8. Optimization Recommendations

1. Run an automatic model warm-up before attendance begins to reduce first-frame latency.
2. Evaluate TensorFlow Lite NNAPI or GPU Delegate acceleration, focusing on FaceNet inference.
3. Reduce repeated feature extraction for the same face track and reuse recent recognition results.
4. Use frame-distributed scheduling in multi-person scenarios, recognizing only new or low-confidence faces in each frame to avoid long blocking operations.
5. Establish fixed benchmarks for 5-, 10-, and 20-person scenarios before deciding whether a lighter quantized FaceNet model is required.

## 9. Conclusion

The current version performs face detection, identity recognition, and automatic attendance entirely on the tablet, satisfying the on-device processing requirements of Edge AI. In the final stable single-face scenario, the average end-to-end latency was approximately **952 ms**, with a P95 latency of approximately **963 ms**. FaceNet recognition was the primary source of processing time.

The current version is suitable for functional demonstrations and small-scale attendance. Reliable full-classroom deployment will require realistic multi-person stress testing, followed by optimization of FaceNet inference and multi-face scheduling.
