package com.smartclassroom.tablet;

import android.Manifest;
import android.app.AlertDialog;
import android.app.Activity;
import android.content.Context;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.Face;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.FaceDetector;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Size;
import android.view.Gravity;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.text.SimpleDateFormat;

import org.tensorflow.lite.Interpreter;

public class MainActivity extends Activity {
    private static final int REQUEST_CAMERA = 100;
    private static final long FACE_SCAN_INTERVAL_MS = 250;
    private static final int FACE_SCAN_WIDTH = 1280;
    private static final int MAX_FACES = 24;
    private static final float MIN_SOFTWARE_FACE_CONFIDENCE = 0.25f;
    private static final int FACE_FEATURE_SIZE = 32;
    private static final float RECOGNITION_THRESHOLD = 0.72f;
    private static final float TRACK_RECOGNITION_THRESHOLD = 0.68f;
    private static final int DETECTION_CONFIRM_HITS = 1;
    private static final int RECOGNITION_CONFIRM_HITS = 2;
    private static final long FACE_TRACK_TTL_MS = 3500L;
    private static final String FACE_MODEL_XML = "face-detection-adas-0001.xml";
    private static final String FACE_MODEL_BIN = "face-detection-adas-0001.bin";
    private static final String TFLITE_FACE_DETECTOR = "face_detection.tflite";
    private static final String TFLITE_FACE_EMBEDDER = "facenet.tflite";
    private static final String FACE_DB_PREFS = "smart_classroom_face_db";
    private static final String FACE_DB_NAMES = "names";
    private static final String ATTENDANCE_PREFS = "smart_classroom_attendance";

    private TextureView preview;
    private TextView status;
    private TextView attendanceList;
    private FaceOverlay faceOverlay;
    private HandlerThread cameraThread;
    private Handler cameraHandler;
    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private Size cameraPreviewSize;
    private int cameraSensorOrientation;
    private int cameraLensFacing = CameraCharacteristics.LENS_FACING_FRONT;
    private final Matrix cameraPreviewTransform = new Matrix();
    private boolean faceScanRunning;
    private boolean useFrontCamera = true;
    private int hardwareFaceMode = CaptureRequest.STATISTICS_FACE_DETECT_MODE_OFF;
    private long lastFaceScanAt;
    private FaceDatabase faceDatabase;
    private AttendanceDatabase attendanceDatabase;
    private PerformanceLogger performanceLogger;
    private float[] lastSingleFaceFeature;
    private long lastSingleFaceAt;
    private TfliteFaceDetector tfliteFaceDetector;
    private TfliteFaceEmbedder tfliteFaceEmbedder;
    private final List<FaceTrack> faceTracks = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(
                WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);

        try {
            faceDatabase = new FaceDatabase(this);
            attendanceDatabase = new AttendanceDatabase(this);
            performanceLogger = new PerformanceLogger(this);
            buildUi();
            prepareModel();
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                startCameraThread();
            }
        } catch (Throwable t) {
            showFatalError(t);
        }
    }

    private void buildUi() {
        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.BLACK);

        preview = new TextureView(this);
        preview.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
            @Override
            public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
                openCamera();
            }

            @Override
            public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
                configurePreviewTransform(width, height);
            }

            @Override
            public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
                closeCamera();
                return true;
            }

            @Override
            public void onSurfaceTextureUpdated(SurfaceTexture surface) {
                scanFacesIfNeeded();
            }
        });
        root.addView(preview, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT));

        faceOverlay = new FaceOverlay(this);
        root.addView(faceOverlay, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT));

        TextView title = new TextView(this);
        title.setText("Smart Classroom Face");
        title.setTextSize(20);
        title.setTextColor(Color.WHITE);
        title.setGravity(Gravity.CENTER_VERTICAL);
        title.setPadding(dp(16), 0, dp(16), 0);
        title.setBackgroundColor(Color.argb(120, 0, 0, 0));
        FrameLayout.LayoutParams titleParams = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(54), Gravity.TOP);
        root.addView(title, titleParams);

        status = new TextView(this);
        status.setTextSize(14);
        status.setTextColor(Color.WHITE);
        status.setLineSpacing(dp(2), 1.0f);
        status.setPadding(dp(16), dp(12), dp(16), dp(12));
        status.setBackgroundColor(Color.argb(145, 0, 0, 0));
        FrameLayout.LayoutParams statusParams = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM);
        root.addView(status, statusParams);

        Button cameraButton = new Button(this);
        cameraButton.setText("Camera");
        cameraButton.setAllCaps(false);
        cameraButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    prepareModel();
                    startCameraThread();
                    openCamera();
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
                }
            }
        });
        FrameLayout.LayoutParams buttonParams = new FrameLayout.LayoutParams(
                dp(132), dp(48), Gravity.TOP | Gravity.RIGHT);
        buttonParams.topMargin = dp(70);
        buttonParams.rightMargin = dp(16);
        root.addView(cameraButton, buttonParams);

        Button switchButton = new Button(this);
        switchButton.setText("Front");
        switchButton.setAllCaps(false);
        switchButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                useFrontCamera = !useFrontCamera;
                switchButton.setText(useFrontCamera ? "Front" : "Back");
                closeCamera();
                faceTracks.clear();
                if (faceOverlay != null) {
                    faceOverlay.setFaces(Collections.<RecognizedFace>emptyList());
                }
                setStatus(String.format(Locale.US,
                        "Face model deployed: %s\nCamera: switching to %s camera",
                        FACE_MODEL_XML,
                        useFrontCamera ? "front" : "back"));
                openCamera();
            }
        });
        FrameLayout.LayoutParams switchParams = new FrameLayout.LayoutParams(
                dp(132), dp(48), Gravity.TOP | Gravity.RIGHT);
        switchParams.topMargin = dp(126);
        switchParams.rightMargin = dp(16);
        root.addView(switchButton, switchParams);

        Button enrollButton = new Button(this);
        enrollButton.setText("Enroll");
        enrollButton.setAllCaps(false);
        enrollButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                showEnrollDialog();
            }
        });
        FrameLayout.LayoutParams enrollParams = new FrameLayout.LayoutParams(
                dp(132), dp(48), Gravity.TOP | Gravity.RIGHT);
        enrollParams.topMargin = dp(182);
        enrollParams.rightMargin = dp(16);
        root.addView(enrollButton, enrollParams);

        TextView attendanceTitle = new TextView(this);
        attendanceTitle.setText("Attendance");
        attendanceTitle.setTextSize(17);
        attendanceTitle.setTextColor(Color.WHITE);
        attendanceTitle.setGravity(Gravity.CENTER_VERTICAL);
        attendanceTitle.setPadding(dp(12), 0, dp(12), 0);
        attendanceTitle.setBackgroundColor(Color.argb(175, 0, 0, 0));
        FrameLayout.LayoutParams attendanceTitleParams = new FrameLayout.LayoutParams(
                dp(300), dp(42), Gravity.TOP | Gravity.RIGHT);
        attendanceTitleParams.topMargin = dp(246);
        attendanceTitleParams.rightMargin = dp(16);
        root.addView(attendanceTitle, attendanceTitleParams);

        Button resetAttendanceButton = new Button(this);
        resetAttendanceButton.setText("Reset");
        resetAttendanceButton.setAllCaps(false);
        resetAttendanceButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                attendanceDatabase.clearToday();
                refreshAttendanceList();
            }
        });
        FrameLayout.LayoutParams resetAttendanceParams = new FrameLayout.LayoutParams(
                dp(92), dp(42), Gravity.TOP | Gravity.RIGHT);
        resetAttendanceParams.topMargin = dp(246);
        resetAttendanceParams.rightMargin = dp(16);
        root.addView(resetAttendanceButton, resetAttendanceParams);

        attendanceList = new TextView(this);
        attendanceList.setTextSize(14);
        attendanceList.setTextColor(Color.WHITE);
        attendanceList.setLineSpacing(dp(5), 1.0f);
        attendanceList.setPadding(dp(12), dp(10), dp(12), dp(10));
        attendanceList.setBackgroundColor(Color.argb(155, 0, 0, 0));
        ScrollView attendanceScroll = new ScrollView(this);
        attendanceScroll.addView(attendanceList, new ScrollView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));
        FrameLayout.LayoutParams attendanceParams = new FrameLayout.LayoutParams(
                dp(300), dp(560), Gravity.TOP | Gravity.RIGHT);
        attendanceParams.topMargin = dp(288);
        attendanceParams.rightMargin = dp(16);
        root.addView(attendanceScroll, attendanceParams);

        setContentView(root);
        refreshAttendanceList();
    }

    private void showEnrollDialog() {
        if (lastSingleFaceFeature == null
                || SystemClock.elapsedRealtime() - lastSingleFaceAt > 3000) {
            setStatus("Enrollment needs exactly one visible face. Stand in frame and try again.");
            return;
        }
        final EditText input = new EditText(this);
        input.setSingleLine(true);
        input.setHint("Student name");
        input.setTextColor(Color.BLACK);
        input.setHintTextColor(Color.GRAY);
        input.setPadding(dp(12), dp(6), dp(12), dp(6));
        new AlertDialog.Builder(this)
                .setTitle("Enroll face")
                .setMessage("Enter the name for the current single face.")
                .setView(input)
                .setPositiveButton("Save", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        String name = input.getText().toString().trim();
                        if (name.length() == 0) {
                            setStatus("Enrollment skipped: empty name.");
                            return;
                        }
                        faceDatabase.addSample(name, lastSingleFaceFeature);
                        refreshAttendanceList();
                        setStatus(String.format(Locale.US,
                                "Enrolled: %s\nKnown people: %d  Samples: %d",
                                name,
                                faceDatabase.personCount(),
                                faceDatabase.sampleCount()));
                    }
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    private void prepareModel() {
        File modelDir = new File(getFilesDir(), "model");
        if (!modelDir.exists() && !modelDir.mkdirs()) {
            setStatus("Model directory error: " + modelDir.getAbsolutePath());
            return;
        }

        try {
            copyAssetIfNeeded("model/" + FACE_MODEL_XML, new File(modelDir, FACE_MODEL_XML));
            copyAssetIfNeeded("model/" + FACE_MODEL_BIN, new File(modelDir, FACE_MODEL_BIN));
            copyAssetIfNeeded("model/" + TFLITE_FACE_DETECTOR, new File(modelDir, TFLITE_FACE_DETECTOR));
            copyAssetIfNeeded("model/" + TFLITE_FACE_EMBEDDER, new File(modelDir, TFLITE_FACE_EMBEDDER));
            initTfliteModels(new File(modelDir, TFLITE_FACE_DETECTOR), new File(modelDir, TFLITE_FACE_EMBEDDER));
            seedRosterFromAssets();
            refreshAttendanceList();
            File xml = new File(modelDir, FACE_MODEL_XML);
            File bin = new File(modelDir, FACE_MODEL_BIN);
            setStatus(String.format(Locale.US,
                    "Face models deployed  OV %.1f KB/%.1f MB  TFLite: %s\nCamera: preparing  Face detection: waiting for preview",
                    xml.length() / 1024.0,
                    bin.length() / 1024.0 / 1024.0,
                    tfliteFaceEmbedder != null ? "embedding ready" : "fallback only"));
        } catch (Exception e) {
            setStatus("Model copy failed: " + e.getMessage());
        }
    }

    private void initTfliteModels(File detectorModel, File embedderModel) {
        try {
            if (tfliteFaceDetector == null && detectorModel.isFile() && detectorModel.length() > 1024) {
                tfliteFaceDetector = new TfliteFaceDetector(detectorModel);
            }
            if (tfliteFaceEmbedder == null && embedderModel.isFile() && embedderModel.length() > 1024) {
                tfliteFaceEmbedder = new TfliteFaceEmbedder(embedderModel);
            }
        } catch (Exception e) {
            tfliteFaceDetector = null;
            tfliteFaceEmbedder = null;
            setStatus("TFLite init failed: " + e.getMessage());
        }
    }

    private void seedRosterFromAssets() {
        if (tfliteFaceEmbedder == null) {
            return;
        }
        try {
            String[] entries = getAssets().list("roster");
            if (entries == null) {
                return;
            }
            for (String entry : entries) {
                if (entry.startsWith(".")) {
                    continue;
                }
                String rosterPath = "roster/" + entry;
                String[] childFiles = getAssets().list(rosterPath);
                if (childFiles != null && childFiles.length > 0) {
                    seedStudentFolder(entry, rosterPath, childFiles);
                }
            }
            for (String entry : entries) {
                if (entry.startsWith(".")) {
                    continue;
                }
                String rosterPath = "roster/" + entry;
                String[] childFiles = getAssets().list(rosterPath);
                if ((childFiles == null || childFiles.length == 0) && isImageFile(entry)) {
                    String name = displayNameFromAsset(entry);
                    if (!faceDatabase.hasPerson(name)) {
                        seedStudentPhoto(name, rosterPath);
                    }
                }
            }
        } catch (Exception ignored) {
        }
    }

    private void seedStudentFolder(String folderName, String rosterPath, String[] childFiles) {
        String name = displayNameFromAsset(folderName);
        if (faceDatabase.hasPerson(name)) {
            return;
        }
        for (String child : childFiles) {
            if (isImageFile(child)) {
                seedStudentPhoto(name, rosterPath + "/" + child);
            }
        }
    }

    private void seedStudentPhoto(String name, String assetPath) {
        try (InputStream input = getAssets().open(assetPath)) {
            Bitmap photo = BitmapFactory.decodeStream(input);
            if (photo == null) {
                return;
            }
            List<FaceCandidate> faces = detectTfliteFaces(photo, photo.getWidth(), photo.getHeight());
            if (faces.isEmpty()) {
                faces.add(centerFaceCandidate(photo.getWidth(), photo.getHeight(), photo.getWidth(), photo.getHeight()));
            }
            FaceCandidate face = faces.get(0);
            RectF sourceBox = face.sourceBox;
            for (float scale : new float[]{0.92f, 1.0f, 1.08f}) {
                faceDatabase.addSample(name,
                        tfliteFaceEmbedder.embed(
                                photo,
                                scaleBox(sourceBox, scale, photo.getWidth(), photo.getHeight()),
                                face.leftEye,
                                face.rightEye));
            }
            photo.recycle();
        } catch (Exception ignored) {
        }
    }

    private boolean isImageFile(String name) {
        String lower = name.toLowerCase(Locale.US);
        return lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".png");
    }

    private String displayNameFromAsset(String assetName) {
        int dot = assetName.lastIndexOf('.');
        String base = dot > 0 ? assetName.substring(0, dot) : assetName;
        return base.replace('_', ' ').trim();
    }

    private void copyAssetIfNeeded(String assetName, File output) throws IOException {
        if (output.exists() && output.length() > 0) {
            return;
        }
        InputStream assetStream;
        try {
            assetStream = getAssets().open(assetName);
        } catch (IOException first) {
            assetStream = getAssets().open(assetName.replace('/', '\\'));
        }
        try (InputStream input = assetStream;
             FileOutputStream out = new FileOutputStream(output)) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = input.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
        }
    }

    private void startCameraThread() {
        if (cameraThread != null) {
            return;
        }
        cameraThread = new HandlerThread("SmartClassroomCamera");
        cameraThread.start();
        cameraHandler = new Handler(cameraThread.getLooper());
        if (preview.isAvailable()) {
            openCamera();
        }
    }

    private void openCamera() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED
                || !preview.isAvailable()
                || cameraDevice != null) {
            return;
        }
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            String cameraId = chooseCamera(manager, useFrontCamera);
            if (cameraId == null) {
                setStatus(status.getText() + "\nCamera: no available camera");
                return;
            }
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            hardwareFaceMode = chooseFaceDetectMode(characteristics);
            Integer sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
            Integer lensFacing = characteristics.get(CameraCharacteristics.LENS_FACING);
            cameraSensorOrientation = sensorOrientation == null ? 0 : sensorOrientation;
            cameraLensFacing = lensFacing == null
                    ? CameraCharacteristics.LENS_FACING_FRONT
                    : lensFacing;
            cameraPreviewSize = choosePreviewSize(characteristics, preview.getWidth(), preview.getHeight());
            configurePreviewTransform(preview.getWidth(), preview.getHeight());
            manager.openCamera(cameraId, new CameraDevice.StateCallback() {
                @Override
                public void onOpened(CameraDevice camera) {
                    cameraDevice = camera;
                    startPreview();
                }

                @Override
                public void onDisconnected(CameraDevice camera) {
                    camera.close();
                    cameraDevice = null;
                }

                @Override
                public void onError(CameraDevice camera, int error) {
                    camera.close();
                    cameraDevice = null;
                    setStatus(status.getText() + "\nCamera open failed: " + error);
                }
            }, cameraHandler);
        } catch (CameraAccessException e) {
            setStatus(status.getText() + "\nCamera: " + e.getMessage());
        }
    }

    private int chooseFaceDetectMode(CameraCharacteristics characteristics) {
        int[] modes = characteristics.get(CameraCharacteristics.STATISTICS_INFO_AVAILABLE_FACE_DETECT_MODES);
        if (modes == null) {
            return CaptureRequest.STATISTICS_FACE_DETECT_MODE_OFF;
        }
        int selected = CaptureRequest.STATISTICS_FACE_DETECT_MODE_OFF;
        for (int mode : modes) {
            if (mode == CameraMetadata.STATISTICS_FACE_DETECT_MODE_FULL) {
                return CaptureRequest.STATISTICS_FACE_DETECT_MODE_FULL;
            }
            if (mode == CameraMetadata.STATISTICS_FACE_DETECT_MODE_SIMPLE) {
                selected = CaptureRequest.STATISTICS_FACE_DETECT_MODE_SIMPLE;
            }
        }
        return selected;
    }

    private String chooseCamera(CameraManager manager, boolean preferFront) throws CameraAccessException {
        String fallback = null;
        int targetFacing = preferFront
                ? CameraCharacteristics.LENS_FACING_FRONT
                : CameraCharacteristics.LENS_FACING_BACK;
        for (String id : manager.getCameraIdList()) {
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(id);
            Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
            if (fallback == null) {
                fallback = id;
            }
            if (facing != null && facing == targetFacing) {
                return id;
            }
        }
        return fallback;
    }

    private Size choosePreviewSize(CameraCharacteristics characteristics, int viewWidth, int viewHeight) {
        StreamConfigurationMap map =
                characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        Size[] choices = map == null ? null : map.getOutputSizes(SurfaceTexture.class);
        if (choices == null || choices.length == 0) {
            return new Size(Math.max(1, viewWidth), Math.max(1, viewHeight));
        }

        int relativeRotation = relativeCameraRotation();
        int targetWidth = relativeRotation == 90 || relativeRotation == 270 ? viewHeight : viewWidth;
        int targetHeight = relativeRotation == 90 || relativeRotation == 270 ? viewWidth : viewHeight;
        float targetRatio = targetWidth / (float) Math.max(1, targetHeight);

        List<Size> suitable = new ArrayList<>();
        for (Size choice : choices) {
            if (choice.getWidth() > 2560 || choice.getHeight() > 2560) {
                continue;
            }
            float ratio = choice.getWidth() / (float) choice.getHeight();
            if (Math.abs(ratio - targetRatio) < 0.08f
                    && choice.getWidth() >= Math.min(targetWidth, 1280)
                    && choice.getHeight() >= Math.min(targetHeight, 720)) {
                suitable.add(choice);
            }
        }
        if (!suitable.isEmpty()) {
            return Collections.min(suitable, Comparator.comparingLong(
                    size -> (long) size.getWidth() * size.getHeight()));
        }

        Size best = choices[0];
        float bestDifference = Float.MAX_VALUE;
        for (Size choice : choices) {
            if (choice.getWidth() > 2560 || choice.getHeight() > 2560) {
                continue;
            }
            float difference = Math.abs(
                    choice.getWidth() / (float) choice.getHeight() - targetRatio);
            if (difference < bestDifference) {
                bestDifference = difference;
                best = choice;
            }
        }
        return best;
    }

    private void startPreview() {
        try {
            SurfaceTexture texture = preview.getSurfaceTexture();
            if (texture == null || cameraDevice == null) {
                return;
            }
            if (cameraPreviewSize == null) {
                cameraPreviewSize = new Size(
                        Math.max(preview.getWidth(), 1280),
                        Math.max(preview.getHeight(), 720));
            }
            texture.setDefaultBufferSize(
                    cameraPreviewSize.getWidth(),
                    cameraPreviewSize.getHeight());
            configurePreviewTransform(preview.getWidth(), preview.getHeight());
            Surface surface = new Surface(texture);
            CaptureRequest.Builder builder =
                    cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            builder.addTarget(surface);
            builder.set(CaptureRequest.STATISTICS_FACE_DETECT_MODE, hardwareFaceMode);
            cameraDevice.createCaptureSession(Collections.singletonList(surface),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(CameraCaptureSession session) {
                            captureSession = session;
                            try {
                                session.setRepeatingRequest(builder.build(), new CameraCaptureSession.CaptureCallback() {
                                    @Override
                                    public void onCaptureCompleted(CameraCaptureSession session,
                                                                   CaptureRequest request,
                                                                   TotalCaptureResult result) {
                                        updateHardwareFaces(result);
                                    }
                                }, cameraHandler);
                                setStatus(status.getText().toString()
                                        .replace("Camera: preparing", "Camera: preview running"));
                            } catch (CameraAccessException e) {
                                setStatus(status.getText() + "\nCamera: " + e.getMessage());
                            }
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession session) {
                            setStatus(status.getText() + "\nCamera configuration failed");
                        }
                    }, cameraHandler);
        } catch (CameraAccessException e) {
            setStatus(status.getText() + "\nCamera: " + e.getMessage());
        }
    }

    private void configurePreviewTransform(int viewWidth, int viewHeight) {
        if (preview == null || cameraPreviewSize == null || viewWidth <= 0 || viewHeight <= 0) {
            return;
        }
        int relativeRotation = relativeCameraRotation();
        RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
        RectF bufferRect;
        if (relativeRotation == 90 || relativeRotation == 270) {
            bufferRect = new RectF(
                    0,
                    0,
                    cameraPreviewSize.getHeight(),
                    cameraPreviewSize.getWidth());
        } else {
            bufferRect = new RectF(
                    0,
                    0,
                    cameraPreviewSize.getWidth(),
                    cameraPreviewSize.getHeight());
        }
        float centerX = viewRect.centerX();
        float centerY = viewRect.centerY();
        bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
        Matrix matrix = new Matrix();
        matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
        float scale = Math.max(
                viewHeight / (float) cameraPreviewSize.getHeight(),
                viewWidth / (float) cameraPreviewSize.getWidth());
        matrix.postScale(scale, scale, centerX, centerY);
        matrix.postRotate(relativeRotation, centerX, centerY);
        if (cameraLensFacing == CameraCharacteristics.LENS_FACING_FRONT) {
            matrix.postScale(-1f, 1f, centerX, centerY);
        }
        synchronized (cameraPreviewTransform) {
            cameraPreviewTransform.set(matrix);
        }
        runOnUiThread(() -> preview.setTransform(matrix));
    }

    private int relativeCameraRotation() {
        if (cameraLensFacing == CameraCharacteristics.LENS_FACING_BACK) {
            return (360 - cameraSensorOrientation) % 360;
        }
        return cameraSensorOrientation;
    }

    private void closeCamera() {
        if (captureSession != null) {
            captureSession.close();
            captureSession = null;
        }
        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
        }
    }

    private void updateHardwareFaces(CaptureResult result) {
        Face[] faces = result.get(CaptureResult.STATISTICS_FACES);
        if (faces == null || hardwareFaceMode == CaptureRequest.STATISTICS_FACE_DETECT_MODE_OFF) {
            return;
        }
    }

    private void scanFacesIfNeeded() {
        long now = SystemClock.elapsedRealtime();
        if (faceScanRunning || cameraHandler == null || now - lastFaceScanAt < FACE_SCAN_INTERVAL_MS) {
            return;
        }
        lastFaceScanAt = now;
        faceScanRunning = true;

        int viewWidth = preview.getWidth();
        int viewHeight = preview.getHeight();
        if (viewWidth <= 0 || viewHeight <= 0) {
            faceScanRunning = false;
            return;
        }

        int scanWidth = Math.min(FACE_SCAN_WIDTH, viewWidth);
        if ((scanWidth & 1) == 1) {
            scanWidth--;
        }
        int scanHeight = Math.max(2, Math.round(viewHeight * (scanWidth / (float) viewWidth)));
        Bitmap snapshot = preview.getBitmap(scanWidth, scanHeight);
        if (snapshot == null) {
            faceScanRunning = false;
            return;
        }

        cameraHandler.post(new Runnable() {
            @Override
            public void run() {
                detectFaces(snapshot, viewWidth, viewHeight);
            }
        });
    }

    private void detectFaces(Bitmap snapshot, int viewWidth, int viewHeight) {
        long totalStart = SystemClock.elapsedRealtimeNanos();
        try {
            long detectionStart = SystemClock.elapsedRealtimeNanos();
            List<FaceCandidate> candidates = findFacesInAnyRotation(snapshot, viewWidth, viewHeight);
            double detectionMs = elapsedMs(detectionStart);

            long recognitionStart = SystemClock.elapsedRealtimeNanos();
            List<RecognizedFace> faces = recognizeFaces(snapshot, candidates);
            double recognitionMs = elapsedMs(recognitionStart);
            PerformanceLogger.Snapshot metrics = performanceLogger.record(
                    faces.size(),
                    detectionMs,
                    recognitionMs,
                    elapsedMs(totalStart));
            updateFaceResults(faces, metrics);
        } catch (Throwable t) {
            setStatus("Face detection failed: " + t.getClass().getSimpleName() + ": " + t.getMessage());
        } finally {
            snapshot.recycle();
            faceScanRunning = false;
        }
    }

    private double elapsedMs(long startNanos) {
        return (SystemClock.elapsedRealtimeNanos() - startNanos) / 1_000_000.0;
    }

    private List<FaceCandidate> findFacesInAnyRotation(Bitmap snapshot, int viewWidth, int viewHeight) {
        List<FaceCandidate> tfliteFaces = detectTfliteFaces(snapshot, viewWidth, viewHeight);
        if (!tfliteFaces.isEmpty()) {
            return tfliteFaces;
        }

        int[] rotations = {0, 90, 270, 180};
        List<FaceCandidate> bestFaces = new ArrayList<>();
        for (int rotation : rotations) {
            Bitmap candidate = makeDetectionBitmap(snapshot, rotation);
            try {
                List<FaceCandidate> faces = findFacesInBitmap(candidate, rotation, snapshot.getWidth(), snapshot.getHeight(),
                        viewWidth, viewHeight);
                if (faces.size() > bestFaces.size()) {
                    bestFaces = faces;
                }
            } finally {
                if (candidate != snapshot) {
                    candidate.recycle();
                }
            }
        }
        return bestFaces;
    }

    private List<FaceCandidate> detectTfliteFaces(Bitmap snapshot, int viewWidth, int viewHeight) {
        if (tfliteFaceDetector == null) {
            return Collections.emptyList();
        }

        int[] rotations = {0, 90, 270, 180};
        List<FaceCandidate> allFaces = new ArrayList<>();
        for (int rotation : rotations) {
            Bitmap candidate = makeDetectionBitmap(snapshot, rotation);
            try {
                List<FaceCandidate> rotatedFaces = tfliteFaceDetector.detect(
                        candidate, candidate.getWidth(), candidate.getHeight());
                for (FaceCandidate rotatedFace : rotatedFaces) {
                    RectF sourceBox = clampBox(
                            mapRotatedBoxToSource(rotatedFace.sourceBox, rotation,
                                    snapshot.getWidth(), snapshot.getHeight()),
                            snapshot.getWidth(), snapshot.getHeight());
                    PointF leftEye = rotatedFace.leftEye == null ? null
                            : mapRotatedPointToSource(rotatedFace.leftEye, rotation,
                                    snapshot.getWidth(), snapshot.getHeight());
                    PointF rightEye = rotatedFace.rightEye == null ? null
                            : mapRotatedPointToSource(rotatedFace.rightEye, rotation,
                                    snapshot.getWidth(), snapshot.getHeight());
                    if (leftEye != null && rightEye != null && leftEye.x > rightEye.x) {
                        PointF swap = leftEye;
                        leftEye = rightEye;
                        rightEye = swap;
                    }
                    if (containsDuplicateFace(allFaces, sourceBox)) {
                        continue;
                    }
                    RectF viewBox = new RectF(sourceBox);
                    viewBox.left *= viewWidth / (float) snapshot.getWidth();
                    viewBox.right *= viewWidth / (float) snapshot.getWidth();
                    viewBox.top *= viewHeight / (float) snapshot.getHeight();
                    viewBox.bottom *= viewHeight / (float) snapshot.getHeight();
                    allFaces.add(new FaceCandidate(sourceBox, viewBox, leftEye, rightEye));
                }
            } catch (Exception ignored) {
                // Try the remaining orientations before falling back to the legacy detector.
            } finally {
                if (candidate != snapshot) {
                    candidate.recycle();
                }
            }
        }
        return allFaces;
    }

    private boolean containsDuplicateFace(List<FaceCandidate> faces, RectF box) {
        for (FaceCandidate face : faces) {
            if (iou(face.sourceBox, box) > 0.22f) {
                return true;
            }
            float left = Math.max(face.sourceBox.left, box.left);
            float top = Math.max(face.sourceBox.top, box.top);
            float right = Math.min(face.sourceBox.right, box.right);
            float bottom = Math.min(face.sourceBox.bottom, box.bottom);
            float intersection = Math.max(0f, right - left) * Math.max(0f, bottom - top);
            float smallerArea = Math.min(
                    face.sourceBox.width() * face.sourceBox.height(),
                    box.width() * box.height());
            if (smallerArea > 0f && intersection / smallerArea > 0.58f) {
                return true;
            }
        }
        return false;
    }

    private FaceCandidate centerFaceCandidate(int sourceWidth, int sourceHeight, int viewWidth, int viewHeight) {
        float boxWidth = sourceWidth * 0.62f;
        float boxHeight = sourceHeight * 0.66f;
        float left = (sourceWidth - boxWidth) * 0.5f;
        float top = sourceHeight * 0.14f;
        RectF sourceBox = clampBox(new RectF(left, top, left + boxWidth, top + boxHeight),
                sourceWidth, sourceHeight);
        RectF viewBox = new RectF(sourceBox);
        viewBox.left *= viewWidth / (float) sourceWidth;
        viewBox.right *= viewWidth / (float) sourceWidth;
        viewBox.top *= viewHeight / (float) sourceHeight;
        viewBox.bottom *= viewHeight / (float) sourceHeight;
        return new FaceCandidate(sourceBox, viewBox);
    }

    private Bitmap makeDetectionBitmap(Bitmap source, int rotation) {
        Bitmap rgb565 = source.copy(Bitmap.Config.RGB_565, false);
        if (rotation == 0) {
            return rgb565;
        }
        Matrix matrix = new Matrix();
        matrix.postRotate(rotation);
        Bitmap rotated = Bitmap.createBitmap(rgb565, 0, 0, rgb565.getWidth(), rgb565.getHeight(), matrix, true);
        rgb565.recycle();
        return rotated;
    }

    private List<FaceCandidate> findFacesInBitmap(Bitmap bitmap, int rotation, int sourceWidth, int sourceHeight,
                                                  int viewWidth, int viewHeight) {
        FaceDetector.Face[] faces = new FaceDetector.Face[MAX_FACES];
        FaceDetector detector = new FaceDetector(bitmap.getWidth(), bitmap.getHeight(), MAX_FACES);
        int count = detector.findFaces(bitmap, faces);
        List<FaceCandidate> results = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            FaceDetector.Face face = faces[i];
            if (face == null || face.confidence() < MIN_SOFTWARE_FACE_CONFIDENCE) {
                continue;
            }
            PointF mid = new PointF();
            face.getMidPoint(mid);
            float eyeDistance = face.eyesDistance();
            float halfWidth = eyeDistance * 1.55f;
            float halfHeight = eyeDistance * 1.85f;
            RectF rotatedBox = new RectF(
                    mid.x - halfWidth,
                    mid.y - halfHeight,
                    mid.x + halfWidth,
                    mid.y + halfHeight * 1.30f);
            RectF sourceBox = mapRotatedBoxToSource(rotatedBox, rotation, sourceWidth, sourceHeight);
            RectF boundedSourceBox = clampBox(sourceBox, sourceWidth, sourceHeight);
            RectF viewBox = new RectF(boundedSourceBox);
            viewBox.left *= viewWidth / (float) sourceWidth;
            viewBox.right *= viewWidth / (float) sourceWidth;
            viewBox.top *= viewHeight / (float) sourceHeight;
            viewBox.bottom *= viewHeight / (float) sourceHeight;
            PointF sourceMid = mapRotatedPointToSource(mid, rotation, sourceWidth, sourceHeight);
            PointF leftEye = new PointF(sourceMid.x - eyeDistance * 0.5f, sourceMid.y);
            PointF rightEye = new PointF(sourceMid.x + eyeDistance * 0.5f, sourceMid.y);
            results.add(new FaceCandidate(boundedSourceBox, viewBox, leftEye, rightEye));
        }
        return results;
    }

    private RectF clampBox(RectF box, int width, int height) {
        float left = Math.max(0f, Math.min(width - 1f, box.left));
        float top = Math.max(0f, Math.min(height - 1f, box.top));
        float right = Math.max(left + 1f, Math.min(width, box.right));
        float bottom = Math.max(top + 1f, Math.min(height, box.bottom));
        return new RectF(left, top, right, bottom);
    }

    private RectF mapRotatedBoxToSource(RectF box, int rotation, int sourceWidth, int sourceHeight) {
        if (rotation == 90) {
            return new RectF(
                    box.top,
                    sourceHeight - box.right,
                    box.bottom,
                    sourceHeight - box.left);
        }
        if (rotation == 270) {
            return new RectF(
                    sourceWidth - box.bottom,
                    box.left,
                    sourceWidth - box.top,
                    box.right);
        }
        if (rotation == 180) {
            return new RectF(
                    sourceWidth - box.right,
                    sourceHeight - box.bottom,
                    sourceWidth - box.left,
                    sourceHeight - box.top);
        }
        return box;
    }

    private PointF mapRotatedPointToSource(PointF point, int rotation, int sourceWidth, int sourceHeight) {
        if (rotation == 90) {
            return new PointF(point.y, sourceHeight - point.x);
        }
        if (rotation == 270) {
            return new PointF(sourceWidth - point.y, point.x);
        }
        if (rotation == 180) {
            return new PointF(sourceWidth - point.x, sourceHeight - point.y);
        }
        return point;
    }

    private List<RecognizedFace> recognizeFaces(Bitmap snapshot, List<FaceCandidate> candidates) {
        List<RecognizedFace> results = new ArrayList<>();
        long now = SystemClock.elapsedRealtime();
        pruneFaceTracks(now);
        if (candidates.size() == 1) {
            lastSingleFaceFeature = extractRecognitionFeatures(snapshot, candidates.get(0)).get(0);
            lastSingleFaceAt = SystemClock.elapsedRealtime();
        } else {
            lastSingleFaceFeature = null;
        }

        for (FaceCandidate candidate : candidates) {
            List<float[]> features = extractRecognitionFeatures(snapshot, candidate);
            FaceMatch match = faceDatabase.matchBest(features);
            String label = "Unknown";
            float score = 0f;
            FaceTrack track = findOrCreateTrack(candidate.sourceBox, now);
            if (match != null && match.score >= RECOGNITION_THRESHOLD) {
                score = match.score;
            } else if (match != null && match.score >= TRACK_RECOGNITION_THRESHOLD
                    && match.name.equals(track.bestName)) {
                score = Math.max(match.score, track.bestScore);
                track.update(match.name, score, candidate.sourceBox, now);
            } else if (match != null) {
                score = match.score;
            }
            if (match != null && match.score >= RECOGNITION_THRESHOLD) {
                track.update(match.name, match.score, candidate.sourceBox, now);
            }
            if (track.isUsable(now)) {
                label = track.bestName;
                score = track.bestScore;
            } else {
                track.touch(candidate.sourceBox, now);
            }
            if (!track.isDetectionConfirmed()) {
                continue;
            }
            results.add(new RecognizedFace(candidate.viewBox, label, score));
        }
        return results;
    }

    private void pruneFaceTracks(long now) {
        for (int i = faceTracks.size() - 1; i >= 0; i--) {
            if (now - faceTracks.get(i).lastSeenAt > FACE_TRACK_TTL_MS) {
                faceTracks.remove(i);
            }
        }
    }

    private FaceTrack findOrCreateTrack(RectF box, long now) {
        FaceTrack best = null;
        float bestIou = 0f;
        for (FaceTrack track : faceTracks) {
            float overlap = iou(box, track.lastBox);
            if (overlap > bestIou) {
                bestIou = overlap;
                best = track;
            }
        }
        if (best != null && bestIou >= 0.25f) {
            best.observe(box, now);
            return best;
        }
        FaceTrack created = new FaceTrack(box, now);
        faceTracks.add(created);
        return created;
    }

    private List<float[]> extractRecognitionFeatures(Bitmap source, RectF sourceBox) {
        return extractRecognitionFeatures(source, new FaceCandidate(sourceBox, sourceBox));
    }

    private List<float[]> extractRecognitionFeatures(Bitmap source, FaceCandidate candidate) {
        RectF sourceBox = candidate.sourceBox;
        if (tfliteFaceEmbedder != null) {
            List<float[]> features = new ArrayList<>();
            if (candidate.leftEye != null && candidate.rightEye != null) {
                features.add(tfliteFaceEmbedder.embed(
                        source,
                        sourceBox,
                        candidate.leftEye,
                        candidate.rightEye));
            }
            float baseScale = Math.min(sourceBox.width(), sourceBox.height()) < 90f ? 1.18f : 1.0f;
            for (float scale : new float[]{1.0f, 1.18f}) {
                features.add(tfliteFaceEmbedder.embed(source,
                        scaleBox(sourceBox, scale * baseScale, source.getWidth(), source.getHeight()),
                        null,
                        null));
            }
            return features;
        }
        return extractFaceFeatureVariants(source, sourceBox);
    }

    private List<float[]> extractFaceFeatureVariants(Bitmap source, RectF sourceBox) {
        List<float[]> features = new ArrayList<>();
        float[] scales = {1.0f, 0.82f, 1.16f};
        for (float scale : scales) {
            features.add(extractFaceFeature(source, scaleBox(sourceBox, scale, source.getWidth(), source.getHeight())));
        }
        return features;
    }

    private RectF scaleBox(RectF box, float scale, int width, int height) {
        float cx = (box.left + box.right) * 0.5f;
        float cy = (box.top + box.bottom) * 0.5f;
        float halfWidth = box.width() * scale * 0.5f;
        float halfHeight = box.height() * scale * 0.5f;
        return clampBox(new RectF(cx - halfWidth, cy - halfHeight, cx + halfWidth, cy + halfHeight),
                width, height);
    }

    private float[] extractFaceFeature(Bitmap source, RectF sourceBox) {
        int left = Math.max(0, Math.round(sourceBox.left));
        int top = Math.max(0, Math.round(sourceBox.top));
        int right = Math.min(source.getWidth(), Math.round(sourceBox.right));
        int bottom = Math.min(source.getHeight(), Math.round(sourceBox.bottom));
        int width = Math.max(1, right - left);
        int height = Math.max(1, bottom - top);
        Bitmap crop = Bitmap.createBitmap(source, left, top, width, height);
        Bitmap scaled = Bitmap.createScaledBitmap(crop, FACE_FEATURE_SIZE, FACE_FEATURE_SIZE, true);
        if (crop != scaled) {
            crop.recycle();
        }

        int pixelCount = FACE_FEATURE_SIZE * FACE_FEATURE_SIZE;
        float[] feature = new float[pixelCount];
        float sum = 0f;
        int index = 0;
        for (int y = 0; y < FACE_FEATURE_SIZE; y++) {
            for (int x = 0; x < FACE_FEATURE_SIZE; x++) {
                int color = scaled.getPixel(x, y);
                float gray = ((Color.red(color) * 0.299f)
                        + (Color.green(color) * 0.587f)
                        + (Color.blue(color) * 0.114f)) / 255f;
                feature[index++] = gray;
                sum += gray;
            }
        }
        scaled.recycle();

        float mean = sum / pixelCount;
        float norm = 0f;
        for (int i = 0; i < feature.length; i++) {
            feature[i] -= mean;
            norm += feature[i] * feature[i];
        }
        norm = (float) Math.sqrt(Math.max(norm, 1e-6f));
        for (int i = 0; i < feature.length; i++) {
            feature[i] /= norm;
        }
        return feature;
    }

    private void updateFaceResults(List<RecognizedFace> faces, PerformanceLogger.Snapshot metrics) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                boolean attendanceChanged = false;
                for (RecognizedFace face : faces) {
                    if (!"Unknown".equals(face.label) && faceDatabase.hasPerson(face.label)) {
                        attendanceChanged |= attendanceDatabase.markPresent(face.label);
                    }
                }
                if (attendanceChanged) {
                    refreshAttendanceList();
                }
                if (faceOverlay != null) {
                    faceOverlay.setFaces(transformFaceBoxes(faces));
                }
                if (status != null) {
                    status.setText(String.format(Locale.US,
                            "Local edge face recognition  Camera: %s\nFaces: %d  Known people: %d  Samples: %d\n%s",
                            useFrontCamera ? "front" : "back",
                            faces.size(),
                            faceDatabase.personCount(),
                            faceDatabase.sampleCount(),
                            metrics.toStatusString()));
                }
            }
        });
    }

    private List<RecognizedFace> transformFaceBoxes(List<RecognizedFace> faces) {
        List<RecognizedFace> transformed = new ArrayList<>(faces.size());
        Matrix matrix = new Matrix();
        synchronized (cameraPreviewTransform) {
            matrix.set(cameraPreviewTransform);
        }
        for (RecognizedFace face : faces) {
            RectF box = new RectF(face.box);
            matrix.mapRect(box);
            transformed.add(new RecognizedFace(box, face.label, face.score));
        }
        return transformed;
    }

    private void refreshAttendanceList() {
        if (attendanceList == null || faceDatabase == null || attendanceDatabase == null) {
            return;
        }
        List<String> names = faceDatabase.getNames();
        StringBuilder text = new StringBuilder();
        int presentCount = 0;
        for (String name : names) {
            if (attendanceDatabase.isPresent(name)) {
                presentCount++;
            }
        }
        text.append(String.format(Locale.US, "%d / %d present\n\n", presentCount, names.size()));
        if (names.isEmpty()) {
            text.append("No enrolled students");
        } else {
            for (String name : names) {
                String time = attendanceDatabase.getPresentTime(name);
                text.append(time == null ? "Absent  " : "Present " + time + "  ");
                text.append(name).append('\n');
            }
        }
        attendanceList.setText(text.toString());
    }

    @Override
    protected void onDestroy() {
        closeCamera();
        if (tfliteFaceDetector != null) {
            tfliteFaceDetector.close();
            tfliteFaceDetector = null;
        }
        if (tfliteFaceEmbedder != null) {
            tfliteFaceEmbedder.close();
            tfliteFaceEmbedder = null;
        }
        if (cameraThread != null) {
            cameraThread.quitSafely();
            cameraThread = null;
        }
        if (performanceLogger != null) {
            performanceLogger.close();
            performanceLogger = null;
        }
        super.onDestroy();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA
                && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCameraThread();
        } else {
            setStatus(status.getText() + "\nCamera permission denied");
        }
    }

    private void showFatalError(Throwable t) {
        TextView fallback = new TextView(this);
        fallback.setText("Smart Classroom Edge\nStartup failed:\n"
                + t.getClass().getSimpleName() + ": " + t.getMessage());
        fallback.setTextSize(18);
        fallback.setTextColor(Color.rgb(160, 32, 32));
        fallback.setGravity(Gravity.CENTER);
        fallback.setPadding(dp(24), dp(24), dp(24), dp(24));
        setContentView(fallback);
    }

    private void setStatus(String text) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (status != null) {
                    status.setText(text);
                }
            }
        });
    }

    private int dp(int value) {
        return (int) (value * getResources().getDisplayMetrics().density + 0.5f);
    }

    private static class FaceOverlay extends View {
        private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint labelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint labelBackgroundPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final List<RecognizedFace> faces = new ArrayList<>();

        FaceOverlay(Context context) {
            super(context);
            boxPaint.setStyle(Paint.Style.STROKE);
            boxPaint.setStrokeWidth(5f);
            boxPaint.setColor(Color.rgb(0, 255, 128));
            labelPaint.setColor(Color.rgb(0, 255, 128));
            labelPaint.setTextSize(34f);
            labelPaint.setFakeBoldText(true);
            labelBackgroundPaint.setColor(Color.argb(170, 0, 0, 0));
            labelBackgroundPaint.setStyle(Paint.Style.FILL);
        }

        void setFaces(List<RecognizedFace> boxes) {
            faces.clear();
            faces.addAll(boxes);
            invalidate();
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            for (int i = 0; i < faces.size(); i++) {
                RecognizedFace face = faces.get(i);
                RectF box = face.box;
                canvas.drawRect(box, boxPaint);
                String label = face.score > 0f
                        ? String.format(Locale.US, "%s %.2f", face.label, face.score)
                        : face.label;
                float x = Math.max(8f, box.left);
                float y = Math.max(42f, box.top - 10f);
                float width = labelPaint.measureText(label) + 18f;
                canvas.drawRect(x - 8f, y - 36f, x + width, y + 8f, labelBackgroundPaint);
                canvas.drawText(label, x, y, labelPaint);
            }
        }
    }

    private static class FaceCandidate {
        final RectF sourceBox;
        final RectF viewBox;
        final PointF leftEye;
        final PointF rightEye;

        FaceCandidate(RectF sourceBox, RectF viewBox) {
            this(sourceBox, viewBox, null, null);
        }

        FaceCandidate(RectF sourceBox, RectF viewBox, PointF leftEye, PointF rightEye) {
            this.sourceBox = sourceBox;
            this.viewBox = viewBox;
            this.leftEye = leftEye;
            this.rightEye = rightEye;
        }
    }

    private static class RecognizedFace {
        final RectF box;
        final String label;
        final float score;

        RecognizedFace(RectF box, String label, float score) {
            this.box = box;
            this.label = label;
            this.score = score;
        }
    }

    private static class FaceMatch {
        final String name;
        final float score;

        FaceMatch(String name, float score) {
            this.name = name;
            this.score = score;
        }
    }

    private static class FaceTrack {
        RectF lastBox;
        String bestName = null;
        float bestScore = 0f;
        long lastSeenAt;
        long lastRecognizedAt;
        int detectionHits = 1;
        int recognitionHits = 0;

        FaceTrack(RectF box, long now) {
            lastBox = new RectF(box);
            lastSeenAt = now;
        }

        void observe(RectF box, long now) {
            if (now - lastSeenAt <= 1000L) {
                detectionHits++;
            } else {
                detectionHits = 1;
            }
            touch(box, now);
        }

        void touch(RectF box, long now) {
            lastBox = new RectF(box);
            lastSeenAt = now;
        }

        void update(String name, float score, RectF box, long now) {
            touch(box, now);
            if (name != null && name.equals(bestName)) {
                recognitionHits++;
            } else if (name != null) {
                bestName = name;
                bestScore = score;
                recognitionHits = 1;
            }
            if (name != null && score >= bestScore) {
                bestScore = score;
            }
            lastRecognizedAt = now;
        }

        boolean isDetectionConfirmed() {
            return detectionHits >= DETECTION_CONFIRM_HITS;
        }

        boolean isUsable(long now) {
            return bestName != null
                    && bestScore >= TRACK_RECOGNITION_THRESHOLD
                    && recognitionHits >= RECOGNITION_CONFIRM_HITS
                    && now - lastRecognizedAt <= FACE_TRACK_TTL_MS;
        }
    }

    private static class FaceDatabase {
        private final SharedPreferences preferences;
        private final List<FaceSample> samples = new ArrayList<>();
        private final Set<String> names = new HashSet<>();

        FaceDatabase(Context context) {
            preferences = context.getSharedPreferences(FACE_DB_PREFS, Context.MODE_PRIVATE);
            load();
        }

        void addSample(String name, float[] feature) {
            String cleanName = name.trim();
            if (cleanName.length() == 0 || feature == null) {
                return;
            }
            samples.add(new FaceSample(cleanName, feature.clone()));
            names.add(cleanName);
            save();
        }

        boolean hasPerson(String name) {
            return names.contains(name);
        }

        FaceMatch match(float[] feature) {
            if (feature == null || samples.isEmpty()) {
                return null;
            }
            String bestName = null;
            float bestScore = -1f;
            for (String name : names) {
                float first = -1f;
                float second = -1f;
                int count = 0;
                for (FaceSample sample : samples) {
                    if (!sample.name.equals(name)) {
                        continue;
                    }
                    count++;
                    float score = cosine(feature, sample.feature);
                    if (score > first) {
                        second = first;
                        first = score;
                    } else if (score > second) {
                        second = score;
                    }
                }
                if (count == 0) {
                    continue;
                }
                float score = count >= 2 ? (first + second) * 0.5f : first;
                if (score > bestScore) {
                    bestScore = score;
                    bestName = name;
                }
            }
            return bestName == null ? null : new FaceMatch(bestName, bestScore);
        }

        FaceMatch matchBest(List<float[]> features) {
            if (features == null || features.isEmpty()) {
                return null;
            }
            FaceMatch best = null;
            for (float[] feature : features) {
                FaceMatch candidate = match(feature);
                if (candidate != null && (best == null || candidate.score > best.score)) {
                    best = candidate;
                }
            }
            return best;
        }

        int personCount() {
            return names.size();
        }

        int sampleCount() {
            return samples.size();
        }

        List<String> getNames() {
            List<String> result = new ArrayList<>(names);
            Collections.sort(result);
            return result;
        }

        private void load() {
            samples.clear();
            names.clear();
            String storedNames = preferences.getString(FACE_DB_NAMES, "");
            if (storedNames == null || storedNames.length() == 0) {
                return;
            }
            String[] encodedNames = storedNames.split("\\|", -1);
            for (String encodedName : encodedNames) {
                String name = decodeName(encodedName);
                if (name.length() == 0) {
                    continue;
                }
                names.add(name);
                int count = preferences.getInt("count_" + encodedName, 0);
                for (int i = 0; i < count; i++) {
                    float[] feature = decodeFeature(preferences.getString("sample_" + encodedName + "_" + i, ""));
                    if (feature != null) {
                        samples.add(new FaceSample(name, feature));
                    }
                }
            }
        }

        private void save() {
            SharedPreferences.Editor editor = preferences.edit();
            editor.clear();
            StringBuilder encodedNames = new StringBuilder();
            List<String> orderedNames = new ArrayList<>(names);
            for (int i = 0; i < orderedNames.size(); i++) {
                String name = orderedNames.get(i);
                String encodedName = encodeName(name);
                if (i > 0) {
                    encodedNames.append('|');
                }
                encodedNames.append(encodedName);
                int sampleIndex = 0;
                for (FaceSample sample : samples) {
                    if (sample.name.equals(name)) {
                        editor.putString("sample_" + encodedName + "_" + sampleIndex, encodeFeature(sample.feature));
                        sampleIndex++;
                    }
                }
                editor.putInt("count_" + encodedName, sampleIndex);
            }
            editor.putString(FACE_DB_NAMES, encodedNames.toString());
            editor.apply();
        }

        private static float cosine(float[] a, float[] b) {
            if (a.length != b.length) {
                return -1f;
            }
            float dot = 0f;
            for (int i = 0; i < a.length; i++) {
                dot += a[i] * b[i];
            }
            return dot;
        }

        private static String encodeFeature(float[] feature) {
            StringBuilder builder = new StringBuilder(feature.length * 7);
            for (int i = 0; i < feature.length; i++) {
                if (i > 0) {
                    builder.append(',');
                }
                builder.append(String.format(Locale.US, "%.5f", feature[i]));
            }
            return builder.toString();
        }

        private static float[] decodeFeature(String value) {
            if (value == null || value.length() == 0) {
                return null;
            }
            String[] parts = value.split(",");
            float[] feature = new float[parts.length];
            try {
                for (int i = 0; i < parts.length; i++) {
                    feature[i] = Float.parseFloat(parts[i]);
                }
            } catch (NumberFormatException e) {
                return null;
            }
            return feature;
        }

        private static String encodeName(String name) {
            return name.replace("%", "%25").replace("|", "%7C").replace("_", "%5F");
        }

        private static String decodeName(String name) {
            return name.replace("%5F", "_").replace("%7C", "|").replace("%25", "%");
        }
    }

    private static class AttendanceDatabase {
        private final SharedPreferences preferences;
        private final SimpleDateFormat dayFormat = new SimpleDateFormat("yyyy-MM-dd", Locale.US);
        private final SimpleDateFormat timeFormat = new SimpleDateFormat("HH:mm:ss", Locale.US);

        AttendanceDatabase(Context context) {
            preferences = context.getSharedPreferences(ATTENDANCE_PREFS, Context.MODE_PRIVATE);
        }

        boolean markPresent(String name) {
            String key = keyFor(name);
            if (preferences.contains(key)) {
                return false;
            }
            preferences.edit().putString(key, timeFormat.format(new Date())).apply();
            return true;
        }

        boolean isPresent(String name) {
            return preferences.contains(keyFor(name));
        }

        String getPresentTime(String name) {
            return preferences.getString(keyFor(name), null);
        }

        void clearToday() {
            String prefix = dayFormat.format(new Date()) + "_";
            SharedPreferences.Editor editor = preferences.edit();
            for (String key : preferences.getAll().keySet()) {
                if (key.startsWith(prefix)) {
                    editor.remove(key);
                }
            }
            editor.apply();
        }

        private String keyFor(String name) {
            return dayFormat.format(new Date()) + "_" + FaceDatabase.encodeName(name);
        }
    }

    private static class FaceSample {
        final String name;
        final float[] feature;

        FaceSample(String name, float[] feature) {
            this.name = name;
            this.feature = feature;
        }
    }

    private static class TfliteFaceDetector {
        private final Interpreter interpreter;
        private final int inputSize = 128;
        private final float fullFrameThreshold = 0.55f;
        private final float tileThreshold = 0.70f;
        private final List<Anchor> anchors;

        TfliteFaceDetector(File modelFile) throws IOException {
            interpreter = new Interpreter(mapFile(modelFile));
            anchors = generateBlazeFaceAnchors();
        }

        List<FaceCandidate> detect(Bitmap bitmap, int viewWidth, int viewHeight) {
            List<ScoredBox> candidates = new ArrayList<>();
            runDetector(bitmap, 0, 0, fullFrameThreshold, candidates);
            runTiledDetector(bitmap, candidates);
            candidates.sort((a, b) -> Float.compare(b.score, a.score));

            List<FaceCandidate> faces = new ArrayList<>();
            for (ScoredBox candidate : candidates) {
                if (faces.size() >= MAX_FACES) {
                    break;
                }
                RectF sourceBox = clampStatic(candidate.box, bitmap.getWidth(), bitmap.getHeight());
                if (!isPlausibleFace(sourceBox, candidate.leftEye, candidate.rightEye)) {
                    continue;
                }
                boolean overlaps = false;
                for (FaceCandidate existing : faces) {
                    if (isDuplicateBox(sourceBox, existing.sourceBox)) {
                        overlaps = true;
                        break;
                    }
                }
                if (overlaps) {
                    continue;
                }
                RectF viewBox = new RectF(sourceBox);
                viewBox.left *= viewWidth / (float) bitmap.getWidth();
                viewBox.right *= viewWidth / (float) bitmap.getWidth();
                viewBox.top *= viewHeight / (float) bitmap.getHeight();
                viewBox.bottom *= viewHeight / (float) bitmap.getHeight();
                faces.add(new FaceCandidate(sourceBox, viewBox, candidate.leftEye, candidate.rightEye));
            }
            return faces;
        }

        private void runTiledDetector(Bitmap bitmap, List<ScoredBox> candidates) {
            int columns = 3;
            int rows = 2;
            float overlap = 0.18f;
            int tileWidth = Math.round(bitmap.getWidth() / (columns - (columns - 1) * overlap));
            int tileHeight = Math.round(bitmap.getHeight() / (rows - (rows - 1) * overlap));
            int stepX = Math.max(1, Math.round(tileWidth * (1f - overlap)));
            int stepY = Math.max(1, Math.round(tileHeight * (1f - overlap)));

            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    int left = Math.min(column * stepX, Math.max(0, bitmap.getWidth() - tileWidth));
                    int top = Math.min(row * stepY, Math.max(0, bitmap.getHeight() - tileHeight));
                    int width = Math.min(tileWidth, bitmap.getWidth() - left);
                    int height = Math.min(tileHeight, bitmap.getHeight() - top);
                    if (width < 64 || height < 64) {
                        continue;
                    }
                    Bitmap tile = Bitmap.createBitmap(bitmap, left, top, width, height);
                    try {
                        runDetector(tile, left, top, tileThreshold, candidates);
                    } finally {
                        tile.recycle();
                    }
                }
            }
        }

        private void runDetector(Bitmap bitmap, int offsetX, int offsetY, float scoreThreshold,
                                 List<ScoredBox> candidates) {
            ByteBuffer input = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4);
            input.order(ByteOrder.nativeOrder());
            Bitmap resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true);
            int[] pixels = new int[inputSize * inputSize];
            resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize);
            if (resized != bitmap) {
                resized.recycle();
            }
            for (int pixel : pixels) {
                input.putFloat(((Color.red(pixel) / 127.5f) - 1f));
                input.putFloat(((Color.green(pixel) / 127.5f) - 1f));
                input.putFloat(((Color.blue(pixel) / 127.5f) - 1f));
            }
            input.rewind();

            float[][][] out0 = new float[1][896][];
            float[][][] out1 = new float[1][896][];
            int[] shape0 = interpreter.getOutputTensor(0).shape();
            int[] shape1 = interpreter.getOutputTensor(1).shape();
            out0[0] = new float[shape0[1]][shape0[2]];
            out1[0] = new float[shape1[1]][shape1[2]];
            interpreter.runForMultipleInputsOutputs(new Object[]{input}, mapOutputs(out0, out1));

            float[][] scores;
            float[][] boxes;
            if (shape0[2] == 1) {
                scores = out0[0];
                boxes = out1[0];
            } else {
                scores = out1[0];
                boxes = out0[0];
            }

            int limit = Math.min(scores.length, anchors.size());
            for (int i = 0; i < limit; i++) {
                float score = sigmoid(scores[i][0]);
                if (score < scoreThreshold) {
                    continue;
                }
                float[] b = boxes[i];
                Anchor anchor = anchors.get(i);
                float cx = (b[0] / inputSize + anchor.xCenter) * bitmap.getWidth();
                float cy = (b[1] / inputSize + anchor.yCenter) * bitmap.getHeight();
                float bw = Math.max(1f, (b[2] / inputSize) * bitmap.getWidth());
                float bh = Math.max(1f, (b[3] / inputSize) * bitmap.getHeight());
                RectF box = new RectF(
                        offsetX + cx - bw * 0.5f,
                        offsetY + cy - bh * 0.5f,
                        offsetX + cx + bw * 0.5f,
                        offsetY + cy + bh * 0.5f);
                PointF eye0 = decodeKeypoint(b, anchor, 4, bitmap.getWidth(), bitmap.getHeight());
                PointF eye1 = decodeKeypoint(b, anchor, 6, bitmap.getWidth(), bitmap.getHeight());
                eye0.offset(offsetX, offsetY);
                eye1.offset(offsetX, offsetY);
                PointF leftEye = eye0.x <= eye1.x ? eye0 : eye1;
                PointF rightEye = eye0.x <= eye1.x ? eye1 : eye0;
                candidates.add(new ScoredBox(box, score, leftEye, rightEye));
            }
        }

        private static boolean isPlausibleFace(RectF box, PointF leftEye, PointF rightEye) {
            if (box.width() < 14f || box.height() < 14f) {
                return false;
            }
            float ratio = box.width() / box.height();
            if (ratio < 0.45f || ratio > 1.8f || leftEye == null || rightEye == null) {
                return false;
            }
            float eyeDistance = Math.abs(rightEye.x - leftEye.x);
            float eyeVertical = Math.abs(rightEye.y - leftEye.y);
            if (eyeDistance < box.width() * 0.06f || eyeDistance > box.width() * 0.90f
                    || eyeVertical > box.height() * 0.45f) {
                return false;
            }
            return containsWithMargin(box, leftEye, 0.35f) && containsWithMargin(box, rightEye, 0.35f);
        }

        private static boolean containsWithMargin(RectF box, PointF point, float marginRatio) {
            float marginX = box.width() * marginRatio;
            float marginY = box.height() * marginRatio;
            return point.x >= box.left - marginX && point.x <= box.right + marginX
                    && point.y >= box.top - marginY && point.y <= box.bottom + marginY;
        }

        private static boolean isDuplicateBox(RectF a, RectF b) {
            if (iou(a, b) > 0.22f) {
                return true;
            }
            float left = Math.max(a.left, b.left);
            float top = Math.max(a.top, b.top);
            float right = Math.min(a.right, b.right);
            float bottom = Math.min(a.bottom, b.bottom);
            float intersection = Math.max(0f, right - left) * Math.max(0f, bottom - top);
            float smallerArea = Math.min(a.width() * a.height(), b.width() * b.height());
            return smallerArea > 0f && intersection / smallerArea > 0.58f;
        }

        void close() {
            interpreter.close();
        }

        private static java.util.Map<Integer, Object> mapOutputs(Object out0, Object out1) {
            java.util.Map<Integer, Object> outputs = new java.util.HashMap<>();
            outputs.put(0, out0);
            outputs.put(1, out1);
            return outputs;
        }

        private static float sigmoid(float x) {
            return (float) (1.0 / (1.0 + Math.exp(-x)));
        }

        private PointF decodeKeypoint(float[] boxData, Anchor anchor, int offset, int imageWidth, int imageHeight) {
            if (boxData.length <= offset + 1) {
                return null;
            }
            return new PointF(
                    (boxData[offset] / inputSize + anchor.xCenter) * imageWidth,
                    (boxData[offset + 1] / inputSize + anchor.yCenter) * imageHeight);
        }

        private static List<Anchor> generateBlazeFaceAnchors() {
            List<Anchor> result = new ArrayList<>(896);
            addAnchorGrid(result, 16, 16, 2);
            addAnchorGrid(result, 8, 8, 6);
            return result;
        }

        private static void addAnchorGrid(List<Anchor> anchors, int featureMapWidth,
                                          int featureMapHeight, int anchorsPerCell) {
            for (int y = 0; y < featureMapHeight; y++) {
                for (int x = 0; x < featureMapWidth; x++) {
                    float xCenter = (x + 0.5f) / featureMapWidth;
                    float yCenter = (y + 0.5f) / featureMapHeight;
                    for (int i = 0; i < anchorsPerCell; i++) {
                        anchors.add(new Anchor(xCenter, yCenter));
                    }
                }
            }
        }
    }

    private static class Anchor {
        final float xCenter;
        final float yCenter;

        Anchor(float xCenter, float yCenter) {
            this.xCenter = xCenter;
            this.yCenter = yCenter;
        }
    }

    private static class TfliteFaceEmbedder {
        private final Interpreter interpreter;
        private final int inputWidth;
        private final int inputHeight;
        private final int outputLength;

        TfliteFaceEmbedder(File modelFile) throws IOException {
            interpreter = new Interpreter(mapFile(modelFile));
            int[] inputShape = interpreter.getInputTensor(0).shape();
            inputHeight = inputShape[1];
            inputWidth = inputShape[2];
            int[] outShape = interpreter.getOutputTensor(0).shape();
            int len = 1;
            for (int i = 1; i < outShape.length; i++) {
                len *= outShape[i];
            }
            outputLength = len;
        }

        float[] embed(Bitmap source, RectF sourceBox) {
            return embed(source, sourceBox, null, null);
        }

        float[] embed(Bitmap source, RectF sourceBox, PointF leftEye, PointF rightEye) {
            Bitmap resized = alignedFaceBitmap(source, sourceBox, leftEye, rightEye);

            ByteBuffer input = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3 * 4);
            input.order(ByteOrder.nativeOrder());
            int[] pixels = new int[inputWidth * inputHeight];
            resized.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight);
            resized.recycle();
            for (int pixel : pixels) {
                input.putFloat((Color.red(pixel) - 127.5f) / 128f);
                input.putFloat((Color.green(pixel) - 127.5f) / 128f);
                input.putFloat((Color.blue(pixel) - 127.5f) / 128f);
            }
            input.rewind();

            float[][] output = new float[1][outputLength];
            interpreter.run(input, output);
            float[] feature = output[0];
            normalize(feature);
            return feature;
        }

        private Bitmap alignedFaceBitmap(Bitmap source, RectF sourceBox, PointF leftEye, PointF rightEye) {
            if (leftEye != null && rightEye != null
                    && Math.abs(rightEye.x - leftEye.x) > 2f
                    && Math.abs(rightEye.y - leftEye.y) < source.getHeight() * 0.35f) {
                Bitmap aligned = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888);
                Canvas canvas = new Canvas(aligned);
                canvas.drawColor(Color.BLACK);
                Matrix matrix = new Matrix();
                float[] src = {leftEye.x, leftEye.y, rightEye.x, rightEye.y};
                float[] dst = {
                        inputWidth * 0.35f, inputHeight * 0.38f,
                        inputWidth * 0.65f, inputHeight * 0.38f
                };
                if (matrix.setPolyToPoly(src, 0, dst, 0, 2)) {
                    Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
                    canvas.drawBitmap(source, matrix, paint);
                    return aligned;
                }
                aligned.recycle();
            }

            RectF box = clampStatic(sourceBox, source.getWidth(), source.getHeight());
            Bitmap crop = Bitmap.createBitmap(
                    source,
                    Math.round(box.left),
                    Math.round(box.top),
                    Math.max(1, Math.round(box.width())),
                    Math.max(1, Math.round(box.height())));
            Bitmap resized = Bitmap.createScaledBitmap(crop, inputWidth, inputHeight, true);
            if (crop != resized) {
                crop.recycle();
            }
            return resized;
        }

        void close() {
            interpreter.close();
        }
    }

    private static class ScoredBox {
        final RectF box;
        final float score;
        final PointF leftEye;
        final PointF rightEye;

        ScoredBox(RectF box, float score) {
            this(box, score, null, null);
        }

        ScoredBox(RectF box, float score, PointF leftEye, PointF rightEye) {
            this.box = box;
            this.score = score;
            this.leftEye = leftEye;
            this.rightEye = rightEye;
        }
    }

    private static MappedByteBuffer mapFile(File file) throws IOException {
        try (FileInputStream input = new FileInputStream(file);
             FileChannel channel = input.getChannel()) {
            return channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length());
        }
    }

    private static RectF clampStatic(RectF box, int width, int height) {
        float left = Math.max(0f, Math.min(width - 1f, box.left));
        float top = Math.max(0f, Math.min(height - 1f, box.top));
        float right = Math.max(left + 1f, Math.min(width, box.right));
        float bottom = Math.max(top + 1f, Math.min(height, box.bottom));
        return new RectF(left, top, right, bottom);
    }

    private static float iou(RectF a, RectF b) {
        float left = Math.max(a.left, b.left);
        float top = Math.max(a.top, b.top);
        float right = Math.min(a.right, b.right);
        float bottom = Math.min(a.bottom, b.bottom);
        float inter = Math.max(0f, right - left) * Math.max(0f, bottom - top);
        float union = a.width() * a.height() + b.width() * b.height() - inter;
        return union <= 0f ? 0f : inter / union;
    }

    private static void normalize(float[] feature) {
        float norm = 0f;
        for (float value : feature) {
            norm += value * value;
        }
        norm = (float) Math.sqrt(Math.max(norm, 1e-6f));
        for (int i = 0; i < feature.length; i++) {
            feature[i] /= norm;
        }
    }
}
