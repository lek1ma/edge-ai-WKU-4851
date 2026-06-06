package com.smartclassroom.tablet;

import android.content.Context;
import android.util.Log;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Locale;

final class PerformanceLogger {
    private static final String TAG = "FacePerformance";
    private static final int WINDOW_SIZE = 30;
    private static final int LOG_INTERVAL = 30;

    private final double[] detectionWindow = new double[WINDOW_SIZE];
    private final double[] recognitionWindow = new double[WINDOW_SIZE];
    private final double[] totalWindow = new double[WINDOW_SIZE];
    private final BufferedWriter csvWriter;

    private int windowIndex;
    private int windowCount;
    private long frameCount;
    private Snapshot latest = new Snapshot(0, 0, 0, 0, 0, 0, 0);

    PerformanceLogger(Context context) {
        BufferedWriter writer = null;
        try {
            File directory = new File(context.getFilesDir(), "performance");
            if (directory.exists() || directory.mkdirs()) {
                File output = new File(directory, "performance_log.csv");
                boolean needsHeader = !output.exists() || output.length() == 0;
                writer = new BufferedWriter(new FileWriter(output, true));
                if (needsHeader) {
                    writer.write("frame,timestamp_ms,faces,detection_ms,recognition_ms,total_ms\n");
                    writer.flush();
                }
            }
        } catch (IOException error) {
            Log.w(TAG, "CSV logging unavailable", error);
        }
        csvWriter = writer;
    }

    synchronized Snapshot record(int faces, double detectionMs, double recognitionMs, double totalMs) {
        frameCount++;
        detectionWindow[windowIndex] = detectionMs;
        recognitionWindow[windowIndex] = recognitionMs;
        totalWindow[windowIndex] = totalMs;
        windowIndex = (windowIndex + 1) % WINDOW_SIZE;
        windowCount = Math.min(windowCount + 1, WINDOW_SIZE);

        latest = new Snapshot(
                frameCount,
                faces,
                totalMs,
                average(detectionWindow),
                average(recognitionWindow),
                average(totalWindow),
                totalMs > 0 ? 1000.0 / totalMs : 0);

        writeCsv(faces, detectionMs, recognitionMs, totalMs);
        if (frameCount % LOG_INTERVAL == 0) {
            Log.i(TAG, latest.toLogString());
            flushCsv();
        }
        return latest;
    }

    synchronized Snapshot latest() {
        return latest;
    }

    synchronized void close() {
        if (csvWriter == null) {
            return;
        }
        try {
            csvWriter.flush();
            csvWriter.close();
        } catch (IOException error) {
            Log.w(TAG, "Could not close performance CSV", error);
        }
    }

    private double average(double[] values) {
        if (windowCount == 0) {
            return 0;
        }
        double sum = 0;
        for (int i = 0; i < windowCount; i++) {
            sum += values[i];
        }
        return sum / windowCount;
    }

    private void writeCsv(int faces, double detectionMs, double recognitionMs, double totalMs) {
        if (csvWriter == null) {
            return;
        }
        try {
            csvWriter.write(String.format(Locale.US, "%d,%d,%d,%.3f,%.3f,%.3f\n",
                    frameCount,
                    System.currentTimeMillis(),
                    faces,
                    detectionMs,
                    recognitionMs,
                    totalMs));
        } catch (IOException error) {
            Log.w(TAG, "Could not append performance sample", error);
        }
    }

    private void flushCsv() {
        if (csvWriter == null) {
            return;
        }
        try {
            csvWriter.flush();
        } catch (IOException error) {
            Log.w(TAG, "Could not flush performance CSV", error);
        }
    }

    static final class Snapshot {
        final long frame;
        final int faces;
        final double latestTotalMs;
        final double averageDetectionMs;
        final double averageRecognitionMs;
        final double averageTotalMs;
        final double latestFps;

        Snapshot(long frame, int faces, double latestTotalMs, double averageDetectionMs,
                 double averageRecognitionMs, double averageTotalMs, double latestFps) {
            this.frame = frame;
            this.faces = faces;
            this.latestTotalMs = latestTotalMs;
            this.averageDetectionMs = averageDetectionMs;
            this.averageRecognitionMs = averageRecognitionMs;
            this.averageTotalMs = averageTotalMs;
            this.latestFps = latestFps;
        }

        String toStatusString() {
            return String.format(Locale.US,
                    "Latency avg(30): detect %.1f ms  recognize %.1f ms  total %.1f ms",
                    averageDetectionMs,
                    averageRecognitionMs,
                    averageTotalMs);
        }

        String toLogString() {
            return String.format(Locale.US,
                    "frame=%d faces=%d latest=%.1fms fps=%.1f avgDetect=%.1fms avgRecognize=%.1fms avgTotal=%.1fms",
                    frame,
                    faces,
                    latestTotalMs,
                    latestFps,
                    averageDetectionMs,
                    averageRecognitionMs,
                    averageTotalMs);
        }
    }
}
