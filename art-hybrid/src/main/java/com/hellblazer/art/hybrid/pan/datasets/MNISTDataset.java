package com.hellblazer.art.hybrid.pan.datasets;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;

import java.io.*;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

/**
 * MNIST dataset loader for real data.
 */
public class MNISTDataset {

    private static final String MNIST_BASE_URL = "http://yann.lecun.com/exdb/mnist/";
    private static final String TRAIN_IMAGES = "train-images-idx3-ubyte.gz";
    private static final String TRAIN_LABELS = "train-labels-idx1-ubyte.gz";
    private static final String TEST_IMAGES = "t10k-images-idx3-ubyte.gz";
    private static final String TEST_LABELS = "t10k-labels-idx1-ubyte.gz";

    private static final Path CACHE_DIR = Paths.get(System.getProperty("java.io.tmpdir"), "mnist_cache");

    public record MNISTData(List<Pattern> images, List<Pattern> labels) {}

    /**
     * Load MNIST training data.
     */
    public static MNISTData loadTrainingData() throws IOException {
        return loadTrainingData(60000);
    }

    /**
     * Load subset of MNIST training data.
     */
    public static MNISTData loadTrainingData(int limit) throws IOException {
        ensureCacheDir();
        var imagesPath = downloadIfNeeded(TRAIN_IMAGES);
        var labelsPath = downloadIfNeeded(TRAIN_LABELS);
        return loadData(imagesPath, labelsPath, limit);
    }

    /**
     * Load MNIST test data.
     */
    public static MNISTData loadTestData() throws IOException {
        return loadTestData(10000);
    }

    /**
     * Load subset of MNIST test data.
     */
    public static MNISTData loadTestData(int limit) throws IOException {
        ensureCacheDir();
        var imagesPath = downloadIfNeeded(TEST_IMAGES);
        var labelsPath = downloadIfNeeded(TEST_LABELS);
        return loadData(imagesPath, labelsPath, limit);
    }

    /**
     * Load MNIST data from cached files.
     */
    private static MNISTData loadData(Path imagesPath, Path labelsPath, int limit) throws IOException {
        var images = readImages(imagesPath, limit);
        var labels = readLabels(labelsPath, limit);

        if (images.size() != labels.size()) {
            throw new IOException("Image and label counts don't match");
        }

        // Convert to Patterns
        List<Pattern> imagePatterns = new ArrayList<>();
        List<Pattern> labelPatterns = new ArrayList<>();

        for (int i = 0; i < images.size(); i++) {
            // Normalize pixel values to [0, 1]
            var pixels = images.get(i);
            double[] normalized = new double[pixels.length];
            for (int j = 0; j < pixels.length; j++) {
                normalized[j] = (pixels[j] & 0xFF) / 255.0;
            }
            imagePatterns.add(new DenseVector(normalized));

            // Create one-hot encoded label
            int label = labels.get(i) & 0xFF;
            double[] oneHot = new double[10];
            oneHot[label] = 1.0;
            labelPatterns.add(new DenseVector(oneHot));
        }

        return new MNISTData(imagePatterns, labelPatterns);
    }

    /**
     * Read MNIST images from file.
     */
    private static List<byte[]> readImages(Path path, int limit) throws IOException {
        List<byte[]> images = new ArrayList<>();

        try (var stream = new GZIPInputStream(new FileInputStream(path.toFile()))) {
            // Read header
            byte[] header = new byte[16];
            stream.read(header);
            ByteBuffer bb = ByteBuffer.wrap(header);

            int magic = bb.getInt();
            if (magic != 0x00000803) {
                throw new IOException("Invalid magic number for images: " + magic);
            }

            int numImages = bb.getInt();
            int numRows = bb.getInt();
            int numCols = bb.getInt();

            int imageSize = numRows * numCols;
            int count = Math.min(limit, numImages);

            // Read images
            for (int i = 0; i < count; i++) {
                byte[] image = new byte[imageSize];
                stream.read(image);
                images.add(image);
            }
        }

        return images;
    }

    /**
     * Read MNIST labels from file.
     */
    private static List<Byte> readLabels(Path path, int limit) throws IOException {
        List<Byte> labels = new ArrayList<>();

        try (var stream = new GZIPInputStream(new FileInputStream(path.toFile()))) {
            // Read header
            byte[] header = new byte[8];
            stream.read(header);
            ByteBuffer bb = ByteBuffer.wrap(header);

            int magic = bb.getInt();
            if (magic != 0x00000801) {
                throw new IOException("Invalid magic number for labels: " + magic);
            }

            int numLabels = bb.getInt();
            int count = Math.min(limit, numLabels);

            // Read labels
            for (int i = 0; i < count; i++) {
                labels.add((byte) stream.read());
            }
        }

        return labels;
    }

    /**
     * Download MNIST file if not cached.
     */
    private static Path downloadIfNeeded(String filename) throws IOException {
        var cachedPath = CACHE_DIR.resolve(filename);

        if (Files.exists(cachedPath)) {
            System.out.printf("Using cached MNIST file: %s\n", filename);
            return cachedPath;
        }

        System.out.printf("Downloading MNIST file: %s...\n", filename);
        var url = new URL(MNIST_BASE_URL + filename);

        try (var in = url.openStream();
             var out = Files.newOutputStream(cachedPath)) {

            byte[] buffer = new byte[8192];
            int bytesRead;
            long totalBytes = 0;

            while ((bytesRead = in.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
                totalBytes += bytesRead;
                if (totalBytes % (1024 * 1024) == 0) {
                    System.out.printf("  Downloaded %.1f MB...\n", totalBytes / (1024.0 * 1024.0));
                }
            }

            System.out.printf("  Downloaded %.1f MB total\n", totalBytes / (1024.0 * 1024.0));
        }

        return cachedPath;
    }

    /**
     * Ensure cache directory exists.
     */
    private static void ensureCacheDir() throws IOException {
        if (!Files.exists(CACHE_DIR)) {
            Files.createDirectories(CACHE_DIR);
        }
    }

    /**
     * Clear cached MNIST files.
     */
    public static void clearCache() throws IOException {
        if (Files.exists(CACHE_DIR)) {
            Files.list(CACHE_DIR)
                .filter(path -> path.toString().endsWith(".gz"))
                .forEach(path -> {
                    try {
                        Files.delete(path);
                    } catch (IOException e) {
                        System.err.println("Failed to delete: " + path);
                    }
                });
        }
    }
}