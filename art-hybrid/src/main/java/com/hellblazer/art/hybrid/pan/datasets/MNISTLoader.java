package com.hellblazer.art.hybrid.pan.datasets;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;

import java.io.*;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

/**
 * MNIST dataset loader for PAN benchmarking.
 * Loads the standard MNIST handwritten digit dataset.
 */
public class MNISTLoader {

    private static final String TRAIN_IMAGES = "train-images-idx3-ubyte.gz";
    private static final String TRAIN_LABELS = "train-labels-idx1-ubyte.gz";
    private static final String TEST_IMAGES = "t10k-images-idx3-ubyte.gz";
    private static final String TEST_LABELS = "t10k-labels-idx1-ubyte.gz";

    private static final String BASE_URL = "http://yann.lecun.com/exdb/mnist/";
    private static final String MNIST_DIR = System.getProperty("user.home") + "/.art/datasets/mnist/";

    private static final int IMAGE_MAGIC = 2051;
    private static final int LABEL_MAGIC = 2049;
    private static final int IMAGE_SIZE = 28 * 28;

    public record MNISTData(
        List<Pattern> images,
        List<Pattern> labels,
        int numClasses
    ) {}

    /**
     * Load MNIST training data.
     */
    public static MNISTData loadTrainingData() throws IOException {
        ensureDataDownloaded();
        return loadData(TRAIN_IMAGES, TRAIN_LABELS);
    }

    /**
     * Load MNIST test data.
     */
    public static MNISTData loadTestData() throws IOException {
        ensureDataDownloaded();
        return loadData(TEST_IMAGES, TEST_LABELS);
    }

    /**
     * Load subset of training data.
     */
    public static MNISTData loadTrainingSubset(int size) throws IOException {
        var full = loadTrainingData();
        if (size >= full.images.size()) {
            return full;
        }

        return new MNISTData(
            full.images.subList(0, size),
            full.labels.subList(0, size),
            full.numClasses
        );
    }

    private static MNISTData loadData(String imageFile, String labelFile) throws IOException {
        var imagePath = Paths.get(MNIST_DIR, imageFile);
        var labelPath = Paths.get(MNIST_DIR, labelFile);

        var images = loadImages(imagePath);
        var labels = loadLabels(labelPath);

        if (images.size() != labels.size()) {
            throw new IOException("Image and label counts don't match");
        }

        return new MNISTData(images, labels, 10);
    }

    private static List<Pattern> loadImages(Path path) throws IOException {
        try (var gzis = new GZIPInputStream(Files.newInputStream(path));
             var dis = new DataInputStream(gzis)) {

            int magic = dis.readInt();
            if (magic != IMAGE_MAGIC) {
                throw new IOException("Invalid image file magic: " + magic);
            }

            int numImages = dis.readInt();
            int rows = dis.readInt();
            int cols = dis.readInt();

            if (rows != 28 || cols != 28) {
                throw new IOException("Unexpected image dimensions: " + rows + "x" + cols);
            }

            List<Pattern> images = new ArrayList<>(numImages);
            byte[] buffer = new byte[IMAGE_SIZE];

            for (int i = 0; i < numImages; i++) {
                dis.readFully(buffer);
                double[] pixels = new double[IMAGE_SIZE];
                for (int j = 0; j < IMAGE_SIZE; j++) {
                    pixels[j] = (buffer[j] & 0xFF) / 255.0;
                }
                images.add(new DenseVector(pixels));
            }

            return images;
        }
    }

    private static List<Pattern> loadLabels(Path path) throws IOException {
        try (var gzis = new GZIPInputStream(Files.newInputStream(path));
             var dis = new DataInputStream(gzis)) {

            int magic = dis.readInt();
            if (magic != LABEL_MAGIC) {
                throw new IOException("Invalid label file magic: " + magic);
            }

            int numLabels = dis.readInt();
            List<Pattern> labels = new ArrayList<>(numLabels);

            for (int i = 0; i < numLabels; i++) {
                int label = dis.readByte() & 0xFF;
                double[] oneHot = new double[10];
                oneHot[label] = 1.0;
                labels.add(new DenseVector(oneHot));
            }

            return labels;
        }
    }

    private static void ensureDataDownloaded() throws IOException {
        var dir = Paths.get(MNIST_DIR);
        if (!Files.exists(dir)) {
            Files.createDirectories(dir);
        }

        String[] files = {TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS};
        for (String file : files) {
            var path = dir.resolve(file);
            if (!Files.exists(path)) {
                downloadFile(BASE_URL + file, path);
            }
        }
    }

    private static void downloadFile(String urlString, Path destination) throws IOException {
        System.out.println("Downloading " + urlString + " to " + destination);

        var url = new URL(urlString);
        try (var in = url.openStream();
             var out = Files.newOutputStream(destination)) {

            byte[] buffer = new byte[8192];
            int bytesRead;
            long totalBytes = 0;

            while ((bytesRead = in.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
                totalBytes += bytesRead;
                if (totalBytes % (1024 * 1024) == 0) {
                    System.out.print(".");
                }
            }
            System.out.println(" Done (" + totalBytes + " bytes)");
        }
    }

    /**
     * Convert label Pattern to class index.
     */
    public static int getClassIndex(Pattern label) {
        for (int i = 0; i < label.dimension(); i++) {
            if (label.get(i) > 0.5) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Create one-hot label for class index.
     */
    public static Pattern createOneHotLabel(int classIndex, int numClasses) {
        double[] oneHot = new double[numClasses];
        oneHot[classIndex] = 1.0;
        return new DenseVector(oneHot);
    }
}