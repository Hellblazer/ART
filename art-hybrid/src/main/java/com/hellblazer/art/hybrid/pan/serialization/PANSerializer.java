package com.hellblazer.art.hybrid.pan.serialization;

import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.hybrid.pan.parameters.CNNConfig;
import com.hellblazer.art.hybrid.pan.parameters.PANParameters;
import com.hellblazer.art.hybrid.pan.weight.BPARTWeight;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Serialization support for PAN models.
 * Saves and loads trained PAN models in a compact binary format.
 */
public class PANSerializer {

    private static final int VERSION = 1;
    private static final String MAGIC = "PAN_MODEL";

    /**
     * Save PAN model state to file.
     */
    public static void saveModel(Path path, SavedPANModel model, boolean compress) throws IOException {
        if (compress) {
            // For compressed files, write header and data separately
            try (OutputStream out = Files.newOutputStream(path)) {
                // Write uncompressed header
                DataOutputStream headerDos = new DataOutputStream(out);
                headerDos.writeUTF(MAGIC);
                headerDos.writeInt(VERSION);
                headerDos.writeBoolean(true);
                headerDos.writeLong(System.currentTimeMillis());
                headerDos.flush();

                // Write compressed data
                try (GZIPOutputStream gzOut = new GZIPOutputStream(out);
                     DataOutputStream gzDos = new DataOutputStream(gzOut)) {
                    writeModelData(gzDos, model);
                }
            }
        } else {
            // For uncompressed, write everything normally
            try (OutputStream out = Files.newOutputStream(path);
                 DataOutputStream dos = new DataOutputStream(out)) {
                dos.writeUTF(MAGIC);
                dos.writeInt(VERSION);
                dos.writeBoolean(false);
                dos.writeLong(System.currentTimeMillis());
                writeModelData(dos, model);
            }
        }
    }

    private static void writeModelData(DataOutputStream dos, SavedPANModel model) throws IOException {
        // Write parameters
        writeParameters(dos, model.parameters());

        // Write categories
        dos.writeInt(model.categories().size());
        for (var weight : model.categories()) {
            if (weight instanceof BPARTWeight bpart) {
                writeBPARTWeight(dos, bpart);
            } else {
                throw new IOException("Unsupported weight type: " + weight.getClass());
            }
        }

        // Write category labels mapping
        dos.writeInt(model.categoryToLabel().size());
        for (var entry : model.categoryToLabel().entrySet()) {
            dos.writeInt(entry.getKey());
            dos.writeInt(entry.getValue());
        }

        // Write performance stats
        dos.writeLong(model.totalSamples());
        dos.writeLong(model.correctPredictions());
        dos.writeDouble(model.averageLoss());
        dos.writeLong(model.trainingTimeMs());

        // Write CNN weights
        writeFloatArray(dos, model.cnnConv1Weights());
        writeFloatArray(dos, model.cnnConv2Weights());
    }

    /**
     * Load PAN model state from file.
     */
    public static SavedPANModel loadModel(Path path) throws IOException {
        try (InputStream in = Files.newInputStream(path)) {
            DataInputStream dis = new DataInputStream(in);

            // Read and verify header (always uncompressed)
            String magic = dis.readUTF();
            if (!MAGIC.equals(magic)) {
                throw new IOException("Invalid model file: wrong magic");
            }

            int version = dis.readInt();
            if (version != VERSION) {
                throw new IOException("Unsupported model version: " + version);
            }

            boolean compressed = dis.readBoolean();
            long timestamp = dis.readLong();

            // Read the rest (possibly compressed)
            if (compressed) {
                // The stream position is now after the header
                // Create GZIP stream from current position
                GZIPInputStream gzis = new GZIPInputStream(in);
                DataInputStream gzDis = new DataInputStream(gzis);
                return readModelData(gzDis, timestamp);
            } else {
                // Continue reading from same stream
                return readModelData(dis, timestamp);
            }
        }
    }

    private static SavedPANModel readModelData(DataInputStream dis, long timestamp) throws IOException {
        // Read parameters
        PANParameters parameters = readParameters(dis);

        // Read categories
        int categoryCount = dis.readInt();
        List<WeightVector> categories = new ArrayList<>(categoryCount);
        for (int i = 0; i < categoryCount; i++) {
            categories.add(readBPARTWeight(dis));
        }

        // Read category labels
        int labelCount = dis.readInt();
        Map<Integer, Integer> categoryToLabel = new HashMap<>();
        for (int i = 0; i < labelCount; i++) {
            int category = dis.readInt();
            int label = dis.readInt();
            categoryToLabel.put(category, label);
        }

        // Read performance stats
        long totalSamples = dis.readLong();
        long correctPredictions = dis.readLong();
        double averageLoss = dis.readDouble();
        long trainingTimeMs = dis.readLong();

        // Read CNN weights
        float[] conv1Weights = readFloatArray(dis);
        float[] conv2Weights = readFloatArray(dis);

        return new SavedPANModel(
            parameters,
            categories,
            categoryToLabel,
            totalSamples,
            correctPredictions,
            averageLoss,
            trainingTimeMs,
            timestamp,
            conv1Weights,
            conv2Weights
        );
    }

    /**
     * Saved PAN model representation.
     */
    public record SavedPANModel(
        PANParameters parameters,
        List<WeightVector> categories,
        Map<Integer, Integer> categoryToLabel,
        long totalSamples,
        long correctPredictions,
        double averageLoss,
        long trainingTimeMs,
        long timestamp,
        float[] cnnConv1Weights,  // Added CNN weights
        float[] cnnConv2Weights   // Added CNN weights
    ) {
        /**
         * Get accuracy percentage.
         */
        public double getAccuracy() {
            return totalSamples > 0 ? (double) correctPredictions / totalSamples * 100.0 : 0.0;
        }

        /**
         * Get model summary.
         */
        public String getSummary() {
            return String.format(
                "PAN Model: %d categories, %.2f%% accuracy, %d samples, %.2f hours training",
                categories.size(),
                getAccuracy(),
                totalSamples,
                trainingTimeMs / 3600000.0
            );
        }
    }

    // Private helper methods

    private static void writeParameters(DataOutputStream dos, PANParameters params) throws IOException {
        dos.writeDouble(params.vigilance());
        dos.writeInt(params.maxCategories());

        // Write CNN config
        dos.writeInt(params.cnnConfig().inputSize());
        dos.writeInt(params.cnnConfig().outputFeatures());
        dos.writeUTF(params.cnnConfig().architecture());
        dos.writeInt(params.cnnConfig().numLayers());

        int[] filterSizes = params.cnnConfig().filterSizes();
        dos.writeInt(filterSizes.length);
        for (int size : filterSizes) {
            dos.writeInt(size);
        }

        dos.writeBoolean(params.enableCNNPretraining());
        dos.writeDouble(params.learningRate());
        dos.writeDouble(params.momentum());
        dos.writeDouble(params.weightDecay());
        dos.writeBoolean(params.allowNegativeWeights());
        dos.writeInt(params.hiddenUnits());
        dos.writeDouble(params.stmDecayRate());
        dos.writeDouble(params.ltmConsolidationThreshold());
        dos.writeInt(params.replayBufferSize());
        dos.writeInt(params.replayBatchSize());
        dos.writeDouble(params.replayFrequency());
        dos.writeDouble(params.biasFactor());
    }

    private static PANParameters readParameters(DataInputStream dis) throws IOException {
        double vigilance = dis.readDouble();
        int maxCategories = dis.readInt();

        // Read CNN config
        int inputSize = dis.readInt();
        int outputFeatures = dis.readInt();
        String architecture = dis.readUTF();
        int numLayers = dis.readInt();

        int filterSizesLength = dis.readInt();
        int[] filterSizes = new int[filterSizesLength];
        for (int i = 0; i < filterSizesLength; i++) {
            filterSizes[i] = dis.readInt();
        }

        CNNConfig cnnConfig = new CNNConfig(inputSize, outputFeatures, architecture, numLayers, filterSizes);

        boolean enableCNNPretraining = dis.readBoolean();
        double learningRate = dis.readDouble();
        double momentum = dis.readDouble();
        double weightDecay = dis.readDouble();
        boolean allowNegativeWeights = dis.readBoolean();
        int hiddenUnits = dis.readInt();
        double stmDecayRate = dis.readDouble();
        double ltmConsolidationThreshold = dis.readDouble();
        int replayBufferSize = dis.readInt();
        int replayBatchSize = dis.readInt();
        double replayFrequency = dis.readDouble();
        double biasFactor = dis.readDouble();

        // Default parameters for backward compatibility
        return new PANParameters(
            vigilance, maxCategories, cnnConfig, enableCNNPretraining,
            learningRate, momentum, weightDecay, allowNegativeWeights,
            hiddenUnits, stmDecayRate, ltmConsolidationThreshold,
            replayBufferSize, replayBatchSize, replayFrequency, biasFactor,
            false, 0.0, 1.0  // Default normalization settings for backward compatibility
        );
    }

    private static void writeBPARTWeight(DataOutputStream dos, BPARTWeight weight) throws IOException {
        // Write arrays
        writeDoubleArray(dos, weight.forwardWeights());
        writeDoubleArray(dos, weight.backwardWeights());
        writeDoubleArray(dos, weight.hiddenBias());
        dos.writeDouble(weight.outputBias());
        writeDoubleArray(dos, weight.lastHiddenState());
        dos.writeDouble(weight.lastOutput());
        dos.writeDouble(weight.lastError());
        dos.writeLong(weight.updateCount());
    }

    private static BPARTWeight readBPARTWeight(DataInputStream dis) throws IOException {
        double[] forwardWeights = readDoubleArray(dis);
        double[] backwardWeights = readDoubleArray(dis);
        double[] hiddenBias = readDoubleArray(dis);
        double outputBias = dis.readDouble();
        double[] lastHiddenState = readDoubleArray(dis);
        double lastOutput = dis.readDouble();
        double lastError = dis.readDouble();
        long updateCount = dis.readLong();

        return new BPARTWeight(
            forwardWeights, backwardWeights, hiddenBias, outputBias,
            lastHiddenState, lastOutput, lastError, updateCount
        );
    }

    private static void writeDoubleArray(DataOutputStream dos, double[] array) throws IOException {
        dos.writeInt(array.length);
        for (double value : array) {
            dos.writeDouble(value);
        }
    }

    private static double[] readDoubleArray(DataInputStream dis) throws IOException {
        int length = dis.readInt();
        double[] array = new double[length];
        for (int i = 0; i < length; i++) {
            array[i] = dis.readDouble();
        }
        return array;
    }

    private static void writeFloatArray(DataOutputStream dos, float[] array) throws IOException {
        if (array == null) {
            dos.writeInt(0);
        } else {
            dos.writeInt(array.length);
            for (float value : array) {
                dos.writeFloat(value);
            }
        }
    }

    private static float[] readFloatArray(DataInputStream dis) throws IOException {
        int length = dis.readInt();
        if (length == 0) {
            return new float[0];
        }
        float[] array = new float[length];
        for (int i = 0; i < length; i++) {
            array[i] = dis.readFloat();
        }
        return array;
    }
}