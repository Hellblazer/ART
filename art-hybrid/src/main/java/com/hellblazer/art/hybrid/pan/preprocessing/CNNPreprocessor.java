package com.hellblazer.art.hybrid.pan.preprocessing;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.hybrid.pan.parameters.CNNConfig;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

/**
 * CNN feature extractor for PAN.
 * Handles the float/double conversion boundary between CNN processing and ART framework.
 */
public class CNNPreprocessor implements AutoCloseable {

    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> DOUBLE_SPECIES = DoubleVector.SPECIES_PREFERRED;

    private final CNNConfig config;
    private final boolean usePretrained;
    private final ConvolutionalLayer[] layers;
    private volatile boolean closed = false;

    // Expose weights for pretraining
    private float[] conv1Weights;
    private float[] conv2Weights;

    public CNNPreprocessor(CNNConfig config, boolean usePretrained) {
        this.config = config;
        this.usePretrained = usePretrained;
        this.layers = buildLayers(config);

        if (!usePretrained) {
            initializeWeights();
        }
    }

    /**
     * Extract features from input pattern using CNN.
     */
    public Pattern extractFeatures(Pattern input) {
        if (closed) {
            throw new IllegalStateException("CNNPreprocessor is closed");
        }

        // Convert Pattern (double[]) to float[] for CNN processing
        float[] imageData = patternToFloatArray(input);

        // Reshape to 2D for convolution (assuming square images)
        int size = (int) Math.sqrt(imageData.length);
        float[][][] image = new float[1][size][size];
        int idx = 0;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (idx < imageData.length) {
                    image[0][i][j] = imageData[idx++];
                }
            }
        }

        // Forward pass through CNN layers
        float[][][] current = image;
        for (var layer : layers) {
            current = layer.forward(current);
        }

        // Flatten and convert back to Pattern
        float[] features = flatten(current);
        double[] doubleFeatures = floatToDoubleArray(features);

        return new DenseVector(doubleFeatures);
    }

    /**
     * Convert Pattern to float array for CNN processing.
     */
    private float[] patternToFloatArray(Pattern pattern) {
        float[] result = new float[pattern.dimension()];
        int i = 0;

        // Vectorized conversion
        for (; i < DOUBLE_SPECIES.loopBound(pattern.dimension()); i += DOUBLE_SPECIES.length()) {
            double[] temp = new double[DOUBLE_SPECIES.length()];
            for (int j = 0; j < DOUBLE_SPECIES.length() && i + j < pattern.dimension(); j++) {
                temp[j] = pattern.get(i + j);
            }

            for (int j = 0; j < DOUBLE_SPECIES.length() && i + j < result.length; j++) {
                result[i + j] = (float) temp[j];
            }
        }

        // Scalar tail
        for (; i < pattern.dimension(); i++) {
            result[i] = (float) pattern.get(i);
        }

        return result;
    }

    /**
     * Convert float array to double array for Pattern creation.
     */
    private double[] floatToDoubleArray(float[] input) {
        double[] result = new double[input.length];
        int i = 0;

        // Vectorized conversion
        for (; i < FLOAT_SPECIES.loopBound(input.length); i += FLOAT_SPECIES.length()) {
            var v = FloatVector.fromArray(FLOAT_SPECIES, input, i);
            for (int j = 0; j < FLOAT_SPECIES.length() && i + j < result.length; j++) {
                result[i + j] = v.lane(j);
            }
        }

        // Scalar tail
        for (; i < input.length; i++) {
            result[i] = input[i];
        }

        return result;
    }

    /**
     * Flatten 3D tensor to 1D array.
     */
    private float[] flatten(float[][][] tensor) {
        int totalSize = tensor.length * tensor[0].length * tensor[0][0].length;
        float[] flat = new float[totalSize];
        int idx = 0;

        for (float[][] channel : tensor) {
            for (float[] row : channel) {
                for (float val : row) {
                    flat[idx++] = val;
                }
            }
        }

        // Ensure output size matches config
        if (flat.length != config.outputFeatures()) {
            flat = Arrays.copyOf(flat, config.outputFeatures());
        }

        return flat;
    }

    /**
     * Build CNN layers based on configuration.
     */
    private ConvolutionalLayer[] buildLayers(CNNConfig config) {
        var layerList = new ConvolutionalLayer[config.numLayers()];

        int inputChannels = 1;
        for (int i = 0; i < config.numLayers(); i++) {
            int outputChannels = config.filterSizes()[i];
            layerList[i] = new ConvolutionalLayer(
                inputChannels, outputChannels,
                3, 1, 1  // 3x3 kernels, stride 1, padding 1
            );
            inputChannels = outputChannels;
        }

        return layerList;
    }

    /**
     * Initialize weights for training from scratch.
     */
    private void initializeWeights() {
        for (var layer : layers) {
            layer.initializeWeights();
        }
    }

    /**
     * Estimate memory usage in bytes.
     */
    public long estimateMemoryUsage() {
        long total = 0;
        for (var layer : layers) {
            total += layer.estimateMemoryUsage();
        }
        return total;
    }

    @Override
    public void close() {
        closed = true;
        // Release any resources
        Arrays.fill(layers, null);
    }

    /**
     * Simple convolutional layer implementation.
     */
    private static class ConvolutionalLayer {
        private final int inputChannels;
        private final int outputChannels;
        private final int kernelSize;
        private final int stride;
        private final int padding;
        private float[][][][] weights;
        private float[] bias;

        ConvolutionalLayer(int inputChannels, int outputChannels,
                          int kernelSize, int stride, int padding) {
            this.inputChannels = inputChannels;
            this.outputChannels = outputChannels;
            this.kernelSize = kernelSize;
            this.stride = stride;
            this.padding = padding;
            this.weights = new float[outputChannels][inputChannels][kernelSize][kernelSize];
            this.bias = new float[outputChannels];
        }

        void initializeWeights() {
            var rand = ThreadLocalRandom.current();
            double scale = Math.sqrt(2.0 / (inputChannels * kernelSize * kernelSize));

            for (int o = 0; o < outputChannels; o++) {
                for (int i = 0; i < inputChannels; i++) {
                    for (int k1 = 0; k1 < kernelSize; k1++) {
                        for (int k2 = 0; k2 < kernelSize; k2++) {
                            weights[o][i][k1][k2] = (float) (rand.nextGaussian() * scale);
                        }
                    }
                }
                bias[o] = 0.01f;
            }
        }

        float[][][] forward(float[][][] input) {
            int inputHeight = input[0].length;
            int inputWidth = input[0][0].length;
            int outputHeight = (inputHeight + 2 * padding - kernelSize) / stride + 1;
            int outputWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;

            float[][][] output = new float[outputChannels][outputHeight][outputWidth];

            // Vectorized convolution with SIMD
            for (int oc = 0; oc < outputChannels; oc++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        FloatVector sumVec = FloatVector.zero(FLOAT_SPECIES);
                        float scalarSum = bias[oc];

                        // Vectorize over kernel elements
                        for (int ic = 0; ic < inputChannels && ic < input.length; ic++) {
                            // Flatten kernel for this input channel
                            float[] kernelFlat = new float[kernelSize * kernelSize];
                            float[] inputPatch = new float[kernelSize * kernelSize];

                            for (int kh = 0; kh < kernelSize; kh++) {
                                for (int kw = 0; kw < kernelSize; kw++) {
                                    int idx = kh * kernelSize + kw;
                                    kernelFlat[idx] = weights[oc][ic][kh][kw];

                                    int ih = oh * stride - padding + kh;
                                    int iw = ow * stride - padding + kw;

                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                        inputPatch[idx] = input[ic][ih][iw];
                                    } else {
                                        inputPatch[idx] = 0;
                                    }
                                }
                            }

                            // SIMD dot product
                            int i = 0;
                            for (; i < FLOAT_SPECIES.loopBound(kernelFlat.length); i += FLOAT_SPECIES.length()) {
                                var kVec = FloatVector.fromArray(FLOAT_SPECIES, kernelFlat, i);
                                var iVec = FloatVector.fromArray(FLOAT_SPECIES, inputPatch, i);
                                sumVec = sumVec.add(kVec.mul(iVec));
                            }

                            // Scalar remainder
                            for (; i < kernelFlat.length; i++) {
                                scalarSum += kernelFlat[i] * inputPatch[i];
                            }
                        }

                        float totalSum = sumVec.reduceLanes(VectorOperators.ADD) + scalarSum;
                        // ReLU activation
                        output[oc][oh][ow] = Math.max(0, totalSum);
                    }
                }
            }

            // Simple pooling (2x2 max pooling)
            if (outputHeight > 4 && outputWidth > 4) {
                return maxPool2x2(output);
            }

            return output;
        }

        private float[][][] maxPool2x2(float[][][] input) {
            int channels = input.length;
            int height = input[0].length / 2;
            int width = input[0][0].length / 2;
            float[][][] output = new float[channels][height][width];

            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        float max = input[c][h * 2][w * 2];
                        max = Math.max(max, input[c][h * 2][w * 2 + 1]);
                        max = Math.max(max, input[c][h * 2 + 1][w * 2]);
                        max = Math.max(max, input[c][h * 2 + 1][w * 2 + 1]);
                        output[c][h][w] = max;
                    }
                }
            }

            return output;
        }

        long estimateMemoryUsage() {
            long weightSize = (long) outputChannels * inputChannels * kernelSize * kernelSize * 4;
            long biasSize = outputChannels * 4L;
            return weightSize + biasSize;
        }
    }

    // Getter methods for pretraining

    public float[] getConv1Weights() {
        if (layers.length > 0 && layers[0] instanceof ConvolutionalLayer conv) {
            // Flatten 4D weights to 1D for transfer
            return flatten4D(conv.weights);
        }
        return new float[0];
    }

    public float[] getConv2Weights() {
        if (layers.length > 1 && layers[1] instanceof ConvolutionalLayer conv) {
            // Flatten 4D weights to 1D for transfer
            return flatten4D(conv.weights);
        }
        return new float[0];
    }

    public void setConv1Weights(float[] weights) {
        if (layers.length > 0 && layers[0] instanceof ConvolutionalLayer conv) {
            // Unflatten 1D weights to 4D
            unflatten4D(weights, conv.weights);
        }
    }

    public void setConv2Weights(float[] weights) {
        if (layers.length > 1 && layers[1] instanceof ConvolutionalLayer conv) {
            // Unflatten 1D weights to 4D
            unflatten4D(weights, conv.weights);
        }
    }

    private float[] flatten4D(float[][][][] weights4D) {
        int totalSize = weights4D.length * weights4D[0].length *
                       weights4D[0][0].length * weights4D[0][0][0].length;
        float[] flat = new float[totalSize];
        int idx = 0;
        for (var w1 : weights4D) {
            for (var w2 : w1) {
                for (var w3 : w2) {
                    for (float w4 : w3) {
                        flat[idx++] = w4;
                    }
                }
            }
        }
        return flat;
    }

    private void unflatten4D(float[] flat, float[][][][] weights4D) {
        int idx = 0;
        for (int i = 0; i < weights4D.length && idx < flat.length; i++) {
            for (int j = 0; j < weights4D[i].length && idx < flat.length; j++) {
                for (int k = 0; k < weights4D[i][j].length && idx < flat.length; k++) {
                    for (int l = 0; l < weights4D[i][j][k].length && idx < flat.length; l++) {
                        weights4D[i][j][k][l] = flat[idx++];
                    }
                }
            }
        }
    }
}