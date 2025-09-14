package com.hellblazer.art.hartcq.core.channels;

/**
 * Represents the output from a channel processing operation.
 * Contains the feature vector and metadata about the processing.
 */
public record ChannelOutput(
    float[] features,
    ChannelType channelType,
    long processingTime,
    boolean isValid
) {
    
    public ChannelOutput {
        if (features == null) {
            throw new IllegalArgumentException("Features cannot be null");
        }
        if (channelType == null) {
            throw new IllegalArgumentException("Channel type cannot be null");
        }
    }
    
    /**
     * Creates a valid channel output.
     */
    public static ChannelOutput valid(float[] features, ChannelType channelType, long processingTime) {
        return new ChannelOutput(features, channelType, processingTime, true);
    }
    
    /**
     * Creates an invalid channel output (for error conditions).
     */
    public static ChannelOutput invalid(ChannelType channelType) {
        return new ChannelOutput(new float[0], channelType, 0, false);
    }
    
    /**
     * Gets the dimensionality of the feature vector.
     */
    public int getDimension() {
        return features.length;
    }
    
    /**
     * Creates a copy of this output with normalized features.
     */
    public ChannelOutput normalized() {
        var normalizedFeatures = features.clone();
        normalizeInPlace(normalizedFeatures);
        return new ChannelOutput(normalizedFeatures, channelType, processingTime, isValid);
    }
    
    private static void normalizeInPlace(float[] vector) {
        float sum = 0;
        for (float v : vector) {
            sum += v * v;
        }
        
        if (sum > 0) {
            float norm = (float) Math.sqrt(sum);
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
    }
}