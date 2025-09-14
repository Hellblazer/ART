package com.hellblazer.art.hartcq.core.channels;

import com.hellblazer.art.hartcq.ProcessingWindow;
import com.hellblazer.art.hartcq.Token;
import com.hellblazer.art.hartcq.core.SlidingWindow;

/**
 * CRITICAL: Positional encoding channel using sinusoidal encoding.
 * This is essential for maintaining positional information in the HART-CQ system.
 * 
 * Uses the formula from "Attention is All You Need":
 * PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
 * PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 */
public class PositionalChannel implements Channel {
    
    private static final int ENCODING_DIM = 64; // Dimension of positional encoding
    private static final double BASE = 10000.0;
    
    private final float[][] positionEncodings;
    
    public PositionalChannel() {
        // Pre-compute positional encodings for efficiency
        this.positionEncodings = precomputeEncodings();
    }
    
    /**
     * Pre-computes sinusoidal positional encodings for all positions.
     */
    private float[][] precomputeEncodings() {
        var encodings = new float[SlidingWindow.WINDOW_SIZE][ENCODING_DIM];
        
        for (int pos = 0; pos < SlidingWindow.WINDOW_SIZE; pos++) {
            for (int i = 0; i < ENCODING_DIM / 2; i++) {
                double angle = pos / Math.pow(BASE, (2.0 * i) / ENCODING_DIM);
                encodings[pos][2 * i] = (float) Math.sin(angle);
                encodings[pos][2 * i + 1] = (float) Math.cos(angle);
            }
        }
        
        return encodings;
    }
    
    @Override
    public ChannelOutput process(ProcessingWindow window) {
        var startTime = System.nanoTime();
        var tokens = window.getTokens();
        var output = new float[ENCODING_DIM];
        
        // Combine positional encodings with token presence
        for (int i = 0; i < tokens.size() && i < SlidingWindow.WINDOW_SIZE; i++) {
            var token = tokens.get(i);
            if (token != null) {
                // Add positional encoding weighted by token presence
                for (int j = 0; j < ENCODING_DIM; j++) {
                    output[j] += positionEncodings[i][j];
                }
            }
        }
        
        // Normalize the output
        normalize(output);
        
        var processingTime = System.nanoTime() - startTime;
        return ChannelOutput.valid(output, getChannelType(), processingTime);
    }

    @Override
    public float[] processWindow(Token[] tokens) {
        var output = new float[ENCODING_DIM];
        
        // Combine positional encodings with token presence
        for (int i = 0; i < tokens.length && i < SlidingWindow.WINDOW_SIZE; i++) {
            if (tokens[i] != null) {
                // Add positional encoding weighted by token presence
                for (int j = 0; j < ENCODING_DIM; j++) {
                    output[j] += positionEncodings[i][j];
                }
            }
        }
        
        // Normalize the output
        normalize(output);
        
        return output;
    }
    
    /**
     * Normalizes the vector to unit length.
     */
    private void normalize(float[] vector) {
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
    
    @Override
    public int getOutputDimension() {
        return ENCODING_DIM;
    }
    
    @Override
    public String getName() {
        return "PositionalChannel";
    }
    
    @Override
    public ChannelType getChannelType() {
        return ChannelType.POSITIONAL;
    }
    
    @Override
    public void reset() {
        // No state to reset - encodings are pre-computed
    }
    
    @Override
    public boolean isDeterministic() {
        return true; // Positional encoding is deterministic
    }
    
    /**
     * Gets the raw positional encoding for a specific position.
     * Useful for debugging and visualization.
     */
    public float[] getPositionalEncoding(int position) {
        if (position < 0 || position >= SlidingWindow.WINDOW_SIZE) {
            throw new IllegalArgumentException("Position must be between 0 and " + (SlidingWindow.WINDOW_SIZE - 1));
        }
        return positionEncodings[position].clone();
    }
}