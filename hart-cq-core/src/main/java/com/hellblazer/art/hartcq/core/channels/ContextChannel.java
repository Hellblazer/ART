package com.hellblazer.art.hartcq.core.channels;

import com.hellblazer.art.hartcq.Token;
import com.hellblazer.art.hartcq.ProcessingWindow;
import com.hellblazer.art.hartcq.core.SlidingWindow;

import java.util.Deque;
import java.util.concurrent.ConcurrentLinkedDeque;

/**
 * Context channel that maintains historical information across windows.
 */
public class ContextChannel implements Channel {
    
    private static final int CONTEXT_DIM = 40;
    private static final int HISTORY_SIZE = 5; // Number of previous windows to remember
    
    private final Deque<float[]> windowHistory;
    private volatile float[] previousOutput;
    
    public ContextChannel() {
        this.windowHistory = new ConcurrentLinkedDeque<>();
        this.previousOutput = new float[CONTEXT_DIM];
    }
    
    @Override
    public float[] processWindow(Token[] tokens) {
        var output = new float[CONTEXT_DIM];
        
        // Extract current window features
        var currentFeatures = extractWindowFeatures(tokens);
        
        // Copy current features to first part of output
        System.arraycopy(currentFeatures, 0, output, 0, 20);
        
        // Add temporal context from history
        if (!windowHistory.isEmpty()) {
            int historyOffset = 20;
            int historyIndex = 0;
            
            for (float[] historicalFeatures : windowHistory) {
                if (historyIndex >= 3) break; // Use only 3 most recent windows
                
                // Add weighted historical features
                float weight = 1.0f / (historyIndex + 2); // Decay weight
                for (int i = 0; i < 5; i++) {
                    output[historyOffset + i] += historicalFeatures[i] * weight;
                }
                historyIndex++;
            }
        }
        
        // Add momentum features (change from previous output)
        if (previousOutput != null) {
            for (int i = 0; i < 10; i++) {
                output[30 + i] = output[i] - previousOutput[i];
            }
        }
        
        // Update history
        updateHistory(currentFeatures);
        
        // Store current output for next iteration
        previousOutput = output.clone();
        
        // Normalize
        normalize(output);
        
        return output;
    }
    
    private float[] extractWindowFeatures(Token[] tokens) {
        var features = new float[20];
        
        // Count token types
        int wordCount = 0;
        int punctCount = 0;
        int numberCount = 0;
        int capitalCount = 0;
        
        for (int i = 0; i < tokens.length && i < SlidingWindow.WINDOW_SIZE; i++) {
            if (tokens[i] == null) continue;
            
            switch (tokens[i].getType()) {
                case WORD -> {
                    wordCount++;
                    var text = tokens[i].getText();
                    if (text != null && !text.isEmpty() && Character.isUpperCase(text.charAt(0))) {
                        capitalCount++;
                    }
                }
                case PUNCTUATION -> punctCount++;
                case NUMBER -> numberCount++;
            }
        }
        
        // Store counts as features
        features[0] = (float) wordCount / SlidingWindow.WINDOW_SIZE;
        features[1] = (float) punctCount / SlidingWindow.WINDOW_SIZE;
        features[2] = (float) numberCount / SlidingWindow.WINDOW_SIZE;
        features[3] = (float) capitalCount / SlidingWindow.WINDOW_SIZE;
        
        // Add position-weighted features
        for (int i = 0; i < tokens.length && i < SlidingWindow.WINDOW_SIZE; i++) {
            if (tokens[i] == null) continue;
            
            float positionWeight = (float) (i + 1) / SlidingWindow.WINDOW_SIZE;
            
            if (tokens[i].getType() == Token.TokenType.WORD) {
                features[4] += positionWeight;
            }
            if (tokens[i].getType() == Token.TokenType.PUNCTUATION) {
                features[5] += positionWeight;
            }
        }
        
        // Add window boundary features
        if (tokens.length > 0 && tokens[0] != null) {
            features[6] = tokens[0].getType() == Token.TokenType.WORD ? 1.0f : 0.0f;
            features[7] = tokens[0].getType() == Token.TokenType.PUNCTUATION ? 1.0f : 0.0f;
        }

        int lastIdx = Math.min(tokens.length - 1, SlidingWindow.WINDOW_SIZE - 1);
        if (lastIdx >= 0 && lastIdx < tokens.length && tokens[lastIdx] != null) {
            features[8] = tokens[lastIdx].getType() == Token.TokenType.WORD ? 1.0f : 0.0f;
            features[9] = tokens[lastIdx].getType() == Token.TokenType.PUNCTUATION ? 1.0f : 0.0f;
        }
        
        return features;
    }
    
    private void updateHistory(float[] features) {
        // Add new feature
        windowHistory.addLast(features.clone());

        // Remove oldest if we exceed history size
        while (windowHistory.size() > HISTORY_SIZE) {
            windowHistory.pollFirst();
        }
    }
    
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
        return CONTEXT_DIM;
    }
    
    @Override
    public String getName() {
        return "ContextChannel";
    }
    
    @Override
    public void reset() {
        windowHistory.clear();
        previousOutput = new float[CONTEXT_DIM];
    }
    
    @Override
    public boolean isDeterministic() {
        return false; // Context channel maintains state, so not deterministic
    }

    @Override
    public ChannelType getChannelType() {
        return ChannelType.CONTEXT;
    }

    @Override
    public ChannelOutput process(ProcessingWindow window) {
        var startTime = System.nanoTime();

        // Convert ProcessingWindow to Token array for legacy compatibility
        var tokens = window.getTokens().toArray(new Token[0]);
        var features = processWindow(tokens);

        var processingTime = System.nanoTime() - startTime;
        return ChannelOutput.valid(features, getChannelType(), processingTime);
    }
}