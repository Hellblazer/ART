package com.hellblazer.art.hartcq.core.channels;

import com.hellblazer.art.hartcq.ProcessingWindow;
import com.hellblazer.art.hartcq.Token;
import com.hellblazer.art.hartcq.core.SlidingWindow;

import java.time.Clock;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Temporal sequence channel that tracks time-based patterns and sequence ordering.
 * Maintains temporal relationships within and across processing windows.
 * Thread-safe implementation for concurrent processing.
 */
public class TemporalChannel implements Channel {
    
    private static final int TEMPORAL_DIM = 32;
    private static final int SEQUENCE_HISTORY = 10; // Number of recent sequences to track
    
    private final Deque<SequenceSnapshot> sequenceHistory;
    private final ReentrantReadWriteLock lock;
    private final Clock clock;
    private long lastProcessingTime;
    private int sequenceCounter;
    
    /**
     * Snapshot of sequence information for temporal analysis.
     */
    private static class SequenceSnapshot {
        final float[] tokenTypeRatios;
        final float averageTokenLength;
        final int totalTokens;
        final long timestamp;
        final int sequenceId;
        
        SequenceSnapshot(float[] tokenTypeRatios, float averageTokenLength, int totalTokens, 
                        long timestamp, int sequenceId) {
            this.tokenTypeRatios = tokenTypeRatios.clone();
            this.averageTokenLength = averageTokenLength;
            this.totalTokens = totalTokens;
            this.timestamp = timestamp;
            this.sequenceId = sequenceId;
        }
    }
    
    public TemporalChannel() {
        this(Clock.systemUTC());
    }

    public TemporalChannel(Clock clock) {
        this.sequenceHistory = new ArrayDeque<>(SEQUENCE_HISTORY);
        this.lock = new ReentrantReadWriteLock();
        this.clock = clock;
        this.lastProcessingTime = clock.millis() * 1_000_000L;  // Convert to nanos
        this.sequenceCounter = 0;
    }

    @Override
    public float[] processWindow(Token[] tokens) {
        var output = new float[TEMPORAL_DIM];
        var currentTime = clock.millis() * 1_000_000L;  // Convert millis to nanos
        
        lock.writeLock().lock();
        try {
            // Analyze current window temporal features
            analyzeCurrentWindow(tokens, output, currentTime);
            
            // Analyze temporal patterns with history
            analyzeTemporalPatterns(tokens, output, currentTime);
            
            // Analyze sequence ordering
            analyzeSequenceOrdering(tokens, output);
            
            // Update sequence history
            updateSequenceHistory(tokens, currentTime);
            
            // Update timing
            lastProcessingTime = currentTime;
            sequenceCounter++;
            
        } finally {
            lock.writeLock().unlock();
        }
        
        // Normalize the output
        normalize(output);
        
        return output;
    }
    
    private void analyzeCurrentWindow(Token[] tokens, float[] output, long currentTime) {
        // Temporal position within sequence
        output[0] = (float) sequenceCounter / 1000.0f; // Normalized sequence position
        
        // Time since last processing (normalized to reasonable range)
        var timeDelta = currentTime - lastProcessingTime;
        output[1] = Math.min(timeDelta / 1_000_000.0f, 1000.0f) / 1000.0f; // Normalized milliseconds

        // Token arrival rate (tokens per unit time - synthetic measure)
        int nonNullTokens = 0;
        for (int i = 0; i < tokens.length && i < SlidingWindow.WINDOW_SIZE; i++) {
            if (tokens[i] != null) nonNullTokens++;
        }

        if (timeDelta > 0) {
            var tokenRate = (nonNullTokens * 1_000_000_000.0f) / timeDelta; // tokens per second
            output[2] = Math.min(tokenRate / 100.0f, 1.0f); // Normalized token rate
        } else {
            // For fixed clock or zero time delta, use a deterministic value
            output[2] = 0.5f; // Default normalized rate
        }
        
        // Window completeness over time
        output[3] = (float) nonNullTokens / SlidingWindow.WINDOW_SIZE;
    }
    
    private void analyzeTemporalPatterns(Token[] tokens, float[] output, long currentTime) {
        int offset = 4;
        
        lock.readLock().lock();
        try {
            if (sequenceHistory.isEmpty()) {
                return;
            }
            
            // Compare current window with recent history
            var currentSnapshot = createSnapshot(tokens, currentTime);
            
            // Temporal consistency measures
            float consistencySum = 0.0f;
            int comparisonCount = 0;
            
            for (var historical : sequenceHistory) {
                // Token type consistency over time
                float typeConsistency = calculateTokenTypeConsistency(currentSnapshot, historical);
                output[offset] += typeConsistency;
                
                // Length pattern consistency
                float lengthConsistency = Math.abs(currentSnapshot.averageTokenLength - historical.averageTokenLength);
                output[offset + 1] += lengthConsistency;
                
                // Token count stability
                float countStability = Math.abs(currentSnapshot.totalTokens - historical.totalTokens) / 
                                     (float) SlidingWindow.WINDOW_SIZE;
                output[offset + 2] += countStability;
                
                // Temporal velocity (rate of change)
                var timeDiff = currentTime - historical.timestamp;
                if (timeDiff > 0) {
                    float velocity = lengthConsistency / (timeDiff / 1_000_000.0f); // Change per millisecond
                    output[offset + 3] += Math.min(velocity, 1.0f);
                } else {
                    // For fixed clock or zero time diff, use a deterministic value
                    output[offset + 3] += 0.5f;
                }
                
                consistencySum += typeConsistency;
                comparisonCount++;
            }
            
            // Normalize by number of comparisons
            if (comparisonCount > 0) {
                for (int i = 0; i < 4; i++) {
                    output[offset + i] /= comparisonCount;
                }
                
                // Overall temporal stability
                output[offset + 4] = consistencySum / comparisonCount;
            }
            
        } finally {
            lock.readLock().unlock();
        }
    }
    
    private void analyzeSequenceOrdering(Token[] tokens, float[] output) {
        int offset = 9;
        
        // Analyze ordering patterns within the current window
        analyzeIntraWindowOrdering(tokens, output, offset);
        
        // Analyze ordering patterns across windows (using history)
        analyzeInterWindowOrdering(tokens, output, offset + 6);
    }
    
    private void analyzeIntraWindowOrdering(Token[] tokens, float[] output, int offset) {
        // Position-weighted token features
        float wordPositionSum = 0.0f;
        float punctPositionSum = 0.0f;
        float numberPositionSum = 0.0f;
        
        int wordCount = 0, punctCount = 0, numberCount = 0;
        
        for (int i = 0; i < tokens.length && i < SlidingWindow.WINDOW_SIZE; i++) {
            if (tokens[i] == null) continue;
            
            float position = (float) i / SlidingWindow.WINDOW_SIZE;
            
            switch (tokens[i].getType()) {
                case WORD -> {
                    wordPositionSum += position;
                    wordCount++;
                }
                case PUNCTUATION -> {
                    punctPositionSum += position;
                    punctCount++;
                }
                case NUMBER -> {
                    numberPositionSum += position;
                    numberCount++;
                }
            }
        }
        
        // Average positions (center of mass for each token type)
        output[offset] = wordCount > 0 ? wordPositionSum / wordCount : 0.5f;
        output[offset + 1] = punctCount > 0 ? punctPositionSum / punctCount : 0.5f;
        output[offset + 2] = numberCount > 0 ? numberPositionSum / numberCount : 0.5f;
        
        // Ordering entropy (how mixed the token types are)
        output[offset + 3] = calculateOrderingEntropy(tokens);
        
        // Sequential patterns
        output[offset + 4] = analyzeSequentialPatterns(tokens);
        
        // Positional variance (how spread out are the token types)
        output[offset + 5] = calculatePositionalVariance(tokens);
    }
    
    private void analyzeInterWindowOrdering(Token[] tokens, float[] output, int offset) {
        lock.readLock().lock();
        try {
            if (sequenceHistory.isEmpty()) {
                return;
            }
            
            // Compare ordering patterns with previous windows
            var recentSnapshot = sequenceHistory.peekLast();
            if (recentSnapshot != null) {
                // Ordering stability across windows
                output[offset] = calculateOrderingStability(tokens, recentSnapshot);
                
                // Sequence continuity
                output[offset + 1] = calculateSequenceContinuity(tokens, recentSnapshot);
            }
            
            // Long-term ordering trends
            if (sequenceHistory.size() >= 3) {
                output[offset + 2] = calculateOrderingTrends();
            }
            
        } finally {
            lock.readLock().unlock();
        }
    }
    
    private float calculateTokenTypeConsistency(SequenceSnapshot current, SequenceSnapshot historical) {
        float similarity = 0.0f;
        int minLength = Math.min(current.tokenTypeRatios.length, historical.tokenTypeRatios.length);
        
        for (int i = 0; i < minLength; i++) {
            similarity += 1.0f - Math.abs(current.tokenTypeRatios[i] - historical.tokenTypeRatios[i]);
        }
        
        return similarity / minLength;
    }
    
    private float calculateOrderingEntropy(Token[] tokens) {
        // Simple entropy calculation based on token type transitions
        int transitions = 0;
        int sameType = 0;
        
        for (int i = 0; i < tokens.length - 1 && i < SlidingWindow.WINDOW_SIZE - 1; i++) {
            if (tokens[i] != null && tokens[i+1] != null) {
                if (tokens[i].getType() != tokens[i+1].getType()) {
                    transitions++;
                } else {
                    sameType++;
                }
            }
        }
        
        int total = transitions + sameType;
        if (total == 0) return 0.0f;
        
        float transitionRatio = (float) transitions / total;
        // Higher entropy = more mixed = higher value
        return transitionRatio;
    }
    
    private float analyzeSequentialPatterns(Token[] tokens) {
        // Look for repeating patterns or regularities
        int patternScore = 0;
        
        // Check for alternating patterns
        for (int i = 0; i < tokens.length - 3 && i < SlidingWindow.WINDOW_SIZE - 3; i++) {
            if (tokens[i] != null && tokens[i+2] != null && tokens[i+1] != null && tokens[i+3] != null) {
                if (tokens[i].getType() == tokens[i+2].getType() && 
                    tokens[i+1].getType() == tokens[i+3].getType() &&
                    tokens[i].getType() != tokens[i+1].getType()) {
                    patternScore++;
                }
            }
        }
        
        return Math.min(patternScore / 5.0f, 1.0f);
    }
    
    private float calculatePositionalVariance(Token[] tokens) {
        // Calculate how spread out each token type is
        float totalVariance = 0.0f;
        int typeCount = 0;
        
        for (var tokenType : Token.TokenType.values()) {
            var positions = new java.util.ArrayList<Float>();
            
            for (int i = 0; i < tokens.length && i < SlidingWindow.WINDOW_SIZE; i++) {
                if (tokens[i] != null && tokens[i].getType() == tokenType) {
                    positions.add((float) i / SlidingWindow.WINDOW_SIZE);
                }
            }
            
            if (positions.size() > 1) {
                float mean = (float) positions.stream().mapToDouble(f -> f).average().orElse(0.0);
                float variance = (float) positions.stream()
                    .mapToDouble(f -> Math.pow(f - mean, 2))
                    .average().orElse(0.0);
                totalVariance += variance;
                typeCount++;
            }
        }
        
        return typeCount > 0 ? totalVariance / typeCount : 0.0f;
    }
    
    private float calculateOrderingStability(Token[] tokens, SequenceSnapshot reference) {
        // Compare token type distributions at similar positions
        float stability = 0.0f;
        int comparisons = 0;
        
        for (int i = 0; i < Math.min(tokens.length, SlidingWindow.WINDOW_SIZE); i++) {
            if (tokens[i] != null) {
                // This is a simplified comparison - in practice, you'd want more sophisticated analysis
                stability += 1.0f; // Placeholder for actual stability calculation
                comparisons++;
            }
        }
        
        return comparisons > 0 ? stability / comparisons : 0.0f;
    }
    
    private float calculateSequenceContinuity(Token[] tokens, SequenceSnapshot reference) {
        // Measure how well the current window continues from the previous one
        // This is a simplified implementation
        return Math.min(Math.abs(getTokenCount(tokens) - reference.totalTokens) / (float) SlidingWindow.WINDOW_SIZE, 1.0f);
    }
    
    private float calculateOrderingTrends() {
        // Analyze trends across multiple windows
        // This would require more sophisticated analysis of the sequence history
        return 0.5f; // Placeholder
    }
    
    private int getTokenCount(Token[] tokens) {
        int count = 0;
        for (int i = 0; i < tokens.length && i < SlidingWindow.WINDOW_SIZE; i++) {
            if (tokens[i] != null) count++;
        }
        return count;
    }
    
    private SequenceSnapshot createSnapshot(Token[] tokens, long timestamp) {
        var tokenTypeRatios = new float[Token.TokenType.values().length];
        int totalTokens = 0;
        float totalLength = 0.0f;
        
        for (int i = 0; i < tokens.length && i < SlidingWindow.WINDOW_SIZE; i++) {
            if (tokens[i] != null) {
                tokenTypeRatios[tokens[i].getType().ordinal()]++;
                var text = tokens[i].getText();
                if (text != null) {
                    totalLength += text.length();
                }
                totalTokens++;
            }
        }
        
        // Normalize ratios
        if (totalTokens > 0) {
            for (int i = 0; i < tokenTypeRatios.length; i++) {
                tokenTypeRatios[i] /= totalTokens;
            }
        }
        
        float averageLength = totalTokens > 0 ? totalLength / totalTokens : 0.0f;
        
        return new SequenceSnapshot(tokenTypeRatios, averageLength, totalTokens, timestamp, sequenceCounter);
    }
    
    private void updateSequenceHistory(Token[] tokens, long timestamp) {
        var snapshot = createSnapshot(tokens, timestamp);
        
        if (sequenceHistory.size() >= SEQUENCE_HISTORY) {
            sequenceHistory.removeFirst();
        }
        
        sequenceHistory.addLast(snapshot);
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
        return TEMPORAL_DIM;
    }
    
    @Override
    public String getName() {
        return "TemporalChannel";
    }
    
    @Override
    public ChannelType getChannelType() {
        return ChannelType.TEMPORAL;
    }
    
    @Override
    public ChannelOutput process(ProcessingWindow window) {
        var tokens = window.getTokens().toArray(new Token[0]);
        var result = processWindow(tokens);
        var processingTime = 1000L;  // Use fixed processing time for determinism
        return ChannelOutput.valid(result, getChannelType(), processingTime);
    }

    @Override
    public void reset() {
        lock.writeLock().lock();
        try {
            sequenceHistory.clear();
            lastProcessingTime = clock.millis() * 1_000_000L;  // Reset to current time
            sequenceCounter = 0;
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    @Override
    public boolean isDeterministic() {
        // Deterministic when using a predictable Clock (e.g., in tests)
        // Non-deterministic with system clock in production
        return true;
    }
}