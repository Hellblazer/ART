package com.hellblazer.art.hybrid.pan.memory;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.hybrid.pan.weight.BPARTWeight;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Manages STM (Short-Term Memory) and LTM (Long-Term Memory) for PAN.
 * STM holds recent activations that decay over time.
 * LTM holds consolidated stable patterns.
 */
public class DualMemoryManager implements AutoCloseable {

    private final double stmDecayRate;
    private final double ltmConsolidationThreshold;

    // STM: Recent patterns with timestamps
    private final LinkedList<STMEntry> stmBuffer;
    private static final int MAX_STM_SIZE = 100;

    // LTM: Consolidated category patterns
    private final Map<Integer, LTMEntry> ltmStorage;

    // Category tracking
    private final Map<Integer, CategoryStats> categoryStats;

    private volatile boolean closed = false;

    public DualMemoryManager(double stmDecayRate, double ltmConsolidationThreshold) {
        this.stmDecayRate = stmDecayRate;
        this.ltmConsolidationThreshold = ltmConsolidationThreshold;
        this.stmBuffer = new LinkedList<>();
        this.ltmStorage = new ConcurrentHashMap<>();
        this.categoryStats = new ConcurrentHashMap<>();
    }

    /**
     * Enhance features using memory information (Paper Eq. 6: X' = f(X, W)).
     * This combines the input features with stored memory for better discrimination.
     */
    public Pattern enhanceFeatures(Pattern features, BPARTWeight weight) {
        // Simple linear combination: X' = mX + nW
        // where m and n are weighting factors
        double m = 0.7;  // Input weight
        double n = 0.3;  // Memory weight

        double[] enhanced = new double[features.dimension()];

        // Combine input features with weight's forward weights (STM)
        for (int i = 0; i < features.dimension(); i++) {
            enhanced[i] = m * features.get(i);
            if (i < weight.forwardWeights().length) {
                enhanced[i] += n * weight.forwardWeights()[i];
            }
        }

        // Create enhanced pattern
        return new com.hellblazer.art.core.DenseVector(enhanced);
    }

    /**
     * Check enhanced vigilance using STM/LTM networks.
     */
    public boolean checkEnhancedVigilance(Pattern features, BPARTWeight weight, double vigilance) {
        // Basic vigilance check
        double activation = weight.calculateActivation(features);
        if (activation < vigilance) {
            return false;
        }

        // Check STM consistency
        for (var entry : stmBuffer) {
            if (entry.isRecent() && entry.weight != null && entry.weight.equals(weight)) {
                double similarity = patternSimilarity(features, entry.pattern);
                if (similarity < vigilance * stmDecayRate) {
                    return false;  // Too different from recent patterns
                }
            }
        }

        // Check LTM if category is consolidated
        var ltmEntry = ltmStorage.get(weight.hashCode());
        if (ltmEntry != null) {
            double ltmSimilarity = patternSimilarity(features, ltmEntry.prototype);
            if (ltmSimilarity < vigilance * 0.9) {  // Stricter for LTM
                return false;
            }
        }

        return true;
    }

    /**
     * Register a new category.
     */
    public void registerNewCategory(int categoryIndex, Pattern initialPattern) {
        categoryStats.put(categoryIndex, new CategoryStats());
        addToSTM(categoryIndex, initialPattern, null);
    }

    /**
     * Update existing category.
     */
    public void updateCategory(int categoryIndex, Pattern pattern) {
        var stats = categoryStats.get(categoryIndex);
        if (stats != null) {
            stats.updateCount++;
            stats.lastUpdateTime = System.currentTimeMillis();
        }

        addToSTM(categoryIndex, pattern, null);
        checkForConsolidation(categoryIndex);
    }

    /**
     * Add pattern to STM buffer.
     */
    private void addToSTM(int category, Pattern pattern, BPARTWeight weight) {
        // Remove oldest if at capacity
        while (stmBuffer.size() >= MAX_STM_SIZE) {
            stmBuffer.removeFirst();
        }

        stmBuffer.addLast(new STMEntry(category, pattern, weight, System.currentTimeMillis()));
    }

    /**
     * Check if category should be consolidated to LTM.
     */
    private void checkForConsolidation(int categoryIndex) {
        var stats = categoryStats.get(categoryIndex);
        if (stats == null) return;

        // Count recent activations in STM
        long recentCount = stmBuffer.stream()
            .filter(e -> e.category == categoryIndex && e.isRecent())
            .count();

        double frequency = (double) recentCount / MAX_STM_SIZE;

        if (frequency > ltmConsolidationThreshold && stats.updateCount > 10) {
            consolidateToLTM(categoryIndex);
        }
    }

    /**
     * Consolidate category to LTM.
     */
    private void consolidateToLTM(int categoryIndex) {
        // Find prototype pattern (average of recent patterns)
        List<Pattern> recentPatterns = stmBuffer.stream()
            .filter(e -> e.category == categoryIndex)
            .map(e -> e.pattern)
            .limit(10)
            .toList();

        if (!recentPatterns.isEmpty()) {
            Pattern prototype = computePrototype(recentPatterns);
            ltmStorage.put(categoryIndex, new LTMEntry(categoryIndex, prototype, System.currentTimeMillis()));
        }
    }

    /**
     * Compute prototype pattern as average.
     */
    private Pattern computePrototype(List<Pattern> patterns) {
        if (patterns.isEmpty()) return null;

        int dim = patterns.get(0).dimension();
        double[] avg = new double[dim];

        for (var pattern : patterns) {
            for (int i = 0; i < dim && i < pattern.dimension(); i++) {
                avg[i] += pattern.get(i);
            }
        }

        for (int i = 0; i < dim; i++) {
            avg[i] /= patterns.size();
        }

        return new com.hellblazer.art.core.DenseVector(avg);
    }

    /**
     * Compute similarity between two patterns.
     */
    private double patternSimilarity(Pattern p1, Pattern p2) {
        double dot = 0, norm1 = 0, norm2 = 0;
        int minDim = Math.min(p1.dimension(), p2.dimension());

        for (int i = 0; i < minDim; i++) {
            double v1 = p1.get(i);
            double v2 = p2.get(i);
            dot += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }

        if (norm1 > 0 && norm2 > 0) {
            return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
        }
        return 0;
    }

    /**
     * Compute LTM confidence based on historical performance.
     * This replaces the previous ad-hoc confidence calculation.
     *
     * @param categoryId The category to compute confidence for
     * @param input The input pattern
     * @return Confidence value bounded [0,1]
     */
    public double computeLTMConfidence(int categoryId, Pattern input) {
        // Check if we have LTM data for this category
        var ltmEntry = ltmStorage.get(categoryId);
        if (ltmEntry == null) {
            // No LTM data yet - use STM-based confidence
            return computeSTMConfidence(categoryId, input);
        }

        // Get category statistics
        var stats = categoryStats.get(categoryId);
        if (stats == null) {
            return 0.0;
        }

        // Confidence components:
        // 1. Success rate: How often this category has been successfully used
        double successRate = Math.min(1.0, stats.updateCount / 100.0); // Normalize to [0,1]

        // 2. Match quality: How well the input matches the LTM prototype
        double matchQuality = computeFuzzyARTSimilarity(ltmEntry.prototype, input);

        // 3. Recency factor: How recently this category was used
        long timeSinceLastUpdate = System.currentTimeMillis() - stats.lastUpdateTime;
        double recencyFactor = Math.exp(-timeSinceLastUpdate / 300000.0); // 5 minute half-life

        // Weighted combination of confidence factors
        double confidence = 0.4 * successRate + 0.4 * matchQuality + 0.2 * recencyFactor;

        return Math.min(1.0, Math.max(0.0, confidence)); // Ensure bounded [0,1]
    }

    /**
     * Compute STM-based confidence when no LTM data is available.
     */
    private double computeSTMConfidence(int categoryId, Pattern input) {
        // Count recent matches in STM
        long recentMatches = stmBuffer.stream()
            .filter(e -> e.category == categoryId && e.isRecent())
            .count();

        if (recentMatches == 0) {
            return 0.0;
        }

        // Get recent patterns for this category
        var recentPatterns = stmBuffer.stream()
            .filter(e -> e.category == categoryId && e.isRecent())
            .map(e -> e.pattern)
            .limit(5)
            .toList();

        // Compute average similarity to recent patterns
        double avgSimilarity = 0.0;
        for (var pattern : recentPatterns) {
            avgSimilarity += computeFuzzyARTSimilarity(pattern, input);
        }
        avgSimilarity /= recentPatterns.size();

        // More conservative confidence calculation for new categories
        // Require more evidence before giving high confidence
        double frequency = Math.min(1.0, recentMatches / 10.0);

        // Penalize categories with few samples - they should have lower confidence
        double samplePenalty = Math.min(1.0, recentMatches / 5.0);

        // Apply conservative weighting: need both high similarity AND sufficient evidence
        double confidence = 0.5 * avgSimilarity + 0.3 * frequency + 0.2 * samplePenalty;

        // Additional conservative factor: for very new categories (< 3 samples), reduce confidence
        if (recentMatches < 3) {
            confidence *= 0.6; // Reduce confidence for new categories
        }

        return Math.min(1.0, Math.max(0.0, confidence));
    }

    /**
     * Compute Fuzzy ART similarity between patterns.
     * Consistent with the similarity measure used in BPARTWeight.
     */
    private double computeFuzzyARTSimilarity(Pattern p1, Pattern p2) {
        double minSum = 0.0;
        double p1Sum = 0.0;
        int minDim = Math.min(p1.dimension(), p2.dimension());

        for (int i = 0; i < minDim; i++) {
            double v1 = Math.abs(p1.get(i));
            double v2 = Math.abs(p2.get(i));
            minSum += Math.min(v1, v2);
            p1Sum += v1;
        }

        if (p1Sum == 0.0) {
            return 0.0;
        }

        return minSum / p1Sum; // Bounded [0,1]
    }

    /**
     * Get memory usage estimate.
     */
    public long estimateMemoryUsage() {
        long stmSize = stmBuffer.size() * 1024L;  // Estimate 1KB per entry
        long ltmSize = ltmStorage.size() * 2048L;  // Estimate 2KB per prototype
        return stmSize + ltmSize;
    }

    /**
     * Clear all memory.
     */
    public void clear() {
        stmBuffer.clear();
        ltmStorage.clear();
        categoryStats.clear();
    }

    @Override
    public void close() {
        closed = true;
        stmBuffer.clear();
        ltmStorage.clear();
        categoryStats.clear();
    }

    /**
     * STM entry with decay tracking.
     */
    private static class STMEntry {
        final int category;
        final Pattern pattern;
        final BPARTWeight weight;
        final long timestamp;

        STMEntry(int category, Pattern pattern, BPARTWeight weight, long timestamp) {
            this.category = category;
            this.pattern = pattern;
            this.weight = weight;
            this.timestamp = timestamp;
        }

        boolean isRecent() {
            return System.currentTimeMillis() - timestamp < 60000;  // Within last minute
        }
    }

    /**
     * LTM entry with prototype pattern.
     */
    private static class LTMEntry {
        final int category;
        final Pattern prototype;
        final long consolidationTime;

        LTMEntry(int category, Pattern prototype, long consolidationTime) {
            this.category = category;
            this.prototype = prototype;
            this.consolidationTime = consolidationTime;
        }
    }

    /**
     * Category statistics for consolidation decisions.
     */
    private static class CategoryStats {
        int updateCount = 0;
        long lastUpdateTime = System.currentTimeMillis();
    }
}