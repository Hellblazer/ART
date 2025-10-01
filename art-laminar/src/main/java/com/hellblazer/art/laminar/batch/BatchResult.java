package com.hellblazer.art.laminar.batch;

import com.hellblazer.art.core.Pattern;

import java.util.Arrays;
import java.util.Objects;

/**
 * Results from batch processing operation.
 *
 * <p>Contains per-pattern outputs and batch-level statistics for analyzing
 * batch processing performance and results.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * var result = circuit.processBatch(patterns);
 *
 * // Access individual results
 * for (int i = 0; i < result.batchSize(); i++) {
 *     var patternResult = result.getResult(i);
 *     System.out.printf("Pattern %d: category=%d, resonating=%s%n",
 *         i, patternResult.categoryId(), patternResult.resonating());
 * }
 *
 * // Check batch statistics
 * System.out.printf("Processed %d patterns in %.2f ms%n",
 *     result.batchSize(),
 *     result.statistics().totalTimeNanos() / 1e6);
 * }</pre>
 *
 * @param outputs output pattern for each input (expectation if resonating)
 * @param categoryIds category ID for each pattern (-1 if mismatch)
 * @param activationValues activation value for each pattern
 * @param resonating whether each pattern achieved resonance
 * @param statistics batch-level performance metrics
 *
 * @see BatchProcessable#processBatch(Pattern[])
 * @author Hal Hildebrand
 */
public record BatchResult(
    Pattern[] outputs,
    int[] categoryIds,
    double[] activationValues,
    boolean[] resonating,
    BatchStatistics statistics
) {
    /**
     * Validate batch result consistency.
     *
     * @throws NullPointerException if any parameter is null
     * @throws IllegalArgumentException if array lengths don't match
     */
    public BatchResult {
        Objects.requireNonNull(outputs, "outputs cannot be null");
        Objects.requireNonNull(categoryIds, "categoryIds cannot be null");
        Objects.requireNonNull(activationValues, "activationValues cannot be null");
        Objects.requireNonNull(resonating, "resonating cannot be null");
        Objects.requireNonNull(statistics, "statistics cannot be null");

        if (outputs.length != categoryIds.length ||
            outputs.length != activationValues.length ||
            outputs.length != resonating.length) {
            throw new IllegalArgumentException(
                String.format("Array length mismatch: outputs=%d, categoryIds=%d, " +
                             "activationValues=%d, resonating=%d",
                    outputs.length, categoryIds.length,
                    activationValues.length, resonating.length));
        }

        if (outputs.length != statistics.batchSize()) {
            throw new IllegalArgumentException(
                String.format("Array length %d != statistics batch size %d",
                    outputs.length, statistics.batchSize()));
        }

        // Validate no null patterns
        for (int i = 0; i < outputs.length; i++) {
            if (outputs[i] == null) {
                throw new NullPointerException("outputs[" + i + "] is null");
            }
        }

        // Deep copy arrays for immutability
        outputs = outputs.clone();
        categoryIds = categoryIds.clone();
        activationValues = activationValues.clone();
        resonating = resonating.clone();
    }

    /**
     * Get result for specific pattern index.
     *
     * @param index pattern index in batch
     * @return pattern result
     * @throws IndexOutOfBoundsException if index out of bounds
     */
    public PatternResult getResult(int index) {
        if (index < 0 || index >= outputs.length) {
            throw new IndexOutOfBoundsException(
                "Index " + index + " out of bounds for batch size " + outputs.length);
        }
        return new PatternResult(
            outputs[index],
            categoryIds[index],
            activationValues[index],
            resonating[index]
        );
    }

    /**
     * Get number of patterns in batch.
     *
     * @return batch size
     */
    public int batchSize() {
        return outputs.length;
    }

    /**
     * Get count of patterns that achieved resonance.
     *
     * @return number of resonating patterns
     */
    public int getResonanceCount() {
        int count = 0;
        for (var r : resonating) {
            if (r) count++;
        }
        return count;
    }

    /**
     * Get count of patterns that did not achieve resonance.
     *
     * @return number of mismatching patterns
     */
    public int getMismatchCount() {
        return batchSize() - getResonanceCount();
    }

    /**
     * Get resonance rate as percentage.
     *
     * @return resonance rate in [0.0, 100.0]
     */
    public double getResonanceRate() {
        return (100.0 * getResonanceCount()) / batchSize();
    }

    /**
     * Get count of new categories created during batch processing.
     * Extracted from statistics.
     *
     * @return number of new categories
     */
    public int getNewCategoryCount() {
        return statistics.categoriesCreated();
    }

    /**
     * Get array of category IDs for resonating patterns only.
     *
     * @return category IDs of resonating patterns
     */
    public int[] getResonatingCategoryIds() {
        return Arrays.stream(categoryIds)
            .filter(id -> id >= 0)
            .toArray();
    }

    /**
     * Get average activation value across resonating patterns.
     *
     * @return average activation (0.0 if no resonance)
     */
    public double getAverageActivation() {
        double sum = 0.0;
        int count = 0;
        for (int i = 0; i < resonating.length; i++) {
            if (resonating[i]) {
                sum += activationValues[i];
                count++;
            }
        }
        return count > 0 ? sum / count : 0.0;
    }

    /**
     * Check if all patterns achieved resonance.
     *
     * @return true if all patterns resonated
     */
    public boolean allResonating() {
        return getResonanceCount() == batchSize();
    }

    /**
     * Check if no patterns achieved resonance.
     *
     * @return true if no patterns resonated
     */
    public boolean noneResonating() {
        return getResonanceCount() == 0;
    }

    /**
     * Get patterns that achieved resonance.
     *
     * @return array of resonating patterns (in original order)
     */
    public Pattern[] getResonatingPatterns() {
        return Arrays.stream(outputs)
            .filter(p -> {
                var idx = Arrays.asList(outputs).indexOf(p);
                return resonating[idx];
            })
            .toArray(Pattern[]::new);
    }

    @Override
    public String toString() {
        return String.format(
            "BatchResult[size=%d, resonance=%d (%.1f%%), newCategories=%d, %.2fms total]",
            batchSize(),
            getResonanceCount(),
            getResonanceRate(),
            getNewCategoryCount(),
            statistics.totalTimeNanos() / 1e6
        );
    }

    /**
     * Generate detailed report string.
     *
     * @return multi-line detailed report
     */
    public String toDetailedString() {
        var sb = new StringBuilder();
        sb.append("=== Batch Processing Results ===\n");
        sb.append(String.format("Batch size:      %d patterns\n", batchSize()));
        sb.append(String.format("Resonating:      %d (%.1f%%)\n",
            getResonanceCount(), getResonanceRate()));
        sb.append(String.format("Mismatches:      %d (%.1f%%)\n",
            getMismatchCount(), 100.0 - getResonanceRate()));
        sb.append(String.format("New categories:  %d\n", getNewCategoryCount()));
        sb.append(String.format("Avg activation:  %.3f\n", getAverageActivation()));
        sb.append("\n");
        sb.append(statistics.toDetailedString());
        return sb.toString();
    }
}
