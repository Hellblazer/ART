package com.hellblazer.art.supervised;

import com.hellblazer.art.core.ARTMAPResult;

import java.util.List;

/**
 * Enhanced result types for VectorizedARTMAP operations with performance metrics and detailed tracking.
 * Extends the base ARTMAPResult with vectorization-specific information and match tracking details.
 */
public sealed interface VectorizedARTMAPResult
    permits VectorizedARTMAPResult.Success, 
            VectorizedARTMAPResult.Prediction,
            VectorizedARTMAPResult.MapFieldMismatch,
            VectorizedARTMAPResult.MatchTrackingSearch {
    
    /**
     * Successful VectorizedARTMAP training operation with performance metrics.
     * 
     * @param artAIndex the category index in ARTa that was activated/created
     * @param artBIndex the category index in ARTb that was activated/created
     * @param artAActivation the activation value from ARTa
     * @param artBActivation the activation value from ARTb
     * @param mapFieldActivation the map field resonance value (properly calculated)
     * @param wasNewMapping whether a new mapping was created in the map field
     * @param executionTimeNanos the execution time in nanoseconds
     * @param vectorizationUsed whether vectorized operations were used
     * @param parallelProcessingUsed whether parallel processing was used
     */
    public record Success(
        int artAIndex,
        int artBIndex,
        double artAActivation,
        double artBActivation,
        double mapFieldActivation,
        boolean wasNewMapping,
        long executionTimeNanos,
        boolean vectorizationUsed,
        boolean parallelProcessingUsed
    ) implements VectorizedARTMAPResult {
        
        /**
         * Convert to base ARTMAPResult.Success for backward compatibility.
         */
        public ARTMAPResult.Success toBaseResult() {
            return new ARTMAPResult.Success(
                artAIndex, artBIndex, artAActivation, artBActivation,
                mapFieldActivation, wasNewMapping
            );
        }
    }
    
    /**
     * Enhanced prediction result with confidence metrics.
     * 
     * @param artAIndex the category index in ARTa that was activated
     * @param predictedBIndex the predicted category index in ARTb based on map field
     * @param artAActivation the activation value from ARTa
     * @param confidence the confidence in the prediction based on map field strength
     * @param mapFieldStrength the strength of the ARTa->ARTb mapping
     * @param executionTimeNanos the prediction execution time in nanoseconds
     * @param vectorizationUsed whether vectorized operations were used
     */
    public record Prediction(
        int artAIndex,
        int predictedBIndex,
        double artAActivation,
        double confidence,
        double mapFieldStrength,
        long executionTimeNanos,
        boolean vectorizationUsed
    ) implements VectorizedARTMAPResult {
        
        /**
         * Convert to base ARTMAPResult.Prediction for backward compatibility.
         */
        public ARTMAPResult.Prediction toBaseResult() {
            return new ARTMAPResult.Prediction(
                artAIndex, predictedBIndex, artAActivation, confidence
            );
        }
    }
    
    /**
     * Enhanced map field mismatch with detailed tracking information.
     * 
     * @param artAIndex the ARTa category that caused the mismatch
     * @param expectedBIndex the ARTb category from the map field
     * @param actualBIndex the ARTb category from current target
     * @param mapFieldActivation the map field activation that failed
     * @param resetTriggered whether ARTa reset/search was triggered
     * @param vigilanceAtMismatch the vigilance level when mismatch occurred
     * @param executionTimeNanos the execution time in nanoseconds
     */
    public record MapFieldMismatch(
        int artAIndex,
        int expectedBIndex,
        int actualBIndex,
        double mapFieldActivation,
        boolean resetTriggered,
        double vigilanceAtMismatch,
        long executionTimeNanos
    ) implements VectorizedARTMAPResult {
        
        /**
         * Convert to base ARTMAPResult.MapFieldMismatch for backward compatibility.
         */
        public ARTMAPResult.MapFieldMismatch toBaseResult() {
            return new ARTMAPResult.MapFieldMismatch(
                artAIndex, expectedBIndex, actualBIndex, mapFieldActivation, resetTriggered
            );
        }
    }
    
    /**
     * New result type specific to VectorizedARTMAP for detailed match tracking information.
     * This provides complete visibility into the vigilance search process.
     * 
     * @param initialVigilance the starting vigilance level
     * @param finalVigilance the final vigilance level reached
     * @param vigilanceSearchSteps detailed steps taken during vigilance search
     * @param searchExhausted whether the search was exhausted (reached max attempts/vigilance)
     * @param categoriesSearched number of categories searched
     * @param finalResult the ultimate result after search completion
     * @param totalExecutionTimeNanos the total execution time including search
     */
    public record MatchTrackingSearch(
        double initialVigilance,
        double finalVigilance,
        List<VigilanceSearchStep> vigilanceSearchSteps,
        boolean searchExhausted,
        int categoriesSearched,
        VectorizedARTMAPResult finalResult,
        long totalExecutionTimeNanos
    ) implements VectorizedARTMAPResult {
        
        /**
         * Check if vigilance search occurred.
         * @return true if any vigilance search steps were taken
         */
        public boolean vigilanceSearchOccurred() {
            return !vigilanceSearchSteps.isEmpty();
        }
        
        /**
         * Get the number of vigilance increments performed.
         * @return the count of vigilance search steps
         */
        public int vigilanceIncrementCount() {
            return vigilanceSearchSteps.size();
        }
    }
    
    /**
     * Detailed information about a single step in the vigilance search process.
     * 
     * @param stepNumber the step number in the search sequence (0-based)
     * @param vigilanceLevel the vigilance level for this step
     * @param artAIndex the ARTa category tested at this vigilance level
     * @param activation the activation value achieved
     * @param vigilanceTestResult whether the vigilance test passed
     * @param mapFieldConsistent whether the map field mapping was consistent
     * @param executionTimeNanos the execution time for this step
     */
    public record VigilanceSearchStep(
        int stepNumber,
        double vigilanceLevel,
        int artAIndex,
        double activation,
        boolean vigilanceTestResult,
        boolean mapFieldConsistent,
        long executionTimeNanos
    ) {
        
        /**
         * Check if this step resulted in a successful match.
         * @return true if both vigilance test and map field consistency passed
         */
        public boolean wasSuccessful() {
            return vigilanceTestResult && mapFieldConsistent;
        }
    }
    
    /**
     * Performance metrics for VectorizedARTMAP operations.
     * 
     * @param totalTrainingOperations total number of training operations performed
     * @param totalPredictionOperations total number of prediction operations performed
     * @param matchTrackingSearches total number of match tracking searches performed
     * @param averageSearchDepth average depth of vigilance searches
     * @param mapFieldMismatches total number of map field mismatches encountered
     * @param averageTrainingTime average training time in milliseconds
     * @param averagePredictionTime average prediction time in milliseconds
     * @param vectorizationEfficiency percentage of operations that used vectorization
     * @param parallelProcessingEfficiency percentage of operations that used parallel processing
     */
    public record PerformanceMetrics(
        long totalTrainingOperations,
        long totalPredictionOperations,
        long matchTrackingSearches,
        double averageSearchDepth,
        long mapFieldMismatches,
        double averageTrainingTime,
        double averagePredictionTime,
        double vectorizationEfficiency,
        double parallelProcessingEfficiency
    ) {
        
        /**
         * Get total operations performed.
         * @return sum of training and prediction operations
         */
        public long totalOperations() {
            return totalTrainingOperations + totalPredictionOperations;
        }
        
        /**
         * Get match tracking frequency.
         * @return ratio of searches to training operations
         */
        public double matchTrackingFrequency() {
            return totalTrainingOperations > 0 ? 
                (double) matchTrackingSearches / totalTrainingOperations : 0.0;
        }
        
        /**
         * Get map field mismatch rate.
         * @return ratio of mismatches to training operations
         */
        public double mapFieldMismatchRate() {
            return totalTrainingOperations > 0 ?
                (double) mapFieldMismatches / totalTrainingOperations : 0.0;
        }
        
        /**
         * Get overall efficiency score (0-1).
         * @return weighted average of vectorization and parallel processing efficiency
         */
        public double overallEfficiency() {
            return (vectorizationEfficiency * 0.6) + (parallelProcessingEfficiency * 0.4);
        }
    }
    
    // ================== Enhanced Interface Methods ==================
    
    /**
     * Get the execution time for this operation.
     * @return execution time in nanoseconds
     */
    default long getExecutionTimeNanos() {
        return switch (this) {
            case Success s -> s.executionTimeNanos;
            case Prediction p -> p.executionTimeNanos;
            case MapFieldMismatch m -> m.executionTimeNanos;
            case MatchTrackingSearch mt -> mt.totalExecutionTimeNanos;
        };
    }
    
    /**
     * Check if vectorization was used in this operation.
     * @return true if vectorized operations were used
     */
    default boolean wasVectorizationUsed() {
        return switch (this) {
            case Success s -> s.vectorizationUsed;
            case Prediction p -> p.vectorizationUsed;
            case MapFieldMismatch m -> false; // Mismatches don't use vectorization
            case MatchTrackingSearch mt -> mt.finalResult != null && mt.finalResult.wasVectorizationUsed();
        };
    }
    
    /**
     * Check if parallel processing was used in this operation.
     * @return true if parallel processing was used
     */
    default boolean wasParallelProcessingUsed() {
        return switch (this) {
            case Success s -> s.parallelProcessingUsed;
            case Prediction p -> false; // Current implementation doesn't parallelize prediction
            case MapFieldMismatch m -> false;
            case MatchTrackingSearch mt -> mt.finalResult != null && mt.finalResult.wasParallelProcessingUsed();
        };
    }
    
    /**
     * Get a performance summary string.
     * @return formatted performance summary
     */
    default String getPerformanceSummary() {
        var timeMs = getExecutionTimeNanos() / 1_000_000.0;
        var vectorized = wasVectorizationUsed() ? "SIMD" : "Standard";
        var parallel = wasParallelProcessingUsed() ? "Parallel" : "Serial";
        
        return String.format("%s [%.3fms, %s, %s]", 
            getClass().getSimpleName(), timeMs, vectorized, parallel);
    }
    
    /**
     * Check if this result indicates successful operation completion.
     * @return true for Success and Prediction results
     */
    default boolean isSuccess() {
        return this instanceof Success || 
               (this instanceof MatchTrackingSearch mt && mt.finalResult instanceof Success);
    }
    
    /**
     * Check if this result represents a successful prediction.
     * @return true for Prediction results
     */
    default boolean isPrediction() {
        return this instanceof Prediction;
    }
    
    /**
     * Check if this result represents a map field mismatch.
     * @return true for MapFieldMismatch results
     */
    default boolean isMapFieldMismatch() {
        return this instanceof MapFieldMismatch ||
               (this instanceof MatchTrackingSearch mt && mt.finalResult instanceof MapFieldMismatch);
    }
    
    /**
     * Check if this result involved match tracking search.
     * @return true for MatchTrackingSearch results
     */
    default boolean involvedMatchTracking() {
        return this instanceof MatchTrackingSearch;
    }
    
    /**
     * Get the ultimate ARTa category index for any result type.
     * @return the ARTa category index, or -1 if not applicable
     */
    default int getArtAIndex() {
        return switch (this) {
            case Success s -> s.artAIndex;
            case Prediction p -> p.artAIndex;
            case MapFieldMismatch m -> m.artAIndex;
            case MatchTrackingSearch mt -> mt.finalResult != null ? mt.finalResult.getArtAIndex() : -1;
        };
    }
    
    /**
     * Get the ultimate ARTa activation for any result type.
     * @return the ARTa activation value, or NaN if not applicable
     */
    default double getArtAActivation() {
        return switch (this) {
            case Success s -> s.artAActivation;
            case Prediction p -> p.artAActivation;
            case MapFieldMismatch m -> Double.NaN;
            case MatchTrackingSearch mt -> mt.finalResult != null ? mt.finalResult.getArtAActivation() : Double.NaN;
        };
    }
    
    /**
     * Convert to base ARTMAPResult for backward compatibility.
     * @return equivalent base ARTMAPResult
     */
    default ARTMAPResult toBaseResult() {
        return switch (this) {
            case Success s -> s.toBaseResult();
            case Prediction p -> p.toBaseResult();
            case MapFieldMismatch m -> m.toBaseResult();
            case MatchTrackingSearch mt -> mt.finalResult != null ? mt.finalResult.toBaseResult() : 
                new ARTMAPResult.MapFieldMismatch(-1, -1, -1, 0.0, true);
        };
    }
}