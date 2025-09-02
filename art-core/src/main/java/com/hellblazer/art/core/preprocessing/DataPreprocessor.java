package com.hellblazer.art.core.preprocessing;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

public class DataPreprocessor {
    
    public record NormalizedData(double[][] normalized, double[] min, double[] max) {}
    
    public record DataBounds(double[] min, double[] max) {}
    
    public NormalizedData normalize(double[][] data) {
        if (data.length == 0) {
            throw new IllegalArgumentException("Data cannot be empty");
        }
        
        var numRows = data.length;
        var numCols = data[0].length;
        var min = new double[numCols];
        var max = new double[numCols];
        
        Arrays.fill(min, Double.MAX_VALUE);
        Arrays.fill(max, -Double.MAX_VALUE);
        
        // Find min and max for each column
        for (var row : data) {
            for (int j = 0; j < numCols; j++) {
                if (!Double.isNaN(row[j])) {
                    min[j] = Math.min(min[j], row[j]);
                    max[j] = Math.max(max[j], row[j]);
                }
            }
        }
        
        return normalize(data, min, max);
    }
    
    public NormalizedData normalize(double[][] data, double[] min, double[] max) {
        if (data.length == 0) {
            throw new IllegalArgumentException("Data cannot be empty");
        }
        
        var numRows = data.length;
        var numCols = data[0].length;
        var normalized = new double[numRows][numCols];
        
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                var range = max[j] - min[j];
                if (range == 0 || Double.isNaN(data[i][j])) {
                    normalized[i][j] = 0.0;
                } else {
                    normalized[i][j] = (data[i][j] - min[j]) / range;
                }
            }
        }
        
        return new NormalizedData(normalized, min, max);
    }
    
    public double[][] denormalize(double[][] normalized, double[] min, double[] max) {
        var numRows = normalized.length;
        var numCols = normalized[0].length;
        var denormalized = new double[numRows][numCols];
        
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                denormalized[i][j] = normalized[i][j] * (max[j] - min[j]) + min[j];
            }
        }
        
        return denormalized;
    }
    
    public double[][] complementCode(double[][] data) {
        var numRows = data.length;
        var numCols = data[0].length;
        var complemented = new double[numRows][numCols * 2];
        
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                complemented[i][j] = data[i][j];
                complemented[i][j + numCols] = 1.0 - data[i][j];
            }
        }
        
        return complemented;
    }
    
    public double[][] deComplementCode(double[][] complementCoded) {
        var numRows = complementCoded.length;
        var totalCols = complementCoded[0].length;
        
        if (totalCols % 2 != 0) {
            throw new IllegalArgumentException("Number of columns must be even for de-complement coding");
        }
        
        var numCols = totalCols / 2;
        var deComplemented = new double[numRows][numCols];
        
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                var original = complementCoded[i][j];
                var complement = 1.0 - complementCoded[i][j + numCols];
                deComplemented[i][j] = (original + complement) / 2.0;
            }
        }
        
        return deComplemented;
    }
    
    public double[][] l1Normalize(double[][] data) {
        var numRows = data.length;
        var numCols = data[0].length;
        var normalized = new double[numRows][numCols];
        
        for (int i = 0; i < numRows; i++) {
            var norm = 0.0;
            for (int j = 0; j < numCols; j++) {
                norm += Math.abs(data[i][j]);
            }
            
            if (norm > 0) {
                for (int j = 0; j < numCols; j++) {
                    normalized[i][j] = data[i][j] / norm;
                }
            }
        }
        
        return normalized;
    }
    
    public double[][] l2Normalize(double[][] data) {
        var numRows = data.length;
        var numCols = data[0].length;
        var normalized = new double[numRows][numCols];
        
        for (int i = 0; i < numRows; i++) {
            var sumSquares = 0.0;
            for (int j = 0; j < numCols; j++) {
                sumSquares += data[i][j] * data[i][j];
            }
            var norm = Math.sqrt(sumSquares);
            
            if (norm > 0) {
                for (int j = 0; j < numCols; j++) {
                    normalized[i][j] = data[i][j] / norm;
                }
            }
        }
        
        return normalized;
    }
    
    public double[][] handleMissingValues(double[][] data, MissingValueStrategy strategy) {
        var numRows = data.length;
        var numCols = data[0].length;
        var result = new double[numRows][numCols];
        
        // Copy data
        for (int i = 0; i < numRows; i++) {
            System.arraycopy(data[i], 0, result[i], 0, numCols);
        }
        
        if (strategy == MissingValueStrategy.MEAN) {
            // Calculate column means excluding NaN values
            var columnMeans = new double[numCols];
            for (int j = 0; j < numCols; j++) {
                var sum = 0.0;
                var count = 0;
                for (int i = 0; i < numRows; i++) {
                    if (!Double.isNaN(data[i][j])) {
                        sum += data[i][j];
                        count++;
                    }
                }
                columnMeans[j] = count > 0 ? sum / count : 0.0;
            }
            
            // Replace NaN values with column means
            for (int i = 0; i < numRows; i++) {
                for (int j = 0; j < numCols; j++) {
                    if (Double.isNaN(result[i][j])) {
                        result[i][j] = columnMeans[j];
                    }
                }
            }
        } else if (strategy == MissingValueStrategy.ZERO) {
            // Replace NaN values with zero
            for (int i = 0; i < numRows; i++) {
                for (int j = 0; j < numCols; j++) {
                    if (Double.isNaN(result[i][j])) {
                        result[i][j] = 0.0;
                    }
                }
            }
        }
        
        return result;
    }
    
    public DataBounds findBounds(double[][]... dataBatches) {
        if (dataBatches.length == 0 || dataBatches[0].length == 0) {
            throw new IllegalArgumentException("Data batches cannot be empty");
        }
        
        var numCols = dataBatches[0][0].length;
        var min = new double[numCols];
        var max = new double[numCols];
        
        Arrays.fill(min, Double.MAX_VALUE);
        Arrays.fill(max, -Double.MAX_VALUE);
        
        for (var batch : dataBatches) {
            for (var row : batch) {
                for (int j = 0; j < numCols; j++) {
                    if (!Double.isNaN(row[j])) {
                        min[j] = Math.min(min[j], row[j]);
                        max[j] = Math.max(max[j], row[j]);
                    }
                }
            }
        }
        
        return new DataBounds(min, max);
    }
    
    // Pattern-aware methods for working with ART Pattern objects
    
    /**
     * Convert Pattern array to 2D double array.
     */
    public static double[][] patternsToArray(Pattern[] patterns) {
        if (patterns.length == 0) {
            return new double[0][0];
        }
        var data = new double[patterns.length][];
        for (int i = 0; i < patterns.length; i++) {
            // Extract data from Pattern - if it's a DenseVector, get its data array
            if (patterns[i] instanceof DenseVector dv) {
                data[i] = Arrays.copyOf(dv.data(), dv.data().length);
            } else {
                // For other Pattern implementations, extract element by element
                var dim = patterns[i].dimension();
                data[i] = new double[dim];
                for (int j = 0; j < dim; j++) {
                    data[i][j] = patterns[i].get(j);
                }
            }
        }
        return data;
    }
    
    /**
     * Convert 2D double array to Pattern array.
     */
    public static Pattern[] arrayToPatterns(double[][] data) {
        var patterns = new Pattern[data.length];
        for (int i = 0; i < data.length; i++) {
            patterns[i] = Pattern.of(data[i]);
        }
        return patterns;
    }
    
    /**
     * Apply complement coding to Pattern arrays.
     */
    public Pattern[] complementCode(Pattern[] patterns) {
        var data = patternsToArray(patterns);
        var complemented = complementCode(data);
        return arrayToPatterns(complemented);
    }
    
    /**
     * Fit the preprocessor to data and transform it.
     */
    public Pattern[] fitTransform(Pattern[] patterns) {
        if (pipeline != null) {
            var data = patternsToArray(patterns);
            var processed = pipeline.fit(data);
            processed = pipeline.transform(processed);
            return arrayToPatterns(processed);
        }
        // Default: just apply complement coding for ART algorithms
        return complementCode(patterns);
    }
    
    /**
     * Transform data using fitted parameters.
     */
    public Pattern[] transform(Pattern[] patterns) {
        if (pipeline != null) {
            var data = patternsToArray(patterns);
            var processed = pipeline.transform(data);
            return arrayToPatterns(processed);
        }
        // Default: just apply complement coding
        return complementCode(patterns);
    }
    
    private Pipeline pipeline;
    
    /**
     * Create a builder for configuring the preprocessor.
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * Enhanced builder for creating configured DataPreprocessor instances.
     */
    public static class Builder {
        private final List<PipelineStep> steps = new ArrayList<>();
        private boolean normalizeFirst = false;
        private double[] min;
        private double[] max;
        
        public Builder addNormalization() {
            normalizeFirst = true;
            return this;
        }
        
        public Builder addNormalization(double[] min, double[] max) {
            normalizeFirst = true;
            this.min = min;
            this.max = max;
            return this;
        }
        
        public Builder addComplementCoding() {
            steps.add(new PipelineStep(PreprocessingStep.COMPLEMENT_CODE, new Object[0]));
            return this;
        }
        
        public Builder addL1Normalization() {
            steps.add(new PipelineStep(PreprocessingStep.L1_NORMALIZE, new Object[0]));
            return this;
        }
        
        public Builder addL2Normalization() {
            steps.add(new PipelineStep(PreprocessingStep.L2_NORMALIZE, new Object[0]));
            return this;
        }
        
        public Builder handleMissingValues(MissingValueStrategy strategy) {
            steps.add(new PipelineStep(PreprocessingStep.HANDLE_MISSING, new Object[]{strategy}));
            return this;
        }
        
        public DataPreprocessor build() {
            var preprocessor = new DataPreprocessor();
            
            // Build the pipeline with the configured steps
            var allSteps = new ArrayList<PipelineStep>();
            
            // Add normalization first if requested
            if (normalizeFirst) {
                allSteps.add(new PipelineStep(PreprocessingStep.NORMALIZE, 
                    min != null ? new Object[]{min, max} : new Object[0]));
            }
            
            // Add all other steps
            allSteps.addAll(steps);
            
            if (!allSteps.isEmpty()) {
                preprocessor.pipeline = new Pipeline(allSteps);
            }
            
            return preprocessor;
        }
    }
    
    public PipelineBuilder createPipeline() {
        return new PipelineBuilder();
    }
    
    public static class PipelineBuilder {
        private final java.util.List<PipelineStep> steps = new java.util.ArrayList<>();
        
        public PipelineBuilder addStep(PreprocessingStep step, Object... params) {
            steps.add(new PipelineStep(step, params));
            return this;
        }
        
        public Pipeline build() {
            return new Pipeline(steps);
        }
    }
    
    public static class Pipeline {
        private final java.util.List<PipelineStep> steps;
        private final DataPreprocessor preprocessor;
        private NormalizedData normalizedData;
        
        Pipeline(java.util.List<PipelineStep> steps) {
            this.steps = steps;
            this.preprocessor = new DataPreprocessor();
        }
        
        public double[][] fit(double[][] data) {
            // Store normalization parameters if normalization is used
            for (var step : steps) {
                if (step.type == PreprocessingStep.NORMALIZE) {
                    if (step.params.length >= 2) {
                        // Use provided min/max
                        var min = (double[]) step.params[0];
                        var max = (double[]) step.params[1];
                        normalizedData = preprocessor.normalize(data, min, max);
                    } else {
                        // Compute min/max from data
                        normalizedData = preprocessor.normalize(data);
                    }
                    break;
                }
            }
            return data;
        }
        
        public double[][] transform(double[][] data) {
            var result = data;
            
            for (var step : steps) {
                result = switch (step.type) {
                    case NORMALIZE -> {
                        if (normalizedData != null) {
                            var normalized = preprocessor.normalize(result, 
                                normalizedData.min(), normalizedData.max());
                            yield normalized.normalized();
                        } else {
                            var normalized = preprocessor.normalize(result);
                            yield normalized.normalized();
                        }
                    }
                    case COMPLEMENT_CODE -> preprocessor.complementCode(result);
                    case L1_NORMALIZE -> preprocessor.l1Normalize(result);
                    case L2_NORMALIZE -> preprocessor.l2Normalize(result);
                    case HANDLE_MISSING -> {
                        var strategy = step.params.length > 0 ? 
                            (MissingValueStrategy) step.params[0] : MissingValueStrategy.MEAN;
                        yield preprocessor.handleMissingValues(result, strategy);
                    }
                };
            }
            
            return result;
        }
        
        public double[][] process(double[][] data) {
            fit(data);
            return transform(data);
        }
    }
    
    static record PipelineStep(PreprocessingStep type, Object[] params) {}
}

enum MissingValueStrategy {
    MEAN,
    ZERO,
    MEDIAN,
    FORWARD_FILL,
    BACKWARD_FILL
}

enum PreprocessingStep {
    NORMALIZE,
    COMPLEMENT_CODE,
    L1_NORMALIZE,
    L2_NORMALIZE,
    HANDLE_MISSING
}