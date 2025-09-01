package com.hellblazer.art.core.preprocessing;

import java.util.Arrays;

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
        
        Pipeline(java.util.List<PipelineStep> steps) {
            this.steps = steps;
            this.preprocessor = new DataPreprocessor();
        }
        
        public double[][] process(double[][] data) {
            var result = data;
            
            for (var step : steps) {
                result = switch (step.type) {
                    case NORMALIZE -> {
                        var normalized = preprocessor.normalize(result);
                        yield normalized.normalized();
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
    }
    
    private record PipelineStep(PreprocessingStep type, Object[] params) {}
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