package com.hellblazer.art.performance.algorithms;

import java.util.Arrays;
import java.util.Objects;

/**
 * Parameters for VectorizedSMART (Self-Monitoring Adaptive Resonance Theory).
 * 
 * SMART performs hierarchical clustering using multiple ART modules with
 * different vigilance values that monotonically increase in restrictiveness.
 */
public class VectorizedSMARTParameters {
    
    // Hierarchical parameters
    private final double[] vigilanceValues;  // Must be monotonically increasing
    private final int numLayers;
    private final boolean allowInterLayerConnections;
    
    // Core ART parameters  
    private final double alpha;
    private final double learningRate;
    private final int inputDimension;
    
    // Module type selection
    private final ModuleType moduleType;
    
    // Performance parameters
    private final boolean parallelLayerProcessing;
    private final int maxCategoriesPerLayer;
    private final double pruningThreshold;
    
    // Hierarchical mapping control
    private final boolean trackCategoryPaths;
    private final double minInterLayerSimilarity;
    
    // Base parameters
    private final VectorizedParameters baseParameters;
    
    public enum ModuleType {
        FUZZY_ART,
        GAUSSIAN_ART, 
        BAYESIAN_ART,
        HYPERSPHERE_ART,
        TOPO_ART
    }
    
    public VectorizedSMARTParameters(
            double[] vigilanceValues,
            double alpha,
            double learningRate,
            int inputDimension,
            ModuleType moduleType,
            boolean allowInterLayerConnections,
            boolean parallelLayerProcessing,
            int maxCategoriesPerLayer,
            double pruningThreshold,
            boolean trackCategoryPaths,
            double minInterLayerSimilarity,
            VectorizedParameters baseParameters) {
        
        validateParameters(vigilanceValues, alpha, learningRate, inputDimension,
                         maxCategoriesPerLayer, pruningThreshold, minInterLayerSimilarity);
        
        this.vigilanceValues = Arrays.copyOf(vigilanceValues, vigilanceValues.length);
        this.numLayers = vigilanceValues.length;
        this.alpha = alpha;
        this.learningRate = learningRate;
        this.inputDimension = inputDimension;
        this.moduleType = Objects.requireNonNull(moduleType, "Module type cannot be null");
        this.allowInterLayerConnections = allowInterLayerConnections;
        this.parallelLayerProcessing = parallelLayerProcessing;
        this.maxCategoriesPerLayer = maxCategoriesPerLayer;
        this.pruningThreshold = pruningThreshold;
        this.trackCategoryPaths = trackCategoryPaths;
        this.minInterLayerSimilarity = minInterLayerSimilarity;
        this.baseParameters = Objects.requireNonNull(baseParameters, "Base parameters cannot be null");
    }
    
    /**
     * Create default SMART parameters with 3 hierarchical layers.
     */
    public VectorizedSMARTParameters() {
        this(new double[]{0.3, 0.5, 0.7}, 0.001, 0.5, 10, ModuleType.FUZZY_ART,
             true, true, 100, 0.1, true, 0.5, VectorizedParameters.createDefault());
    }
    
    /**
     * Create SMART parameters with specified vigilance values.
     */
    public static VectorizedSMARTParameters withVigilanceValues(double[] vigilanceValues) {
        return new VectorizedSMARTParameters(
            vigilanceValues, 0.001, 0.5, 10, ModuleType.FUZZY_ART,
            true, true, 100, 0.1, true, 0.5, VectorizedParameters.createDefault()
        );
    }
    
    private static void validateParameters(double[] vigilanceValues, double alpha,
                                          double learningRate, int inputDimension,
                                          int maxCategoriesPerLayer, double pruningThreshold,
                                          double minInterLayerSimilarity) {
        if (vigilanceValues == null || vigilanceValues.length == 0) {
            throw new IllegalArgumentException("Vigilance values cannot be null or empty");
        }
        
        // Check monotonic increase
        for (int i = 1; i < vigilanceValues.length; i++) {
            if (vigilanceValues[i] < vigilanceValues[i-1]) {
                throw new IllegalArgumentException(
                    "Vigilance values must be monotonically increasing, but " +
                    vigilanceValues[i] + " < " + vigilanceValues[i-1]
                );
            }
        }
        
        // Validate ranges
        for (double vigilance : vigilanceValues) {
            if (vigilance < 0.0 || vigilance > 1.0) {
                throw new IllegalArgumentException("Vigilance must be in [0, 1], got: " + vigilance);
            }
        }
        
        if (alpha < 0.0) {
            throw new IllegalArgumentException("Alpha must be non-negative, got: " + alpha);
        }
        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be in [0, 1], got: " + learningRate);
        }
        if (inputDimension < 1) {
            throw new IllegalArgumentException("Input dimension must be >= 1, got: " + inputDimension);
        }
        if (maxCategoriesPerLayer < 1) {
            throw new IllegalArgumentException("Max categories per layer must be >= 1, got: " + maxCategoriesPerLayer);
        }
        if (pruningThreshold < 0.0 || pruningThreshold > 1.0) {
            throw new IllegalArgumentException("Pruning threshold must be in [0, 1], got: " + pruningThreshold);
        }
        if (minInterLayerSimilarity < 0.0 || minInterLayerSimilarity > 1.0) {
            throw new IllegalArgumentException("Min inter-layer similarity must be in [0, 1], got: " + minInterLayerSimilarity);
        }
    }
    
    // Getters
    
    public double[] getVigilanceValues() {
        return Arrays.copyOf(vigilanceValues, vigilanceValues.length);
    }
    
    public double getVigilanceForLayer(int layer) {
        if (layer < 0 || layer >= numLayers) {
            throw new IndexOutOfBoundsException("Layer index out of bounds: " + layer);
        }
        return vigilanceValues[layer];
    }
    
    public int getNumLayers() {
        return numLayers;
    }
    
    public boolean isAllowInterLayerConnections() {
        return allowInterLayerConnections;
    }
    
    public double getAlpha() {
        return alpha;
    }
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public int getInputDimension() {
        return inputDimension;
    }
    
    public ModuleType getModuleType() {
        return moduleType;
    }
    
    public boolean isParallelLayerProcessing() {
        return parallelLayerProcessing;
    }
    
    public int getMaxCategoriesPerLayer() {
        return maxCategoriesPerLayer;
    }
    
    public double getPruningThreshold() {
        return pruningThreshold;
    }
    
    public boolean isTrackCategoryPaths() {
        return trackCategoryPaths;
    }
    
    public double getMinInterLayerSimilarity() {
        return minInterLayerSimilarity;
    }
    
    public VectorizedParameters getBaseParameters() {
        return baseParameters;
    }
    
    /**
     * Check if categories from different layers can be connected.
     */
    public boolean canConnectLayers(int layer1, int layer2) {
        if (!allowInterLayerConnections) {
            return false;
        }
        return Math.abs(layer1 - layer2) == 1;  // Only adjacent layers can connect
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedSMARTParameters{layers=%d, vigilance=%s, " +
                           "moduleType=%s, alpha=%.4f, lr=%.3f, parallel=%b}",
                           numLayers, Arrays.toString(vigilanceValues),
                           moduleType, alpha, learningRate, parallelLayerProcessing);
    }
}