package com.hellblazer.art.nlp.processor.fusion;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.nlp.processor.ChannelResult;

import java.util.*;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * PCA-based feature fusion strategy that reduces dimensionality while preserving
 * the most important information from channel features.
 * Uses Principal Component Analysis to find optimal low-dimensional representation.
 */
public class PCAFusion implements FeatureFusionStrategy {
    private static final Logger log = LoggerFactory.getLogger(PCAFusion.class);
    
    private final int targetDimensions;
    private final double varianceThreshold;
    private final boolean normalizeInput;
    private final boolean centerData;
    
    // PCA components (computed during fusion)
    private double[][] principalComponents;
    private double[] meanVector;
    private double[] eigenvalues;
    private boolean pcaComputed = false;
    
    /**
     * Create PCA fusion with default settings.
     */
    public PCAFusion() {
        this(10, 0.95, true, true);
    }
    
    /**
     * Create PCA fusion with custom parameters.
     * 
     * @param targetDimensions Target number of dimensions for output
     * @param varianceThreshold Minimum variance to preserve (0.0-1.0)
     * @param normalizeInput Whether to normalize input features
     * @param centerData Whether to center data around mean
     */
    public PCAFusion(int targetDimensions, double varianceThreshold, 
                     boolean normalizeInput, boolean centerData) {
        if (targetDimensions <= 0) {
            throw new IllegalArgumentException("Target dimensions must be positive: " + targetDimensions);
        }
        if (varianceThreshold < 0.0 || varianceThreshold > 1.0) {
            throw new IllegalArgumentException("Variance threshold must be in [0.0, 1.0]: " + varianceThreshold);
        }
        
        this.targetDimensions = targetDimensions;
        this.varianceThreshold = varianceThreshold;
        this.normalizeInput = normalizeInput;
        this.centerData = centerData;
    }
    
    @Override
    public DenseVector fuseFeatures(Map<String, ChannelResult> channelResults) {
        Objects.requireNonNull(channelResults, "channelResults cannot be null");
        
        // Filter successful channel results
        var successfulResults = channelResults.entrySet().stream()
            .filter(entry -> entry.getValue().isSuccess())
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        
        if (successfulResults.isEmpty()) {
            log.debug("No successful channel results for PCA fusion");
            return null;
        }
        
        if (successfulResults.size() == 1) {
            // Single channel - return transformed features directly
            var entry = successfulResults.entrySet().iterator().next();
            var features = extractChannelFeatures(entry.getKey(), entry.getValue());
            
            if (features != null && features.length > 0) {
                log.debug("Single channel PCA fusion: {}", entry.getKey());
                return new DenseVector(Arrays.copyOf(features, Math.min(features.length, targetDimensions)));
            }
            return null;
        }
        
        // Extract features from all channels
        var channelFeatureMatrix = new ArrayList<double[]>();
        var channelOrder = new ArrayList<String>();
        
        // Process channels in sorted order for consistency
        for (var channelId : successfulResults.keySet().stream().sorted().toList()) {
            var result = successfulResults.get(channelId);
            var features = extractChannelFeatures(channelId, result);
            
            if (features != null && features.length > 0) {
                if (normalizeInput) {
                    features = normalizeVector(features);
                }
                
                channelFeatureMatrix.add(features);
                channelOrder.add(channelId);
            }
        }
        
        if (channelFeatureMatrix.isEmpty()) {
            log.debug("No features extracted for PCA fusion");
            return null;
        }
        
        // Perform PCA
        var pcaResult = performPCA(channelFeatureMatrix);
        
        if (pcaResult == null || pcaResult.length == 0) {
            log.debug("PCA fusion failed to produce output");
            return null;
        }
        
        log.debug("PCA fusion: {} channels → {} dimensions (variance preserved: {:.3f})", 
                 channelOrder.size(), pcaResult.length, computeVariancePreserved());
        
        return new DenseVector(pcaResult);
    }
    
    /**
     * Perform Principal Component Analysis on channel features.
     */
    private double[] performPCA(List<double[]> featureMatrix) {
        if (featureMatrix.isEmpty()) {
            return null;
        }
        
        var numSamples = featureMatrix.size();
        var numFeatures = featureMatrix.get(0).length;
        
        // Ensure all feature vectors have the same length
        var maxLength = featureMatrix.stream().mapToInt(f -> f.length).max().orElse(0);
        var normalizedMatrix = new ArrayList<double[]>();
        
        for (var features : featureMatrix) {
            var normalized = Arrays.copyOf(features, maxLength);
            // Pad with zeros if necessary
            normalizedMatrix.add(normalized);
        }
        
        numFeatures = maxLength;
        
        // Convert to matrix (samples x features)
        var dataMatrix = new double[numSamples][numFeatures];
        for (var i = 0; i < numSamples; i++) {
            System.arraycopy(normalizedMatrix.get(i), 0, dataMatrix[i], 0, numFeatures);
        }
        
        // Center the data if requested
        meanVector = new double[numFeatures];
        if (centerData) {
            // Compute mean
            for (var j = 0; j < numFeatures; j++) {
                var sum = 0.0;
                for (var i = 0; i < numSamples; i++) {
                    sum += dataMatrix[i][j];
                }
                meanVector[j] = sum / numSamples;
            }
            
            // Center data
            for (var i = 0; i < numSamples; i++) {
                for (var j = 0; j < numFeatures; j++) {
                    dataMatrix[i][j] -= meanVector[j];
                }
            }
        }
        
        // Compute covariance matrix (features x features)
        var covarianceMatrix = computeCovarianceMatrix(dataMatrix);
        
        // Compute eigenvalues and eigenvectors
        var eigenDecomposition = computeEigenDecomposition(covarianceMatrix);
        if (eigenDecomposition == null) {
            log.warn("Eigendecomposition failed");
            return createFallbackResult(normalizedMatrix);
        }
        
        eigenvalues = eigenDecomposition.eigenvalues;
        principalComponents = eigenDecomposition.eigenvectors;
        pcaComputed = true;
        
        // Determine number of components to keep
        var numComponents = determineNumComponents(eigenvalues);
        
        // Project data onto principal components
        return projectData(normalizedMatrix, numComponents);
    }
    
    /**
     * Extract feature vector from channel result.
     */
    private double[] extractChannelFeatures(String channelId, ChannelResult result) {
        // Similar to ConcatenationFusion but optimized for PCA
        return switch (channelId.toLowerCase()) {
            case "fasttext", "semantic" -> createSemanticFeatures(result);
            case "entity", "ner" -> createEntityFeatures(result);
            case "syntactic", "pos" -> createSyntacticFeatures(result);
            default -> createGenericFeatures(result);
        };
    }
    
    private double[] createSemanticFeatures(ChannelResult result) {
        var features = new double[16]; // Larger feature set for PCA
        
        // Basic features
        features[0] = result.confidence();
        features[1] = result.category() >= 0 ? 1.0 : 0.0;
        features[2] = Math.log(1.0 + result.processingTimeMs()) / 10.0;
        
        // Semantic-specific patterns
        for (var i = 3; i < features.length; i++) {
            features[i] = result.confidence() * Math.sin((i * Math.PI) / features.length);
        }
        
        return features;
    }
    
    private double[] createEntityFeatures(ChannelResult result) {
        var features = new double[12];
        
        features[0] = result.confidence();
        features[1] = result.category() >= 0 ? 1.0 : 0.0;
        features[2] = Math.log(1.0 + result.processingTimeMs()) / 10.0;
        
        // Entity-specific patterns
        for (var i = 3; i < features.length; i++) {
            features[i] = result.confidence() * Math.cos((i * Math.PI) / features.length);
        }
        
        return features;
    }
    
    private double[] createSyntacticFeatures(ChannelResult result) {
        var features = new double[14];
        
        features[0] = result.confidence();
        features[1] = result.category() >= 0 ? 1.0 : 0.0;
        features[2] = Math.log(1.0 + result.processingTimeMs()) / 10.0;
        
        // Syntactic patterns
        for (var i = 3; i < features.length; i++) {
            features[i] = result.confidence() * Math.tanh((i - features.length / 2.0) / 3.0);
        }
        
        return features;
    }
    
    private double[] createGenericFeatures(ChannelResult result) {
        return new double[] {
            result.confidence(),
            result.category() >= 0 ? 1.0 : 0.0,
            Math.log(1.0 + result.processingTimeMs()),
            result.metadata().size() / 5.0
        };
    }
    
    /**
     * Normalize feature vector to unit length.
     */
    private double[] normalizeVector(double[] vector) {
        var norm = Math.sqrt(Arrays.stream(vector).map(x -> x * x).sum());
        
        if (norm > 1e-10) {
            for (var i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
        
        return vector;
    }
    
    /**
     * Compute covariance matrix.
     */
    private double[][] computeCovarianceMatrix(double[][] dataMatrix) {
        var numSamples = dataMatrix.length;
        var numFeatures = dataMatrix[0].length;
        var covariance = new double[numFeatures][numFeatures];
        
        for (var i = 0; i < numFeatures; i++) {
            for (var j = 0; j <= i; j++) {
                var sum = 0.0;
                
                for (var k = 0; k < numSamples; k++) {
                    sum += dataMatrix[k][i] * dataMatrix[k][j];
                }
                
                covariance[i][j] = covariance[j][i] = sum / (numSamples - 1);
            }
        }
        
        return covariance;
    }
    
    /**
     * Simple eigendecomposition using power iteration (for small matrices).
     */
    private EigenDecomposition computeEigenDecomposition(double[][] matrix) {
        var n = matrix.length;
        var eigenvalues = new double[n];
        var eigenvectors = new double[n][n];
        
        // For small matrices, use simplified approach
        // In production, would use EJML or Apache Commons Math
        
        try {
            // Power iteration for dominant eigenvector
            var maxIterations = 100;
            var tolerance = 1e-8;
            
            for (var eigenIndex = 0; eigenIndex < Math.min(n, targetDimensions); eigenIndex++) {
                var vector = new double[n];
                // Initialize with random values
                for (var i = 0; i < n; i++) {
                    vector[i] = Math.random() - 0.5;
                }
                
                var eigenvalue = 0.0;
                
                for (var iter = 0; iter < maxIterations; iter++) {
                    var newVector = matrixVectorMultiply(matrix, vector);
                    eigenvalue = vectorNorm(newVector);
                    
                    if (eigenvalue > tolerance) {
                        for (var i = 0; i < n; i++) {
                            newVector[i] /= eigenvalue;
                        }
                    }
                    
                    // Check convergence
                    var diff = 0.0;
                    for (var i = 0; i < n; i++) {
                        diff += Math.abs(newVector[i] - vector[i]);
                    }
                    
                    vector = newVector;
                    
                    if (diff < tolerance) {
                        break;
                    }
                }
                
                eigenvalues[eigenIndex] = eigenvalue;
                System.arraycopy(vector, 0, eigenvectors[eigenIndex], 0, n);
                
                // Deflate matrix (remove this eigenvalue/eigenvector)
                deflateMatrix(matrix, vector, eigenvalue);
            }
            
            return new EigenDecomposition(eigenvalues, eigenvectors);
        } catch (Exception e) {
            log.warn("Eigendecomposition failed: {}", e.getMessage());
            return null;
        }
    }
    
    /**
     * Matrix-vector multiplication.
     */
    private double[] matrixVectorMultiply(double[][] matrix, double[] vector) {
        var result = new double[matrix.length];
        
        for (var i = 0; i < matrix.length; i++) {
            for (var j = 0; j < vector.length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        
        return result;
    }
    
    /**
     * Vector norm (L2).
     */
    private double vectorNorm(double[] vector) {
        return Math.sqrt(Arrays.stream(vector).map(x -> x * x).sum());
    }
    
    /**
     * Deflate matrix by removing eigenvalue/eigenvector contribution.
     */
    private void deflateMatrix(double[][] matrix, double[] eigenvector, double eigenvalue) {
        var n = matrix.length;
        
        for (var i = 0; i < n; i++) {
            for (var j = 0; j < n; j++) {
                matrix[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
            }
        }
    }
    
    /**
     * Determine number of components based on variance threshold.
     */
    private int determineNumComponents(double[] eigenvalues) {
        // Sort eigenvalues in descending order
        var sortedEigenvalues = Arrays.stream(eigenvalues)
            .boxed()
            .sorted(Collections.reverseOrder())
            .mapToDouble(Double::doubleValue)
            .toArray();
        
        var totalVariance = Arrays.stream(sortedEigenvalues).sum();
        
        if (totalVariance <= 0) {
            return Math.min(targetDimensions, eigenvalues.length);
        }
        
        var cumulativeVariance = 0.0;
        var numComponents = 0;
        
        for (var i = 0; i < sortedEigenvalues.length && i < targetDimensions; i++) {
            cumulativeVariance += sortedEigenvalues[i];
            numComponents++;
            
            if (cumulativeVariance / totalVariance >= varianceThreshold) {
                break;
            }
        }
        
        return Math.max(1, numComponents);
    }
    
    /**
     * Project data onto principal components.
     */
    private double[] projectData(List<double[]> featureMatrix, int numComponents) {
        var result = new double[numComponents];
        
        if (principalComponents == null || numComponents == 0) {
            return result;
        }
        
        // Average projection across all samples
        for (var features : featureMatrix) {
            for (var comp = 0; comp < numComponents; comp++) {
                var projection = 0.0;
                
                for (var j = 0; j < Math.min(features.length, principalComponents[comp].length); j++) {
                    projection += features[j] * principalComponents[comp][j];
                }
                
                result[comp] += projection / featureMatrix.size();
            }
        }
        
        return result;
    }
    
    /**
     * Create fallback result when PCA fails.
     */
    private double[] createFallbackResult(List<double[]> featureMatrix) {
        var avgFeatures = new double[targetDimensions];
        var totalSamples = featureMatrix.size();
        
        for (var features : featureMatrix) {
            for (var i = 0; i < Math.min(avgFeatures.length, features.length); i++) {
                avgFeatures[i] += features[i] / totalSamples;
            }
        }
        
        return avgFeatures;
    }
    
    /**
     * Compute variance preserved by current PCA.
     */
    private double computeVariancePreserved() {
        if (!pcaComputed || eigenvalues == null) {
            return 0.0;
        }
        
        var totalVariance = Arrays.stream(eigenvalues).sum();
        if (totalVariance <= 0) {
            return 0.0;
        }
        
        var preservedVariance = Arrays.stream(eigenvalues)
            .limit(Math.min(targetDimensions, eigenvalues.length))
            .sum();
        
        return preservedVariance / totalVariance;
    }
    
    @Override
    public String getStrategyName() {
        return "PCA";
    }
    
    @Override
    public int getOutputDimension() {
        return targetDimensions;
    }
    
    @Override
    public int getMinimumRequiredChannels() {
        return 2; // PCA requires at least 2 samples
    }
    
    /**
     * Get target dimensions.
     */
    public int getTargetDimensions() {
        return targetDimensions;
    }
    
    /**
     * Get variance threshold.
     */
    public double getVarianceThreshold() {
        return varianceThreshold;
    }
    
    /**
     * Check if input normalization is enabled.
     */
    public boolean isNormalizeInput() {
        return normalizeInput;
    }
    
    /**
     * Check if data centering is enabled.
     */
    public boolean isCenterData() {
        return centerData;
    }
    
    /**
     * Get principal components (if computed).
     */
    public double[][] getPrincipalComponents() {
        return pcaComputed && principalComponents != null ? 
               Arrays.copyOf(principalComponents, principalComponents.length) : null;
    }
    
    /**
     * Get eigenvalues (if computed).
     */
    public double[] getEigenvalues() {
        return pcaComputed && eigenvalues != null ? 
               Arrays.copyOf(eigenvalues, eigenvalues.length) : null;
    }
    
    @Override
    public String toString() {
        return String.format("PCAFusion{targetDims=%d, varThreshold=%.2f, normalize=%s, center=%s}",
                           targetDimensions, varianceThreshold, normalizeInput, centerData);
    }
    
    /**
     * Eigendecomposition result holder.
     */
    private static class EigenDecomposition {
        final double[] eigenvalues;
        final double[][] eigenvectors;
        
        EigenDecomposition(double[] eigenvalues, double[][] eigenvectors) {
            this.eigenvalues = eigenvalues;
            this.eigenvectors = eigenvectors;
        }
    }
}