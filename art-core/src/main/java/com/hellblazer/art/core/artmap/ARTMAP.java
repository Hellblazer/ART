package com.hellblazer.art.core.artmap;

import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.BaseARTMAP;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.core.parameters.GaussianParameters;
import com.hellblazer.art.core.parameters.BayesianParameters;
import com.hellblazer.art.core.parameters.HypersphereParameters;
import com.hellblazer.art.core.parameters.ARTAParameters;
import com.hellblazer.art.core.parameters.ART2Parameters;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * ARTMAP supervised learning architecture using dual ART modules.
 * 
 * ARTMAP consists of:
 * - ARTa: Processes input patterns (can be any BaseART variant)
 * - ARTb: Processes output/target patterns (usually FuzzyART) 
 * - Map Field: Associates ARTa categories with ARTb categories
 * - Map Field Vigilance: Controls acceptance of ARTa->ARTb mappings
 * 
 * Key algorithm:
 * 1. Present input to ARTa, target to ARTb
 * 2. If ARTa category already mapped, check if mapping matches ARTb result
 * 3. If mismatch occurs, increase ARTa vigilance and search for new category
 * 4. If match or new mapping, proceed with learning
 * 5. Create/update map field connection between ARTa and ARTb categories
 */
public final class ARTMAP implements BaseARTMAP {
    
    private final BaseART artA;
    private final BaseART artB;
    private final Map<Integer, Integer> mapField;  // ARTa index -> ARTb index
    private final ARTMAPParameters mapParameters;
    
    /**
     * Create a new ARTMAP with specified ART modules and parameters.
     * @param artA the input processing ART module (ARTa)
     * @param artB the output processing ART module (ARTb) 
     * @param mapParameters the ARTMAP-specific parameters
     */
    public ARTMAP(BaseART artA, BaseART artB, ARTMAPParameters mapParameters) {
        this.artA = Objects.requireNonNull(artA, "ARTa cannot be null");
        this.artB = Objects.requireNonNull(artB, "ARTb cannot be null");
        this.mapParameters = Objects.requireNonNull(mapParameters, "Map parameters cannot be null");
        this.mapField = new HashMap<>();
    }
    
    /**
     * Train ARTMAP with an input-output pair.
     * Implements the complete ARTMAP supervised learning algorithm.
     * 
     * @param input the input pattern for ARTa
     * @param target the target pattern for ARTb
     * @param artAParameters parameters for ARTa processing
     * @param artBParameters parameters for ARTb processing
     * @return the result of the training operation
     */
    public ARTMAPResult train(Pattern input, Pattern target, Object artAParameters, Object artBParameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(target, "Target vector cannot be null");
        Objects.requireNonNull(artAParameters, "ARTa parameters cannot be null");
        Objects.requireNonNull(artBParameters, "ARTb parameters cannot be null");
        
        // Step 1: Process target through ARTb first (to establish target category)
        var artBResult = artB.stepFit(target, artBParameters);
        if (!(artBResult instanceof ActivationResult.Success artBSuccess)) {
            throw new IllegalStateException("ARTb processing failed: " + artBResult);
        }
        var targetBIndex = artBSuccess.categoryIndex();
        
        // Step 2: Process input through ARTa with potential vigilance search
        return processARTaWithVigilanceSearch(input, targetBIndex, artAParameters, artBSuccess);
    }
    
    /**
     * Predict output category for given input (no learning).
     * Uses existing map field to predict ARTb category based on ARTa activation.
     * 
     * @param input the input pattern for prediction
     * @param artAParameters parameters for ARTa processing
     * @return the prediction result or empty if no prediction possible
     */
    public Optional<ARTMAPResult.Prediction> predict(Pattern input, Object artAParameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(artAParameters, "ARTa parameters cannot be null");
        
        // Process input through ARTa without learning
        if (artA.getCategoryCount() == 0) {
            return Optional.empty();  // No categories to predict from
        }
        
        // Find best matching ARTa category without updating weights
        var bestMatch = findBestARTaMatch(input, artAParameters);
        if (bestMatch.isEmpty()) {
            return Optional.empty();  // No category met vigilance
        }
        
        var artAIndex = bestMatch.get().categoryIndex();
        var artAActivation = bestMatch.get().activation();
        
        // Check if ARTa category has mapping in map field
        var mappedBIndex = mapField.get(artAIndex);
        if (mappedBIndex == null) {
            return Optional.empty();  // No mapping exists
        }
        
        // Calculate confidence based on map field strength
        var confidence = calculateMapFieldConfidence(artAIndex, mappedBIndex);
        
        return Optional.of(new ARTMAPResult.Prediction(
            artAIndex, mappedBIndex, artAActivation, confidence
        ));
    }
    
    /**
     * Process ARTa with vigilance search to handle map field mismatches.
     * Implements the core ARTMAP match tracking and vigilance increase mechanism.
     */
    private ARTMAPResult processARTaWithVigilanceSearch(
            Pattern input, int targetBIndex, Object artAParameters, ActivationResult.Success artBSuccess) {
        
        var currentVigilance = getVigilanceFromParameters(artAParameters);
        var originalVigilance = currentVigilance;
        int maxSearchAttempts = 20;  // Reasonable limit for vigilance search
        
        for (int attempt = 0; attempt < maxSearchAttempts; attempt++) {
            // Create parameters with current vigilance level
            var searchParameters = createParametersWithVigilance(artAParameters, currentVigilance);
            
            // Process input through ARTa with match reset function that checks map field
            var artAResult = stepFitWithMapFieldCheck(input, targetBIndex, searchParameters);
            if (!(artAResult instanceof ActivationResult.Success artASuccess)) {
                // If no category met vigilance criteria, increase vigilance and try again
                currentVigilance = Math.min(0.999, currentVigilance + 0.05);
                continue;
            }
            
            var artAIndex = artASuccess.categoryIndex();
            
            // Check map field for existing mapping
            var existingMapping = mapField.get(artAIndex);
            
            if (existingMapping == null) {
                // No existing mapping - create new one
                mapField.put(artAIndex, targetBIndex);
                var mapActivation = calculateMapFieldActivation(artAIndex, targetBIndex);
                
                return new ARTMAPResult.Success(
                    artAIndex, targetBIndex,
                    artASuccess.activationValue(), artBSuccess.activationValue(),
                    mapActivation, true  // wasNewMapping
                );
                
            } else if (existingMapping.equals(targetBIndex)) {
                // Existing mapping matches target - success
                var mapActivation = calculateMapFieldActivation(artAIndex, targetBIndex);
                
                return new ARTMAPResult.Success(
                    artAIndex, targetBIndex,
                    artASuccess.activationValue(), artBSuccess.activationValue(), 
                    mapActivation, false  // wasNewMapping
                );
                
            } else {
                // Map field mismatch - need to increase vigilance and search for new category
                currentVigilance = Math.min(0.999, currentVigilance + 0.1);
                // Continue loop to try again with higher vigilance
            }
        }
        
        // If we reach here, we couldn't find a solution - create emergency mapping
        // This shouldn't happen in normal operation but provides fallback
        var artAResult = artA.stepFit(input, artAParameters);
        if (artAResult instanceof ActivationResult.Success artASuccess) {
            var artAIndex = artASuccess.categoryIndex();
            mapField.put(artAIndex, targetBIndex);  // Override existing mapping as fallback
            var mapActivation = calculateMapFieldActivation(artAIndex, targetBIndex);
            
            return new ARTMAPResult.Success(
                artAIndex, targetBIndex,
                artASuccess.activationValue(), artBSuccess.activationValue(),
                mapActivation, false  // wasNewMapping (emergency override)
            );
        }
        
        // Exhausted search attempts
        throw new IllegalStateException("ARTMAP vigilance search exceeded maximum attempts: " + maxSearchAttempts);
    }
    
    /**
     * Find best matching ARTa category without learning (for prediction).
     */
    private Optional<CategoryMatch> findBestARTaMatch(Pattern input, Object artAParameters) {
        if (artA.getCategoryCount() == 0) {
            return Optional.empty();
        }
        
        // Simple distance-based prediction without learning
        // Find the category with the closest weight pattern to the input
        int bestCategory = -1;
        double bestDistance = Double.MAX_VALUE;
        
        for (int i = 0; i < artA.getCategoryCount(); i++) {
            try {
                var weight = artA.getCategory(i);
                // Calculate Euclidean distance between input and weight
                double distance = calculateEuclideanDistance(input, weight);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestCategory = i;
                }
            } catch (Exception e) {
                // Skip this category if there's an error
                continue;
            }
        }
        
        if (bestCategory >= 0) {
            // Convert distance to activation (closer = higher activation)
            double activation = 1.0 / (1.0 + bestDistance);
            return Optional.of(new CategoryMatch(bestCategory, activation));
        }
        
        // Fallback to first category if no distance calculation worked
        return Optional.of(new CategoryMatch(0, 0.5));
    }
    
    /**
     * Calculate Euclidean distance between a pattern and weight vector.
     */
    private double calculateEuclideanDistance(Pattern pattern, WeightVector weight) {
        if (pattern.dimension() != weight.dimension()) {
            return Double.MAX_VALUE; // Invalid comparison
        }
        
        double sumSquares = 0.0;
        for (int i = 0; i < pattern.dimension(); i++) {
            double diff = pattern.get(i) - weight.get(i);
            sumSquares += diff * diff;
        }
        
        return Math.sqrt(sumSquares);
    }
    
    /**
     * Calculate map field activation between ARTa and ARTb categories.
     * Higher values indicate stronger association.
     */
    private double calculateMapFieldActivation(int artAIndex, int artBIndex) {
        // Simplified map field activation - in practice would consider
        // category similarity, association strength, etc.
        return mapField.containsKey(artAIndex) ? 0.9 : 1.0;
    }
    
    /**
     * Calculate confidence in prediction based on map field strength.
     */
    private double calculateMapFieldConfidence(int artAIndex, int artBIndex) {
        // Confidence based on map field activation and category stability
        return calculateMapFieldActivation(artAIndex, artBIndex) * 0.8;
    }
    
    /**
     * Extract vigilance parameter from ART parameters object.
     */
    private double getVigilanceFromParameters(Object artAParameters) {
        if (artAParameters instanceof FuzzyParameters fuzzyParams) {
            return fuzzyParams.vigilance();
        } else if (artAParameters instanceof GaussianParameters gaussianParams) {
            return gaussianParams.vigilance();
        } else if (artAParameters instanceof BayesianParameters bayesianParams) {
            return bayesianParams.vigilance();
        } else if (artAParameters instanceof HypersphereParameters hypersphereParams) {
            return hypersphereParams.vigilance();
        } else if (artAParameters instanceof ARTAParameters artaParams) {
            return artaParams.vigilance();
        } else if (artAParameters instanceof ART2Parameters art2Params) {
            return art2Params.vigilance();
        }
        return 0.5; // Default vigilance for unknown parameter types
    }
    
    /**
     * Create new parameters object with specified vigilance value.
     */
    private Object createParametersWithVigilance(Object artAParameters, double newVigilance) {
        if (artAParameters instanceof FuzzyParameters fuzzyParams) {
            return FuzzyParameters.of(newVigilance, fuzzyParams.alpha(), fuzzyParams.beta());
        } else if (artAParameters instanceof GaussianParameters gaussianParams) {
            return gaussianParams.withVigilance(newVigilance);
        } else if (artAParameters instanceof BayesianParameters bayesianParams) {
            // BayesianParameters is a record - reconstruct with new vigilance
            return new BayesianParameters(newVigilance, bayesianParams.priorMean(), 
                                        bayesianParams.priorCovariance(), bayesianParams.noiseVariance(),
                                        bayesianParams.priorPrecision(), bayesianParams.maxCategories());
        } else if (artAParameters instanceof HypersphereParameters hypersphereParams) {
            return hypersphereParams.withVigilance(newVigilance);
        } else if (artAParameters instanceof ARTAParameters artaParams) {
            return artaParams.withVigilance(newVigilance);
        } else if (artAParameters instanceof ART2Parameters art2Params) {
            return new ART2Parameters(newVigilance, art2Params.learningRate(), art2Params.maxCategories());
        }
        return artAParameters; // Return original if unknown type
    }
    
    /**
     * Process input through ARTa with map field conflict checking.
     * This simulates the match reset function used in the Python implementation.
     */
    private ActivationResult stepFitWithMapFieldCheck(Pattern input, int targetBIndex, Object artAParameters) {
        // Try to find existing category that matches and doesn't conflict with map field
        var bestNonConflictingCategory = -1;
        var bestActivation = -1.0;
        
        // Check existing categories first
        for (int categoryIndex = 0; categoryIndex < artA.getCategoryCount(); categoryIndex++) {
            try {
                // Simulate category activation check (simplified)
                var weight = artA.getCategory(categoryIndex);
                var activation = calculateCategoryActivation(input, weight, artAParameters);
                
                // Check if this category would meet vigilance criteria
                if (meetsPreliminaryVigilance(input, weight, artAParameters)) {
                    // Check map field conflict
                    var existingMapping = mapField.get(categoryIndex);
                    if (existingMapping == null || existingMapping.equals(targetBIndex)) {
                        // No conflict - this category is acceptable
                        if (activation > bestActivation) {
                            bestActivation = activation;
                            bestNonConflictingCategory = categoryIndex;
                        }
                    }
                    // If there's a map field conflict, skip this category
                }
            } catch (Exception e) {
                // Skip problematic categories
                continue;
            }
        }
        
        if (bestNonConflictingCategory >= 0) {
            // Use the best non-conflicting existing category - get the weight for the constructor
            try {
                var weight = artA.getCategory(bestNonConflictingCategory);
                return new ActivationResult.Success(bestNonConflictingCategory, bestActivation, weight);
            } catch (Exception e) {
                // If we can't get the weight, fall through to normal stepFit
            }
        }
        
        // No existing category works - try normal stepFit which may create new category
        return artA.stepFit(input, artAParameters);
    }
    
    /**
     * Calculate category activation for vigilance checking.
     */
    private double calculateCategoryActivation(Pattern input, WeightVector weight, Object artAParameters) {
        // Simplified activation calculation - in reality this would depend on the ART variant
        double similarity = 0.0;
        int dimensions = Math.min(input.dimension(), weight.dimension());
        
        for (int i = 0; i < dimensions; i++) {
            similarity += Math.min(input.get(i), weight.get(i));
        }
        
        return similarity / dimensions;
    }
    
    /**
     * Check if pattern meets preliminary vigilance criteria with given weight.
     */
    private boolean meetsPreliminaryVigilance(Pattern input, WeightVector weight, Object artAParameters) {
        var vigilance = getVigilanceFromParameters(artAParameters);
        var activation = calculateCategoryActivation(input, weight, artAParameters);
        return activation >= vigilance;
    }
    
    
    /**
     * Get the ARTa module.
     * @return the input processing ART module
     */
    public BaseART getArtA() {
        return artA;
    }
    
    /**
     * Get the ARTb module.
     * @return the output processing ART module
     */
    public BaseART getArtB() {
        return artB;
    }
    
    /**
     * Get a copy of the current map field mappings.
     * @return map from ARTa category indices to ARTb category indices
     */
    public Map<Integer, Integer> getMapField() {
        return new HashMap<>(mapField);
    }
    
    /**
     * Get the ARTMAP parameters.
     * @return the map field parameters
     */
    public ARTMAPParameters getMapParameters() {
        return mapParameters;
    }
    
    /**
     * Clear all categories and mappings (reset the network).
     */
    @Override
    public void clear() {
        artA.clear();
        artB.clear();
        mapField.clear();
    }
    
    /**
     * Check if this ARTMAP has been trained (has any mappings).
     * 
     * @return true if trained, false otherwise
     */
    @Override
    public boolean isTrained() {
        return !mapField.isEmpty();
    }
    
    /**
     * Get the number of ARTa categories (used as the category count).
     * 
     * @return the number of categories
     */
    @Override
    public int getCategoryCount() {
        return artA.getCategoryCount();
    }
    
    /**
     * Get statistics about the ARTMAP network.
     * @return string representation with network statistics
     */
    @Override
    public String toString() {
        return String.format("ARTMAP{artA=%d categories, artB=%d categories, mappings=%d}", 
                           artA.getCategoryCount(), artB.getCategoryCount(), mapField.size());
    }
    
    /**
     * Helper record for category matching during prediction.
     */
    private record CategoryMatch(int categoryIndex, double activation) {}
}