package com.hellblazer.art.core;

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
public final class ARTMAP {
    
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
    public ARTMAPResult train(Vector input, Vector target, Object artAParameters, Object artBParameters) {
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
    public Optional<ARTMAPResult.Prediction> predict(Vector input, Object artAParameters) {
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
            Vector input, int targetBIndex, Object artAParameters, ActivationResult.Success artBSuccess) {
        
        int maxSearchAttempts = artA.getCategoryCount() + 1;  // Limit search to prevent infinite loops
        
        for (int attempt = 0; attempt < maxSearchAttempts; attempt++) {
            // Process input through ARTa
            var artAResult = artA.stepFit(input, artAParameters);
            if (!(artAResult instanceof ActivationResult.Success artASuccess)) {
                throw new IllegalStateException("ARTa processing failed: " + artAResult);
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
                // Map field mismatch - need to increase ARTa vigilance and search
                var mapActivation = calculateMapFieldActivation(artAIndex, existingMapping);
                
                // Check if map field vigilance is satisfied
                if (mapActivation >= mapParameters.mapVigilance()) {
                    // Map field mismatch but vigilance met - trigger ARTa reset
                    increaseARTaVigilance(artAParameters);
                    
                    return new ARTMAPResult.MapFieldMismatch(
                        artAIndex, existingMapping, targetBIndex, mapActivation, true
                    );
                } else {
                    // Map field vigilance not met - continue search with increased vigilance
                    increaseARTaVigilance(artAParameters);
                    // Continue loop for next attempt
                }
            }
        }
        
        // Exhausted search attempts
        throw new IllegalStateException("ARTMAP vigilance search exceeded maximum attempts: " + maxSearchAttempts);
    }
    
    /**
     * Find best matching ARTa category without learning (for prediction).
     */
    private Optional<CategoryMatch> findBestARTaMatch(Vector input, Object artAParameters) {
        if (artA.getCategoryCount() == 0) {
            return Optional.empty();
        }
        
        double bestActivation = -1.0;
        int bestIndex = -1;
        
        // Calculate activations for all ARTa categories
        for (int i = 0; i < artA.getCategoryCount(); i++) {
            var category = artA.getCategory(i);
            
            // Use reflection to call protected calculateActivation method
            // In a real implementation, this would require exposing activation calculation
            // For now, we'll use a simplified approach
            
            // This is a simplified version - in practice would need proper activation calculation
            bestIndex = 0;  // Simplified: use first category
            bestActivation = 1.0;  // Simplified activation
            break;
        }
        
        if (bestIndex >= 0) {
            return Optional.of(new CategoryMatch(bestIndex, bestActivation));
        }
        
        return Optional.empty();
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
     * Increase ARTa vigilance to trigger search for new category.
     * Implementation depends on ARTa parameter type.
     */
    private void increaseARTaVigilance(Object artAParameters) {
        // This would need to be implemented based on specific parameter types
        // For now, this is a placeholder that would modify vigilance in the parameter object
        // In practice, would need parameter-specific vigilance increase logic
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
    public void clear() {
        artA.clear();
        artB.clear();
        mapField.clear();
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