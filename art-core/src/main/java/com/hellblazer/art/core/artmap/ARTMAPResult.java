package com.hellblazer.art.core.artmap;

/**
 * Sealed interface representing the result of ARTMAP supervised learning operations.
 * ARTMAP can succeed with learning, succeed with prediction, or fail due to map field mismatch.
 */
public sealed interface ARTMAPResult 
    permits ARTMAPResult.Success, ARTMAPResult.Prediction, ARTMAPResult.MapFieldMismatch {
    
    /**
     * Successful ARTMAP training operation.
     * Indicates that both ARTa and ARTb processed successfully and map field resonance occurred.
     * 
     * @param artAIndex the category index in ARTa that was activated/created
     * @param artBIndex the category index in ARTb that was activated/created  
     * @param artAActivation the activation value from ARTa
     * @param artBActivation the activation value from ARTb
     * @param mapFieldActivation the map field resonance value
     * @param wasNewMapping whether a new mapping was created in the map field
     */
    public record Success(
        int artAIndex,
        int artBIndex, 
        double artAActivation,
        double artBActivation,
        double mapFieldActivation,
        boolean wasNewMapping
    ) implements ARTMAPResult {}
    
    /**
     * Successful ARTMAP prediction operation (no learning).
     * Used when ARTMAP is run in prediction-only mode.
     * 
     * @param artAIndex the category index in ARTa that was activated
     * @param predictedBIndex the predicted category index in ARTb based on map field
     * @param artAActivation the activation value from ARTa
     * @param confidence the confidence in the prediction based on map field strength
     */
    public record Prediction(
        int artAIndex,
        int predictedBIndex,
        double artAActivation, 
        double confidence
    ) implements ARTMAPResult {}
    
    /**
     * Map field mismatch during training.
     * Occurs when ARTa category maps to a different ARTb category than the current target.
     * Triggers ARTa vigilance increase and search for new category.
     * 
     * @param artAIndex the ARTa category that caused the mismatch
     * @param expectedBIndex the ARTb category from the map field
     * @param actualBIndex the ARTb category from current target
     * @param mapFieldActivation the map field activation that failed
     * @param resetTriggered whether ARTa reset/search was triggered
     */
    public record MapFieldMismatch(
        int artAIndex,
        int expectedBIndex, 
        int actualBIndex,
        double mapFieldActivation,
        boolean resetTriggered
    ) implements ARTMAPResult {}
    
    /**
     * Check if this result represents successful training.
     * @return true if this is a Success result
     */
    default boolean isSuccess() {
        return this instanceof Success;
    }
    
    /**
     * Check if this result represents successful prediction.
     * @return true if this is a Prediction result  
     */
    default boolean isPrediction() {
        return this instanceof Prediction;
    }
    
    /**
     * Check if this result represents a map field mismatch.
     * @return true if this is a MapFieldMismatch result
     */
    default boolean isMapFieldMismatch() {
        return this instanceof MapFieldMismatch;
    }
    
    /**
     * Get the ARTa category index for any result type.
     * @return the ARTa category index
     */
    default int getArtAIndex() {
        return switch (this) {
            case Success s -> s.artAIndex;
            case Prediction p -> p.artAIndex;
            case MapFieldMismatch m -> m.artAIndex;
        };
    }
    
    /**
     * Get the ARTa activation for any result type.
     * @return the ARTa activation value
     */
    default double getArtAActivation() {
        return switch (this) {
            case Success s -> s.artAActivation;
            case Prediction p -> p.artAActivation;
            case MapFieldMismatch m -> Double.NaN; // No activation during mismatch
        };
    }
}