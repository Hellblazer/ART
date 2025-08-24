package com.hellblazer.art.core;

import java.util.Objects;

/**
 * Sealed interface representing the result of ART activation computation.
 * Used in the template method pattern to handle activation outcomes in a type-safe manner.
 */
public sealed interface ActivationResult permits ActivationResult.Success, ActivationResult.NoMatch {
    
    /**
     * Successful activation with category selection.
     * @param categoryIndex the index of the selected category
     * @param activationValue the computed activation value
     * @param updatedWeight the updated weight after learning (if applicable)
     */
    record Success(int categoryIndex, double activationValue, WeightVector updatedWeight) implements ActivationResult {
        public Success {
            if (categoryIndex < 0) {
                throw new IllegalArgumentException("Category index must be non-negative, got: " + categoryIndex);
            }
            if (Double.isNaN(activationValue) || Double.isInfinite(activationValue)) {
                throw new IllegalArgumentException("Activation value must be finite, got: " + activationValue);
            }
            Objects.requireNonNull(updatedWeight, "Updated weight cannot be null");
        }
    }
    
    /**
     * No matching category found (all categories failed vigilance test).
     * A new category should be created when this result is returned.
     */
    record NoMatch() implements ActivationResult {
        // Singleton pattern for efficiency
        public static final NoMatch INSTANCE = new NoMatch();
        
        /**
         * Factory method for creating NoMatch instances.
         * @return the singleton NoMatch instance
         */
        public static NoMatch instance() {
            return INSTANCE;
        }
    }
    
    /**
     * Convenient pattern matching for activation results.
     * @param onSuccess handler for successful activation
     * @param onNoMatch handler for no match case
     * @param <T> the return type
     * @return the result of the appropriate handler
     */
    default <T> T match(java.util.function.Function<Success, T> onSuccess,
                       java.util.function.Supplier<T> onNoMatch) {
        return switch (this) {
            case Success success -> onSuccess.apply(success);
            case NoMatch noMatch -> onNoMatch.get();
        };
    }
}