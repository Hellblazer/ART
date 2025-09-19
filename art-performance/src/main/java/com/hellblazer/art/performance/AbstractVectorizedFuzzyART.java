/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 * 
 * This file is part of Java ART Neural Networks.
 * 
 * Java ART Neural Networks is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Java ART Neural Networks is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with Java ART Neural Networks. If not, see <https://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.performance;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.performance.algorithms.VectorizedFuzzyWeight;
import com.hellblazer.art.performance.algorithms.VectorizedParameters;
import com.hellblazer.art.performance.algorithms.VectorizedPerformanceStats;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;

/**
 * Specialized base class for FuzzyART-style vectorized algorithms.
 * 
 * Provides common functionality for algorithms that use:
 * - Complement coding [x, 1-x]
 * - Fuzzy set operations (min, max)
 * - Choice and vigilance parameters
 * - FuzzyART activation and matching semantics
 * 
 * This eliminates the common FuzzyART patterns from individual implementations,
 * allowing them to focus only on their unique algorithmic differences.
 * 
 * Algorithms extending this class: VectorizedFuzzyART, VectorizedBinaryFuzzyART,
 * VectorizedART, VectorizedDualVigilanceART, etc.
 */
public abstract class AbstractVectorizedFuzzyART extends AbstractVectorizedART<VectorizedPerformanceStats, VectorizedParameters> {
    
    private static final Logger log = LoggerFactory.getLogger(AbstractVectorizedFuzzyART.class);
    
    protected AbstractVectorizedFuzzyART(VectorizedParameters defaultParameters) {
        super(defaultParameters);
    }
    
    // === BaseART Integration ===
    
    @Override
    protected final double calculateActivation(Pattern input, WeightVector weight, VectorizedParameters parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(weight, "Weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        var vWeight = convertToVectorizedFuzzyWeight(weight);
        trackVectorOperation();
        return computeVectorizedActivation(input, vWeight, parameters);
    }
    
    @Override
    protected final MatchResult checkVigilance(Pattern input, WeightVector weight, VectorizedParameters parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(weight, "Weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        var vWeight = convertToVectorizedFuzzyWeight(weight);
        trackMatchOperation();
        return computeVectorizedVigilance(input, vWeight, parameters);
    }
    
    @Override
    protected final WeightVector updateWeights(Pattern input, WeightVector weight, VectorizedParameters parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(weight, "Weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        var vWeight = convertToVectorizedFuzzyWeight(weight);
        trackVectorOperation();
        return computeVectorizedWeightUpdate(input, vWeight, parameters);
    }
    
    @Override
    protected final WeightVector createInitialWeight(Pattern input, VectorizedParameters parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        trackVectorOperation();
        return createVectorizedWeightVector(input, parameters);
    }
    
    // === Template Methods for FuzzyART Algorithms ===
    
    /**
     * Compute vectorized activation using algorithm-specific logic.
     * Base implementation provides standard FuzzyART activation.
     * 
     * @param input the input pattern
     * @param weight the category weight
     * @param parameters the algorithm parameters
     * @return the activation value
     */
    protected double computeVectorizedActivation(Pattern input, VectorizedFuzzyWeight weight, VectorizedParameters parameters) {
        var complementInput = VectorizedFuzzyWeight.getComplementCoded(input);
        var inputArray = getCachedFloatArray(complementInput);
        var weightDoubleArray = weight.getWeights();

        // Convert double[] to float[] for SIMD operations
        var weightArray = new float[weightDoubleArray.length];
        for (int j = 0; j < weightDoubleArray.length; j++) {
            weightArray[j] = (float) weightDoubleArray[j];
        }

        // Vectorized fuzzy intersection: min(input, weight)
        var intersectionSum = 0.0f;
        var weightSum = 0.0f;

        int vectorLength = SPECIES.length();
        int i = 0;

        // Process vectors
        for (; i <= inputArray.length - vectorLength; i += vectorLength) {
            var inputVec = FloatVector.fromArray(SPECIES, inputArray, i);
            var weightVec = FloatVector.fromArray(SPECIES, weightArray, i);
            
            // Fuzzy intersection: min(input, weight)
            var intersection = inputVec.min(weightVec);
            intersectionSum += intersection.reduceLanes(VectorOperators.ADD);
            weightSum += weightVec.reduceLanes(VectorOperators.ADD);
        }
        
        // Handle remaining elements
        for (; i < inputArray.length; i++) {
            var intersection = Math.min(inputArray[i], weightArray[i]);
            intersectionSum += intersection;
            weightSum += weightArray[i];
        }
        
        // FuzzyART activation: |I ∩ W| / (α + |W|)
        return intersectionSum / (parameters.alpha() + weightSum);
    }
    
    /**
     * Compute vectorized vigilance test using algorithm-specific logic.
     * Base implementation provides standard FuzzyART vigilance.
     * 
     * @param input the input pattern
     * @param weight the category weight
     * @param parameters the algorithm parameters
     * @return the vigilance test result
     */
    protected MatchResult computeVectorizedVigilance(Pattern input, VectorizedFuzzyWeight weight, VectorizedParameters parameters) {
        var complementInput = VectorizedFuzzyWeight.getComplementCoded(input);
        var inputArray = getCachedFloatArray(complementInput);
        var weightDoubleArray = weight.getWeights();

        // Convert double[] to float[] for SIMD operations
        var weightArray = new float[weightDoubleArray.length];
        for (int j = 0; j < weightDoubleArray.length; j++) {
            weightArray[j] = (float) weightDoubleArray[j];
        }

        // Vectorized vigilance: |I ∩ W| / |I| >= ρ
        var intersectionSum = 0.0f;
        var inputSum = 0.0f;

        int vectorLength = SPECIES.length();
        int i = 0;

        // Process vectors
        for (; i <= inputArray.length - vectorLength; i += vectorLength) {
            var inputVec = FloatVector.fromArray(SPECIES, inputArray, i);
            var weightVec = FloatVector.fromArray(SPECIES, weightArray, i);
            
            var intersection = inputVec.min(weightVec);
            intersectionSum += intersection.reduceLanes(VectorOperators.ADD);
            inputSum += inputVec.reduceLanes(VectorOperators.ADD);
        }
        
        // Handle remaining elements
        for (; i < inputArray.length; i++) {
            var intersection = Math.min(inputArray[i], weightArray[i]);
            intersectionSum += intersection;
            inputSum += inputArray[i];
        }
        
        var match = intersectionSum / inputSum;
        
        // Special handling for one-hot encoded vectors (commonly used in supervised ARTMAP)
        // One-hot vectors have exactly one 1.0 and rest 0.0, which after complement coding
        // results in high similarity between different classes. We need stricter matching.
        if (isLikelyOneHotEncoded(input)) {
            // For one-hot vectors, use exact matching on the non-complement part
            var originalDim = input.dimension();
            var exactMatch = 0.0f;
            
            // Check only the original dimensions (not complement part)
            for (int j = 0; j < originalDim && j < weightDoubleArray.length / 2; j++) {
                if (input.get(j) > 0.5 && weightDoubleArray[j] > 0.5) {
                    exactMatch = 1.0f;
                    break;
                } else if (input.get(j) > 0.5 && weightDoubleArray[j] <= 0.5) {
                    // Mismatch on the "hot" element
                    exactMatch = 0.0f;
                    break;
                }
            }
            
            // Use exact match for one-hot vectors
            match = exactMatch;
        }
        
        return match >= parameters.vigilanceThreshold() ?
               new MatchResult.Accepted(match, parameters.vigilanceThreshold()) :
               new MatchResult.Rejected(match, parameters.vigilanceThreshold());
    }
    
    /**
     * Compute vectorized weight update using algorithm-specific logic.
     * Base implementation provides standard FuzzyART weight update.
     * 
     * @param input the input pattern
     * @param weight the current category weight
     * @param parameters the algorithm parameters
     * @return the updated weight vector
     */
    protected VectorizedFuzzyWeight computeVectorizedWeightUpdate(Pattern input, VectorizedFuzzyWeight weight, VectorizedParameters parameters) {
        var complementInput = VectorizedFuzzyWeight.getComplementCoded(input);
        var inputArray = getCachedFloatArray(complementInput);
        var weightDoubleArray = weight.getWeights();
        var newWeights = new double[weightDoubleArray.length];

        // Convert double[] to float[] for SIMD operations
        var weightArray = new float[weightDoubleArray.length];
        for (int j = 0; j < weightDoubleArray.length; j++) {
            weightArray[j] = (float) weightDoubleArray[j];
        }

        // Vectorized weight update: W_new = β * min(I, W_old) + (1-β) * W_old
        var beta = (float) parameters.learningRate();
        var oneMinusBeta = 1.0f - beta;

        int vectorLength = SPECIES.length();
        int i = 0;

        // Process vectors
        for (; i <= inputArray.length - vectorLength; i += vectorLength) {
            var inputVec = FloatVector.fromArray(SPECIES, inputArray, i);
            var weightVec = FloatVector.fromArray(SPECIES, weightArray, i);
            
            var intersection = inputVec.min(weightVec);
            var newWeightVec = intersection.mul(beta).add(weightVec.mul(oneMinusBeta));
            
            // Convert float results back to double array
            var floatResults = new float[vectorLength];
            newWeightVec.intoArray(floatResults, 0);
            for (int k = 0; k < vectorLength && i + k < newWeights.length; k++) {
                newWeights[i + k] = floatResults[k];
            }
        }
        
        // Handle remaining elements
        for (; i < inputArray.length; i++) {
            var intersection = Math.min(inputArray[i], weightArray[i]);
            newWeights[i] = beta * intersection + oneMinusBeta * weightArray[i];
        }
        
        return new VectorizedFuzzyWeight(newWeights, weight.getOriginalDimension(),
                                       System.currentTimeMillis(), 1);
    }
    
    /**
     * Create new vectorized weight vector from input pattern.
     * Base implementation creates complement-coded FuzzyART weight.
     * 
     * @param input the input pattern
     * @param parameters the algorithm parameters
     * @return the new weight vector
     */
    protected VectorizedFuzzyWeight createVectorizedWeightVector(Pattern input, VectorizedParameters parameters) {
        var complementInput = VectorizedFuzzyWeight.getComplementCoded(input);
        var weights = new double[complementInput.dimension()];
        
        // Initialize with input pattern (fast learning)
        for (int i = 0; i < complementInput.dimension(); i++) {
            weights[i] = complementInput.get(i);
        }
        
        return new VectorizedFuzzyWeight(weights, input.dimension(), System.currentTimeMillis(), 1);
    }
    
    // === Utility Methods ===
    
    /**
     * Check if a pattern is likely one-hot encoded.
     * One-hot patterns have exactly one element close to 1.0 and rest close to 0.0.
     */
    private boolean isLikelyOneHotEncoded(Pattern pattern) {
        int hotCount = 0;
        int zeroCount = 0;
        
        for (int i = 0; i < pattern.dimension(); i++) {
            var val = pattern.get(i);
            if (val > 0.9) {
                hotCount++;
            } else if (val < 0.1) {
                zeroCount++;
            }
        }
        
        // One-hot has exactly one hot element and rest are cold
        return hotCount == 1 && zeroCount == pattern.dimension() - 1;
    }
    
    /**
     * Convert WeightVector to VectorizedFuzzyWeight for compatibility with BaseART.
     */
    private VectorizedFuzzyWeight convertToVectorizedFuzzyWeight(WeightVector weight) {
        if (weight instanceof VectorizedFuzzyWeight vWeight) {
            return vWeight;
        }
        
        // Check if already complement-coded
        if (weight.dimension() % 2 == 0) {
            var weights = new double[weight.dimension()];
            for (int i = 0; i < weight.dimension(); i++) {
                weights[i] = weight.get(i);
            }
            return new VectorizedFuzzyWeight(weights, weight.dimension() / 2, System.currentTimeMillis(), 0);
        } else {
            // Apply complement coding
            var complementWeights = new double[weight.dimension() * 2];
            for (int i = 0; i < weight.dimension(); i++) {
                var w = Math.max(0.0, Math.min(1.0, weight.get(i)));
                complementWeights[i] = w;
                complementWeights[weight.dimension() + i] = 1.0 - w;
            }
            return new VectorizedFuzzyWeight(complementWeights, weight.dimension(), System.currentTimeMillis(), 0);
        }
    }
    
    // === Abstract Method Implementation ===
    
    @Override
    protected final VectorizedPerformanceStats createPerformanceStats(
            long vectorOps, long parallelTasks, long activations,
            long matches, long learnings, double avgTime) {
        return new VectorizedPerformanceStats(
            vectorOps,
            parallelTasks,
            avgTime,
            Runtime.getRuntime().availableProcessors(), // activeThreads
            1000, // cacheSize - placeholder
            getCategoryCount(),
            activations,
            matches,
            learnings
        );
    }
}