package com.hellblazer.art.core;

import java.util.Arrays;
import java.util.Objects;

/**
 * Weight vector implementation for ART-A (Attentional ART) that includes both
 * category weights and attention weights for dynamic feature focusing.
 * 
 * ART-A weights consist of:
 * - Category weights: Standard ART category prototype (like fuzzy weights)
 * - Attention weights: Dynamic weights indicating feature importance/discriminability
 * 
 * The attention mechanism allows the network to focus on the most discriminative
 * features for each category, improving learning efficiency and category separation.
 */
public final class ARTAWeight implements WeightVector {
    
    private final double[] categoryWeights;
    private final double[] attentionWeights;
    private final int hashCode;
    
    /**
     * Create ART-A weights with category and attention weights.
     * @param categoryWeights the category prototype weights (must not be null or empty)
     * @param attentionWeights the attention weights (must match category weights length)
     */
    public ARTAWeight(double[] categoryWeights, double[] attentionWeights) {
        Objects.requireNonNull(categoryWeights, "Category weights cannot be null");
        Objects.requireNonNull(attentionWeights, "Attention weights cannot be null");
        
        if (categoryWeights.length == 0) {
            throw new IllegalArgumentException("Category weights cannot be empty");
        }
        if (attentionWeights.length != categoryWeights.length) {
            throw new IllegalArgumentException("Attention weights length (" + attentionWeights.length + 
                ") must match category weights length (" + categoryWeights.length + ")");
        }
        
        // Validate attention weights are in valid range
        for (int i = 0; i < attentionWeights.length; i++) {
            if (attentionWeights[i] < 0.0 || attentionWeights[i] > 1.0 || 
                Double.isNaN(attentionWeights[i]) || Double.isInfinite(attentionWeights[i])) {
                throw new IllegalArgumentException("Attention weight at index " + i + 
                    " must be finite and in range [0, 1], got: " + attentionWeights[i]);
            }
        }
        
        // Validate category weights are finite
        for (int i = 0; i < categoryWeights.length; i++) {
            if (Double.isNaN(categoryWeights[i]) || Double.isInfinite(categoryWeights[i])) {
                throw new IllegalArgumentException("Category weight at index " + i + 
                    " must be finite, got: " + categoryWeights[i]);
            }
        }
        
        this.categoryWeights = categoryWeights.clone();
        this.attentionWeights = attentionWeights.clone();
        this.hashCode = Objects.hash(Arrays.hashCode(categoryWeights), Arrays.hashCode(attentionWeights));
    }
    
    /**
     * Create ART-A weights with equal attention weights.
     * @param categoryWeights the category prototype weights
     * @param uniformAttentionWeight the uniform attention weight for all dimensions
     * @return new ARTAWeight instance
     */
    public static ARTAWeight withUniformAttention(double[] categoryWeights, double uniformAttentionWeight) {
        Objects.requireNonNull(categoryWeights, "Category weights cannot be null");
        if (uniformAttentionWeight < 0.0 || uniformAttentionWeight > 1.0 || 
            Double.isNaN(uniformAttentionWeight) || Double.isInfinite(uniformAttentionWeight)) {
            throw new IllegalArgumentException("Uniform attention weight must be finite and in range [0, 1], got: " + 
                uniformAttentionWeight);
        }
        
        var attentionWeights = new double[categoryWeights.length];
        Arrays.fill(attentionWeights, uniformAttentionWeight);
        return new ARTAWeight(categoryWeights, attentionWeights);
    }
    
    /**
     * Create ART-A weights from a Pattern with uniform attention.
     * @param categoryVector the category prototype as a vector
     * @param uniformAttentionWeight the uniform attention weight
     * @return new ARTAWeight instance
     */
    public static ARTAWeight fromVector(Pattern categoryVector, double uniformAttentionWeight) {
        Objects.requireNonNull(categoryVector, "Category vector cannot be null");
        var data = new double[categoryVector.dimension()];
        for (int i = 0; i < data.length; i++) {
            data[i] = categoryVector.get(i);
        }
        return withUniformAttention(data, uniformAttentionWeight);
    }
    
    /**
     * Create ART-A weights with maximum attention (all 1.0).
     * @param categoryWeights the category prototype weights
     * @return new ARTAWeight instance with full attention
     */
    public static ARTAWeight withMaxAttention(double[] categoryWeights) {
        return withUniformAttention(categoryWeights, 1.0);
    }
    
    /**
     * Get the category weights (prototype).
     * @return a copy of the category weights array
     */
    public double[] getCategoryWeights() {
        return categoryWeights.clone();
    }
    
    /**
     * Get the attention weights.
     * @return a copy of the attention weights array
     */
    public double[] getAttentionWeights() {
        return attentionWeights.clone();
    }
    
    /**
     * Get category weight at specific index.
     * @param index the dimension index
     * @return the category weight at that index
     * @throws IndexOutOfBoundsException if index is invalid
     */
    public double getCategoryWeight(int index) {
        if (index < 0 || index >= categoryWeights.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for length " + categoryWeights.length);
        }
        return categoryWeights[index];
    }
    
    /**
     * Get attention weight at specific index.
     * @param index the dimension index
     * @return the attention weight at that index
     * @throws IndexOutOfBoundsException if index is invalid
     */
    public double getAttentionWeight(int index) {
        if (index < 0 || index >= attentionWeights.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for length " + attentionWeights.length);
        }
        return attentionWeights[index];
    }
    
    /**
     * Create new ARTAWeight with updated category weights.
     * @param newCategoryWeights the new category weights
     * @return new ARTAWeight instance with updated category weights
     */
    public ARTAWeight withCategoryWeights(double[] newCategoryWeights) {
        return new ARTAWeight(newCategoryWeights, this.attentionWeights);
    }
    
    /**
     * Create new ARTAWeight with updated attention weights.
     * @param newAttentionWeights the new attention weights
     * @return new ARTAWeight instance with updated attention weights
     */
    public ARTAWeight withAttentionWeights(double[] newAttentionWeights) {
        return new ARTAWeight(this.categoryWeights, newAttentionWeights);
    }
    
    /**
     * Create new ARTAWeight with updated category and attention weights.
     * @param newCategoryWeights the new category weights
     * @param newAttentionWeights the new attention weights
     * @return new ARTAWeight instance with both weights updated
     */
    public ARTAWeight withWeights(double[] newCategoryWeights, double[] newAttentionWeights) {
        return new ARTAWeight(newCategoryWeights, newAttentionWeights);
    }
    
    /**
     * Calculate attention-weighted distance to input vector.
     * Uses attention weights to emphasize important features.
     * @param input the input vector
     * @return attention-weighted distance
     */
    public double attentionWeightedDistance(Pattern input) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        
        if (input.dimension() != categoryWeights.length) {
            throw new IllegalArgumentException("Input dimension (" + input.dimension() + 
                ") must match weight dimension (" + categoryWeights.length + ")");
        }
        
        double sum = 0.0;
        for (int i = 0; i < categoryWeights.length; i++) {
            double diff = input.get(i) - categoryWeights[i];
            sum += attentionWeights[i] * diff * diff;  // Weighted squared difference
        }
        
        return Math.sqrt(sum);
    }
    
    /**
     * Calculate attention-weighted similarity to input vector.
     * Higher values indicate better match with attention weighting.
     * @param input the input vector
     * @return attention-weighted similarity (0 to 1)
     */
    public double attentionWeightedSimilarity(Pattern input) {
        double distance = attentionWeightedDistance(input);
        return 1.0 / (1.0 + distance);  // Convert distance to similarity
    }
    
    @Override
    public int dimension() {
        return categoryWeights.length;
    }
    
    @Override
    public double get(int index) {
        return getCategoryWeight(index);
    }
    
    @Override
    public double l1Norm() {
        double sum = 0.0;
        for (double weight : categoryWeights) {
            sum += Math.abs(weight);
        }
        return sum;
    }
    
    @Override
    public WeightVector update(Pattern input, Object parameters) {
        // ARTAWeight updates are handled by ARTA.updateWeights() method
        // Individual weights cannot update themselves without the full ART-A context
        // This method returns this weight unchanged as per WeightVector interface contract
        return this;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        var other = (ARTAWeight) obj;
        return Arrays.equals(categoryWeights, other.categoryWeights) &&
               Arrays.equals(attentionWeights, other.attentionWeights);
    }
    
    @Override
    public int hashCode() {
        return hashCode;
    }
    
    @Override
    public String toString() {
        return String.format("ARTAWeight{dim=%d, category=%s, attention=%s}",
                           categoryWeights.length,
                           Arrays.toString(categoryWeights),
                           Arrays.toString(attentionWeights));
    }
    
    /**
     * Create a compact string representation for debugging.
     * @return compact string with first few values of each weight type
     */
    public String toCompactString() {
        var maxShow = Math.min(3, categoryWeights.length);
        var catStr = new StringBuilder();
        var attStr = new StringBuilder();
        
        for (int i = 0; i < maxShow; i++) {
            if (i > 0) {
                catStr.append(",");
                attStr.append(",");
            }
            catStr.append(String.format("%.3f", categoryWeights[i]));
            attStr.append(String.format("%.3f", attentionWeights[i]));
        }
        
        if (categoryWeights.length > maxShow) {
            catStr.append("...");
            attStr.append("...");
        }
        
        return String.format("ARTAWeight{dim=%d, cat=[%s], att=[%s]}", 
                           categoryWeights.length, catStr, attStr);
    }
}