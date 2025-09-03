package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.artmap.ARTMAPParameters;
import java.util.Objects;

/**
 * Immutable parameters for VectorizedSalienceARTMAP with salience-aware mapping.
 * Extends standard ARTMAP parameters with salience-specific configuration.
 */
public record VectorizedSalienceARTMAPParameters(
    double mapVigilance,
    double baselineVigilance,
    double vigilanceIncrement,
    double maxVigilance,
    boolean enableMatchTracking,
    boolean enableParallelSearch,
    int maxSearchAttempts,
    VectorizedSalienceParameters artAParams,
    VectorizedSalienceParameters artBParams,
    boolean enableCrossSalienceAdaptation,
    double salienceTransferRate,
    SalienceMappingStrategy mappingStrategy
) {
    
    /**
     * Strategies for mapping salience between ART modules
     */
    public enum SalienceMappingStrategy {
        WEIGHTED_AVERAGE,   // Average weighted by category activation
        MAX_SALIENCE,       // Use maximum salience value
        ADAPTIVE           // Dynamically adjust based on performance
    }
    
    /**
     * Constructor with validation
     */
    public VectorizedSalienceARTMAPParameters {
        if (mapVigilance < 0.0 || mapVigilance > 1.0) {
            throw new IllegalArgumentException("Map vigilance must be in [0,1], got: " + mapVigilance);
        }
        if (baselineVigilance < 0.0 || baselineVigilance > 1.0) {
            throw new IllegalArgumentException("Baseline vigilance must be in [0,1], got: " + baselineVigilance);
        }
        if (vigilanceIncrement < 0.0 || vigilanceIncrement > 1.0) {
            throw new IllegalArgumentException("Vigilance increment must be in [0,1], got: " + vigilanceIncrement);
        }
        if (maxVigilance < 0.0 || maxVigilance > 1.0) {
            throw new IllegalArgumentException("Max vigilance must be in [0,1], got: " + maxVigilance);
        }
        if (maxVigilance < mapVigilance) {
            throw new IllegalArgumentException("Max vigilance must be >= map vigilance");
        }
        if (maxSearchAttempts < 1) {
            throw new IllegalArgumentException("Max search attempts must be positive, got: " + maxSearchAttempts);
        }
        if (salienceTransferRate < 0.0 || salienceTransferRate > 1.0) {
            throw new IllegalArgumentException("Salience transfer rate must be in [0,1], got: " + salienceTransferRate);
        }
        
        Objects.requireNonNull(artAParams, "ART-A parameters cannot be null");
        Objects.requireNonNull(artBParams, "ART-B parameters cannot be null");
        Objects.requireNonNull(mappingStrategy, "Mapping strategy cannot be null");
    }
    
    /**
     * Create default parameters
     */
    public static VectorizedSalienceARTMAPParameters defaults() {
        return new VectorizedSalienceARTMAPParameters(
            0.9,                                    // mapVigilance
            0.0,                                    // baselineVigilance
            0.05,                                   // vigilanceIncrement
            0.95,                                   // maxVigilance
            true,                                   // enableMatchTracking
            false,                                  // enableParallelSearch
            10,                                     // maxSearchAttempts
            VectorizedSalienceParameters.createDefault(),  // artAParams
            VectorizedSalienceParameters.createDefault(),  // artBParams
            false,                                  // enableCrossSalienceAdaptation
            0.01,                                   // salienceTransferRate
            SalienceMappingStrategy.WEIGHTED_AVERAGE  // mappingStrategy
        );
    }
    
    /**
     * Convert to base ARTMAPParameters
     */
    public ARTMAPParameters toARTMAPParameters() {
        return new ARTMAPParameters(
            mapVigilance,
            baselineVigilance
        );
    }
    
    /**
     * Builder for fluent parameter configuration
     */
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        private double mapVigilance = 0.9;
        private double baselineVigilance = 0.0;
        private double vigilanceIncrement = 0.05;
        private double maxVigilance = 0.95;
        private boolean enableMatchTracking = true;
        private boolean enableParallelSearch = false;
        private int maxSearchAttempts = 10;
        private VectorizedSalienceParameters artAParams = VectorizedSalienceParameters.createDefault();
        private VectorizedSalienceParameters artBParams = VectorizedSalienceParameters.createDefault();
        private boolean enableCrossSalienceAdaptation = false;
        private double salienceTransferRate = 0.01;
        private SalienceMappingStrategy mappingStrategy = SalienceMappingStrategy.WEIGHTED_AVERAGE;
        
        public Builder mapVigilance(double mapVigilance) {
            this.mapVigilance = mapVigilance;
            return this;
        }
        
        public Builder baselineVigilance(double baselineVigilance) {
            this.baselineVigilance = baselineVigilance;
            return this;
        }
        
        public Builder vigilanceIncrement(double vigilanceIncrement) {
            this.vigilanceIncrement = vigilanceIncrement;
            return this;
        }
        
        public Builder maxVigilance(double maxVigilance) {
            this.maxVigilance = maxVigilance;
            return this;
        }
        
        public Builder enableMatchTracking(boolean enableMatchTracking) {
            this.enableMatchTracking = enableMatchTracking;
            return this;
        }
        
        public Builder enableParallelSearch(boolean enableParallelSearch) {
            this.enableParallelSearch = enableParallelSearch;
            return this;
        }
        
        public Builder maxSearchAttempts(int maxSearchAttempts) {
            this.maxSearchAttempts = maxSearchAttempts;
            return this;
        }
        
        public Builder artAParams(VectorizedSalienceParameters artAParams) {
            this.artAParams = artAParams;
            return this;
        }
        
        public Builder artBParams(VectorizedSalienceParameters artBParams) {
            this.artBParams = artBParams;
            return this;
        }
        
        public Builder enableCrossSalienceAdaptation(boolean enableCrossSalienceAdaptation) {
            this.enableCrossSalienceAdaptation = enableCrossSalienceAdaptation;
            return this;
        }
        
        public Builder salienceTransferRate(double salienceTransferRate) {
            this.salienceTransferRate = salienceTransferRate;
            return this;
        }
        
        public Builder mappingStrategy(SalienceMappingStrategy mappingStrategy) {
            this.mappingStrategy = mappingStrategy;
            return this;
        }
        
        public VectorizedSalienceARTMAPParameters build() {
            return new VectorizedSalienceARTMAPParameters(
                mapVigilance,
                baselineVigilance,
                vigilanceIncrement,
                maxVigilance,
                enableMatchTracking,
                enableParallelSearch,
                maxSearchAttempts,
                artAParams,
                artBParams,
                enableCrossSalienceAdaptation,
                salienceTransferRate,
                mappingStrategy
            );
        }
    }
    
    // With methods for immutable updates
    public VectorizedSalienceARTMAPParameters withMapVigilance(double newMapVigilance) {
        return new VectorizedSalienceARTMAPParameters(
            newMapVigilance,
            baselineVigilance,
            vigilanceIncrement,
            maxVigilance,
            enableMatchTracking,
            enableParallelSearch,
            maxSearchAttempts,
            artAParams,
            artBParams,
            enableCrossSalienceAdaptation,
            salienceTransferRate,
            mappingStrategy
        );
    }
    
    public VectorizedSalienceARTMAPParameters withBaselineVigilance(double newBaselineVigilance) {
        return new VectorizedSalienceARTMAPParameters(
            mapVigilance,
            newBaselineVigilance,
            vigilanceIncrement,
            maxVigilance,
            enableMatchTracking,
            enableParallelSearch,
            maxSearchAttempts,
            artAParams,
            artBParams,
            enableCrossSalienceAdaptation,
            salienceTransferRate,
            mappingStrategy
        );
    }
    
    public VectorizedSalienceARTMAPParameters withVigilanceIncrement(double newVigilanceIncrement) {
        return new VectorizedSalienceARTMAPParameters(
            mapVigilance,
            baselineVigilance,
            newVigilanceIncrement,
            maxVigilance,
            enableMatchTracking,
            enableParallelSearch,
            maxSearchAttempts,
            artAParams,
            artBParams,
            enableCrossSalienceAdaptation,
            salienceTransferRate,
            mappingStrategy
        );
    }
    
    public VectorizedSalienceARTMAPParameters withEnableMatchTracking(boolean newEnableMatchTracking) {
        return new VectorizedSalienceARTMAPParameters(
            mapVigilance,
            baselineVigilance,
            vigilanceIncrement,
            maxVigilance,
            newEnableMatchTracking,
            enableParallelSearch,
            maxSearchAttempts,
            artAParams,
            artBParams,
            enableCrossSalienceAdaptation,
            salienceTransferRate,
            mappingStrategy
        );
    }
    
    public VectorizedSalienceARTMAPParameters withEnableCrossSalienceAdaptation(boolean newEnableCrossSalienceAdaptation) {
        return new VectorizedSalienceARTMAPParameters(
            mapVigilance,
            baselineVigilance,
            vigilanceIncrement,
            maxVigilance,
            enableMatchTracking,
            enableParallelSearch,
            maxSearchAttempts,
            artAParams,
            artBParams,
            newEnableCrossSalienceAdaptation,
            salienceTransferRate,
            mappingStrategy
        );
    }
    
    public VectorizedSalienceARTMAPParameters withSalienceTransferRate(double newSalienceTransferRate) {
        return new VectorizedSalienceARTMAPParameters(
            mapVigilance,
            baselineVigilance,
            vigilanceIncrement,
            maxVigilance,
            enableMatchTracking,
            enableParallelSearch,
            maxSearchAttempts,
            artAParams,
            artBParams,
            enableCrossSalienceAdaptation,
            newSalienceTransferRate,
            mappingStrategy
        );
    }
}