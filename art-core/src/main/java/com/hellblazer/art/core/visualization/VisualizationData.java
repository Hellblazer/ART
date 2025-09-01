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
package com.hellblazer.art.core.visualization;

import java.util.List;
import java.util.Map;

/**
 * Container for visualization data from ART algorithms.
 * 
 * This class provides a structured way to pass visualization information
 * from ART algorithms to visualization components. It's designed to be
 * lightweight and flexible to support different types of visualizations.
 * 
 * @author Hal Hildebrand
 */
public class VisualizationData {
    
    /**
     * Represents a cluster for visualization purposes.
     */
    public record Cluster(
        int id,
        double[] center,
        double[] bounds,
        int pointCount,
        double activation,
        Map<String, Object> properties
    ) {}
    
    /**
     * Represents a data point for visualization.
     */
    public record DataPoint(
        double[] coordinates,
        int clusterId,
        double activation,
        Map<String, Object> properties
    ) {}
    
    /**
     * Represents weight vector data for visualization.
     */
    public record WeightVector(
        int categoryIndex,
        double[] weights,
        double vigilance,
        int patternCount
    ) {}
    
    private final List<Cluster> clusters;
    private final List<DataPoint> dataPoints;
    private final List<WeightVector> weightVectors;
    private final double[][] weightMatrix;
    private final Map<String, Object> metadata;
    private final String algorithmType;
    private final long timestamp;
    
    /**
     * Create visualization data.
     * 
     * @param clusters list of clusters
     * @param dataPoints list of data points
     * @param weightVectors list of weight vectors
     * @param weightMatrix weight matrix (may be null)
     * @param metadata additional metadata
     * @param algorithmType type of ART algorithm
     */
    public VisualizationData(
        List<Cluster> clusters,
        List<DataPoint> dataPoints, 
        List<WeightVector> weightVectors,
        double[][] weightMatrix,
        Map<String, Object> metadata,
        String algorithmType
    ) {
        this.clusters = clusters != null ? List.copyOf(clusters) : List.of();
        this.dataPoints = dataPoints != null ? List.copyOf(dataPoints) : List.of();
        this.weightVectors = weightVectors != null ? List.copyOf(weightVectors) : List.of();
        this.weightMatrix = weightMatrix != null ? cloneMatrix(weightMatrix) : null;
        this.metadata = metadata != null ? Map.copyOf(metadata) : Map.of();
        this.algorithmType = algorithmType != null ? algorithmType : "Unknown";
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * Get the clusters.
     * 
     * @return list of clusters
     */
    public List<Cluster> getClusters() {
        return clusters;
    }
    
    /**
     * Get the data points.
     * 
     * @return list of data points
     */
    public List<DataPoint> getDataPoints() {
        return dataPoints;
    }
    
    /**
     * Get the weight vectors.
     * 
     * @return list of weight vectors
     */
    public List<WeightVector> getWeightVectors() {
        return weightVectors;
    }
    
    /**
     * Get the weight matrix.
     * 
     * @return weight matrix, or null if not available
     */
    public double[][] getWeightMatrix() {
        return weightMatrix != null ? cloneMatrix(weightMatrix) : null;
    }
    
    /**
     * Get metadata.
     * 
     * @return metadata map
     */
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    /**
     * Get the algorithm type.
     * 
     * @return algorithm type string
     */
    public String getAlgorithmType() {
        return algorithmType;
    }
    
    /**
     * Get the timestamp when this data was created.
     * 
     * @return timestamp in milliseconds
     */
    public long getTimestamp() {
        return timestamp;
    }
    
    /**
     * Get the number of categories/clusters.
     * 
     * @return cluster count
     */
    public int getClusterCount() {
        return clusters.size();
    }
    
    /**
     * Get the dimensionality of the data.
     * 
     * @return number of dimensions, or 0 if no data points
     */
    public int getDimensionality() {
        if (!dataPoints.isEmpty()) {
            return dataPoints.get(0).coordinates().length;
        }
        if (!clusters.isEmpty() && clusters.get(0).center() != null) {
            return clusters.get(0).center().length;
        }
        return 0;
    }
    
    /**
     * Check if this data contains cluster information.
     * 
     * @return true if clusters are available
     */
    public boolean hasClusters() {
        return !clusters.isEmpty();
    }
    
    /**
     * Check if this data contains data points.
     * 
     * @return true if data points are available
     */
    public boolean hasDataPoints() {
        return !dataPoints.isEmpty();
    }
    
    /**
     * Check if this data contains weight vectors.
     * 
     * @return true if weight vectors are available
     */
    public boolean hasWeightVectors() {
        return !weightVectors.isEmpty();
    }
    
    /**
     * Check if this data contains a weight matrix.
     * 
     * @return true if weight matrix is available
     */
    public boolean hasWeightMatrix() {
        return weightMatrix != null;
    }
    
    /**
     * Clone a matrix for safe copying.
     * 
     * @param matrix the matrix to clone
     * @return cloned matrix
     */
    private double[][] cloneMatrix(double[][] matrix) {
        if (matrix == null) return null;
        
        var clone = new double[matrix.length][];
        for (int i = 0; i < matrix.length; i++) {
            if (matrix[i] != null) {
                clone[i] = matrix[i].clone();
            }
        }
        return clone;
    }
    
    /**
     * Builder for creating VisualizationData instances.
     */
    public static class Builder {
        private List<Cluster> clusters = List.of();
        private List<DataPoint> dataPoints = List.of();
        private List<WeightVector> weightVectors = List.of();
        private double[][] weightMatrix = null;
        private Map<String, Object> metadata = Map.of();
        private String algorithmType = "Unknown";
        
        public Builder clusters(List<Cluster> clusters) {
            this.clusters = clusters;
            return this;
        }
        
        public Builder dataPoints(List<DataPoint> dataPoints) {
            this.dataPoints = dataPoints;
            return this;
        }
        
        public Builder weightVectors(List<WeightVector> weightVectors) {
            this.weightVectors = weightVectors;
            return this;
        }
        
        public Builder weightMatrix(double[][] weightMatrix) {
            this.weightMatrix = weightMatrix;
            return this;
        }
        
        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }
        
        public Builder algorithmType(String algorithmType) {
            this.algorithmType = algorithmType;
            return this;
        }
        
        public VisualizationData build() {
            return new VisualizationData(
                clusters, dataPoints, weightVectors, 
                weightMatrix, metadata, algorithmType
            );
        }
    }
    
    /**
     * Create a new builder.
     * 
     * @return new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }
}