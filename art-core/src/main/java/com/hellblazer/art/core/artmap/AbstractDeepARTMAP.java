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
package com.hellblazer.art.core.artmap;

import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.BaseARTMAP;
import com.hellblazer.art.core.ScikitClusterer;
import java.util.List;

/**
 * Abstract base class for DeepARTMAP implementations providing common interface
 * for both standard and vectorized versions.
 * 
 * DeepARTMAP is a generalization of ARTMAP that allows an arbitrary number of ART 
 * modules to be used in a hierarchical learning structure. DeepARTMAP supports 
 * both supervised and unsupervised modes:
 * 
 * - Supervised Mode: Uses class labels to train SimpleARTMAP layers
 * - Unsupervised Mode: Uses ARTMAP + SimpleARTMAP chain for hierarchical clustering
 * 
 * Key Features:
 * - Multi-channel data processing (list of input matrices)
 * - Hierarchical label propagation through layers
 * - Deep label concatenation from all layers
 * - Support for arbitrary number of ART modules
 * 
 * Reference: Python implementation at reference parity/artlib/hierarchical/DeepARTMAP.py
 * Paper: "Deep ARTMAP: Generalized Hierarchical Learning with Adaptive Resonance Theory"
 * 
 * @author Hal Hildebrand
 */
public abstract class AbstractDeepARTMAP extends BaseART<DeepARTMAPParameters> implements ScikitClusterer<DeepARTMAPResult> {
    
    protected final List<BaseART> modules;
    protected final List<BaseARTMAP> layers;
    protected Boolean supervised;
    protected boolean trained;
    protected int[][] storedDeepLabels;  // Store actual deep labels from training
    protected int totalCategoryCount;
    
    /**
     * Create a new AbstractDeepARTMAP with specified ART modules.
     * 
     * @param modules the list of ART modules to use as building blocks
     * @throws IllegalArgumentException if modules is null, empty, or contains null elements
     */
    protected AbstractDeepARTMAP(List<BaseART> modules) {
        super(); // Call BaseART constructor
        if (modules == null) {
            throw new IllegalArgumentException("modules cannot be null");
        }
        
        if (modules.isEmpty()) {
            throw new IllegalArgumentException("Must provide at least one ART module");
        }
        
        // Check for null elements in modules before creating defensive copy
        for (int i = 0; i < modules.size(); i++) {
            if (modules.get(i) == null) {
                throw new IllegalArgumentException("modules cannot contain null elements");
            }
        }
        
        this.modules = List.copyOf(modules); // Defensive copy
        this.layers = new java.util.ArrayList<>();
        this.supervised = null;
        this.trained = false;
        this.storedDeepLabels = null;
        this.totalCategoryCount = 0;
    }
    
    /**
     * Get the list of ART modules used by this DeepARTMAP.
     * 
     * @return immutable list of ART modules
     */
    public final List<BaseART> getModules() {
        return modules;
    }
    
    /**
     * Get the number of ART modules.
     * 
     * @return the number of modules
     */
    public final int getModuleCount() {
        return modules.size();
    }
    
    /**
     * Get the list of trained layers.
     * 
     * @return immutable list of layers
     */
    public final List<BaseARTMAP> getLayers() {
        return List.copyOf(layers);
    }
    
    /**
     * Get the number of layers in the hierarchy.
     * 
     * @return the number of layers
     */
    public final int getLayerCount() {
        return layers.size();
    }
    
    /**
     * Check if this DeepARTMAP is in supervised mode.
     * 
     * @return true if supervised, false if unsupervised, null if not yet trained
     */
    public final Boolean isSupervised() {
        return supervised;
    }
    
    /**
     * Check if this DeepARTMAP has been trained.
     * 
     * @return true if trained, false otherwise
     */
    public final boolean isTrained() {
        return trained;
    }
    
    /**
     * Get the total number of categories across all layers.
     * 
     * @return the total category count
     */
    public final int getTotalCategoryCount() {
        return totalCategoryCount;
    }
    
    /**
     * Get the stored deep labels from training.
     * 
     * @return array of deep labels, or null if not trained
     */
    public final int[][] getStoredDeepLabels() {
        return storedDeepLabels == null ? null : storedDeepLabels.clone();
    }
    
    /**
     * Clear the DeepARTMAP state and reset for new training.
     * This extends BaseART's clear() functionality.
     */
    public void clearDeepARTMAP() {
        super.clear();
        layers.clear();
        supervised = null;
        trained = false;
        storedDeepLabels = null;
        totalCategoryCount = 0;
        
        // Clear all modules
        modules.forEach(BaseART::clear);
    }
    
    /**
     * Get a string representation of the DeepARTMAP state.
     * 
     * @return string representation
     */
    @Override
    public String toString() {
        return String.format("%s{modules: %d, layers: %d, supervised: %s, trained: %s, categories: %d}",
            getClass().getSimpleName(), getModuleCount(), getLayerCount(), 
            isSupervised(), isTrained(), getTotalCategoryCount());
    }
}