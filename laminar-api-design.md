# Laminar ART API Design

## Executive Summary

This document defines the comprehensive API and interface design for the laminar ART implementation. The design integrates with existing VectorizedARTAlgorithm interfaces, maintains SklearnWrapper compatibility, and provides both standard and vectorized implementations following patterns in art-core and art-performance modules.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                      │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐   │
│  │SklearnWrapper│  │ Visualizers │  │  Monitoring  │   │
│  └──────────────┘  └─────────────┘  └──────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                  Laminar ART Core API                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │            ILaminarCircuit Interface             │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐      │
│  │  ILayer  │  │ IPathway │  │IResonanceController│     │
│  └──────────┘  └──────────┘  └──────────────────┘      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                 Integration Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐   │
│  │ARTAlgorithm  │  │VectorizedART │  │  Adapters   │   │
│  └──────────────┘  └──────────────┘  └─────────────┘   │
└──────────────────────────────────────────────────────────┘
```

## Core Interfaces

### 1. ILaminarCircuit Interface

```java
package com.hellblazer.art.laminar.core;

import com.hellblazer.art.core.ARTAlgorithm;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.laminar.parameters.ILaminarParameters;
import java.util.List;
import java.util.Map;

/**
 * Core interface for laminar ART circuit implementations.
 * Extends ARTAlgorithm to maintain compatibility with existing infrastructure
 * while adding laminar-specific functionality.
 *
 * @param <P> Parameter type extending ILaminarParameters
 * @author Hal Hildebrand
 */
public interface ILaminarCircuit<P extends ILaminarParameters>
    extends ARTAlgorithm<P> {

    // === Layer Management ===

    /**
     * Add a layer to the circuit at specified depth.
     *
     * @param layer The layer to add
     * @param depth Laminar depth (0 = input layer, higher = deeper processing)
     * @return This circuit for fluent configuration
     */
    ILaminarCircuit<P> addLayer(ILayer layer, int depth);

    /**
     * Get layer at specified depth.
     *
     * @param depth The laminar depth
     * @return The layer at that depth, or null if not present
     */
    ILayer getLayer(int depth);

    /**
     * Get all layers organized by depth.
     *
     * @return Map of depth to layer
     */
    Map<Integer, ILayer> getLayers();

    // === Pathway Management ===

    /**
     * Connect layers with a pathway.
     *
     * @param pathway The pathway to add
     * @return This circuit for fluent configuration
     */
    ILaminarCircuit<P> connectLayers(IPathway pathway);

    /**
     * Get all pathways in the circuit.
     *
     * @return List of all pathways
     */
    List<IPathway> getPathways();

    /**
     * Get pathways connected to a specific layer.
     *
     * @param layerId The layer identifier
     * @return List of connected pathways
     */
    List<IPathway> getPathwaysForLayer(String layerId);

    // === Resonance Control ===

    /**
     * Set the resonance controller for the circuit.
     *
     * @param controller The resonance controller
     * @return This circuit for fluent configuration
     */
    ILaminarCircuit<P> setResonanceController(IResonanceController controller);

    /**
     * Get the current resonance controller.
     *
     * @return The resonance controller
     */
    IResonanceController getResonanceController();

    // === Circuit Dynamics ===

    /**
     * Perform one processing cycle through the circuit.
     *
     * @param input Input pattern
     * @param parameters Circuit parameters
     * @return Processing result with layer activations
     */
    LaminarActivationResult processCycle(Pattern input, P parameters);

    /**
     * Check if the circuit has reached resonance.
     *
     * @return true if in resonant state
     */
    boolean isResonant();

    /**
     * Get the current resonance score (0.0 to 1.0).
     *
     * @return Current resonance level
     */
    double getResonanceScore();

    /**
     * Reset circuit to initial state without clearing learned weights.
     */
    void resetActivations();

    // === Monitoring and Events ===

    /**
     * Register a listener for circuit events.
     *
     * @param listener The event listener
     */
    void addListener(ICircuitEventListener listener);

    /**
     * Remove a registered listener.
     *
     * @param listener The listener to remove
     */
    void removeListener(ICircuitEventListener listener);

    /**
     * Get circuit state snapshot for visualization/debugging.
     *
     * @return Current circuit state
     */
    CircuitState getState();
}
```

### 2. ILayer Interface

```java
package com.hellblazer.art.laminar.core;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.ILayerParameters;
import java.util.List;

/**
 * Interface for laminar circuit layers with shunting dynamics.
 * Implements the Grossberg shunting equations for neural activation.
 *
 * @author Hal Hildebrand
 */
public interface ILayer {

    // === Layer Identity ===

    /**
     * Get unique identifier for this layer.
     *
     * @return Layer ID
     */
    String getId();

    /**
     * Get layer type for routing and processing decisions.
     *
     * @return Layer type
     */
    LayerType getType();

    /**
     * Get the number of neurons in this layer.
     *
     * @return Neuron count
     */
    int size();

    // === Activation Dynamics ===

    /**
     * Update layer activation based on inputs.
     * Implements shunting equations: dx/dt = -Ax + (B-x)E - (x+C)I
     *
     * @param excitation Excitatory input (E)
     * @param inhibition Inhibitory input (I)
     * @param parameters Layer parameters (A, B, C, etc.)
     * @param dt Time step for integration
     */
    void updateActivation(Pattern excitation, Pattern inhibition,
                          ILayerParameters parameters, double dt);

    /**
     * Get current activation pattern.
     *
     * @return Current layer activation
     */
    Pattern getActivation();

    /**
     * Set activation directly (for initialization).
     *
     * @param activation New activation pattern
     */
    void setActivation(Pattern activation);

    /**
     * Reset layer to resting state.
     */
    void reset();

    // === Signal Processing ===

    /**
     * Process bottom-up input signal.
     *
     * @param input Input pattern from lower layer
     * @param parameters Processing parameters
     * @return Transformed signal
     */
    Pattern processBottomUp(Pattern input, ILayerParameters parameters);

    /**
     * Process top-down feedback signal.
     *
     * @param feedback Feedback from higher layer
     * @param parameters Processing parameters
     * @return Transformed signal
     */
    Pattern processTopDown(Pattern feedback, ILayerParameters parameters);

    /**
     * Process horizontal (lateral) connections.
     *
     * @param lateral Input from same-level layers
     * @param parameters Processing parameters
     * @return Transformed signal
     */
    Pattern processLateral(Pattern lateral, ILayerParameters parameters);

    // === Learning ===

    /**
     * Update weights based on current activation and learning signal.
     *
     * @param learningSignal Signal indicating what to learn
     * @param learningRate Learning rate parameter
     */
    void updateWeights(Pattern learningSignal, double learningRate);

    /**
     * Get current weight matrix.
     *
     * @return Weight matrix
     */
    WeightMatrix getWeights();

    /**
     * Check if layer weights are plastic (can learn).
     *
     * @return true if weights can be modified
     */
    boolean isPlastic();

    /**
     * Enable or disable learning for this layer.
     *
     * @param plastic true to enable learning
     */
    void setPlastic(boolean plastic);

    // === Monitoring ===

    /**
     * Get layer statistics for monitoring.
     *
     * @return Layer statistics
     */
    LayerStatistics getStatistics();

    /**
     * Register activation listener.
     *
     * @param listener Activation event listener
     */
    void addActivationListener(ILayerActivationListener listener);
}

/**
 * Layer types for the laminar circuit.
 */
enum LayerType {
    INPUT,          // F0: Input preprocessing layer
    FEATURE,        // F1: Feature representation layer
    CATEGORY,       // F2: Category representation layer
    ATTENTION,      // Attentional gain control layer
    EXPECTATION,    // Top-down expectation layer
    BOUNDARY,       // Boundary completion layer
    SURFACE,        // Surface filling-in layer
    CUSTOM          // User-defined layer type
}
```

### 3. IPathway Interface

```java
package com.hellblazer.art.laminar.core;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.IPathwayParameters;

/**
 * Interface for connections between layers in laminar circuits.
 * Supports bottom-up, top-down, and horizontal pathways.
 *
 * @author Hal Hildebrand
 */
public interface IPathway {

    // === Pathway Identity ===

    /**
     * Get unique identifier for this pathway.
     *
     * @return Pathway ID
     */
    String getId();

    /**
     * Get pathway type.
     *
     * @return Type of connection
     */
    PathwayType getType();

    /**
     * Get source layer ID.
     *
     * @return Source layer identifier
     */
    String getSourceLayerId();

    /**
     * Get target layer ID.
     *
     * @return Target layer identifier
     */
    String getTargetLayerId();

    // === Signal Propagation ===

    /**
     * Propagate signal through the pathway.
     *
     * @param input Signal from source layer
     * @param parameters Pathway parameters
     * @return Transformed signal for target layer
     */
    Pattern propagate(Pattern input, IPathwayParameters parameters);

    /**
     * Get propagation delay in time steps.
     *
     * @return Delay in time steps
     */
    int getDelay();

    /**
     * Set propagation delay.
     *
     * @param delay Delay in time steps
     */
    void setDelay(int delay);

    // === Connection Weights ===

    /**
     * Get connection weight matrix.
     *
     * @return Weight matrix
     */
    WeightMatrix getWeights();

    /**
     * Update connection weights.
     *
     * @param source Source layer activation
     * @param target Target layer activation
     * @param learningRate Learning rate
     */
    void updateWeights(Pattern source, Pattern target, double learningRate);

    /**
     * Check if pathway weights are adaptive.
     *
     * @return true if weights can change
     */
    boolean isAdaptive();

    /**
     * Enable or disable weight adaptation.
     *
     * @param adaptive true to enable learning
     */
    void setAdaptive(boolean adaptive);

    // === Modulation ===

    /**
     * Apply gain modulation to the pathway.
     *
     * @param gain Multiplicative gain factor
     */
    void applyGain(double gain);

    /**
     * Get current gain value.
     *
     * @return Current gain
     */
    double getGain();

    /**
     * Enable or disable this pathway.
     *
     * @param enabled true to enable signal flow
     */
    void setEnabled(boolean enabled);

    /**
     * Check if pathway is enabled.
     *
     * @return true if signal can flow
     */
    boolean isEnabled();
}

/**
 * Types of pathways in laminar circuits.
 */
enum PathwayType {
    BOTTOM_UP,      // Feedforward connections
    TOP_DOWN,       // Feedback connections
    HORIZONTAL,     // Lateral connections within layer
    DIAGONAL,       // Skip connections across layers
    MODULATORY      // Gain control connections
}
```

### 4. IResonanceController Interface

```java
package com.hellblazer.art.laminar.core;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.IResonanceParameters;

/**
 * Interface for controlling resonance and attention in laminar circuits.
 * Implements the matching and reset mechanisms of ART.
 *
 * @author Hal Hildebrand
 */
public interface IResonanceController {

    // === Resonance Detection ===

    /**
     * Check if current circuit state is resonant.
     *
     * @param bottomUp Bottom-up input pattern
     * @param topDown Top-down expectation pattern
     * @param parameters Resonance parameters
     * @return true if resonance achieved
     */
    boolean isResonant(Pattern bottomUp, Pattern topDown,
                       IResonanceParameters parameters);

    /**
     * Calculate match score between patterns.
     *
     * @param bottomUp Bottom-up input
     * @param topDown Top-down expectation
     * @return Match score (0.0 to 1.0)
     */
    double calculateMatch(Pattern bottomUp, Pattern topDown);

    /**
     * Get current vigilance parameter.
     *
     * @return Vigilance value
     */
    double getVigilance();

    /**
     * Set vigilance parameter.
     *
     * @param vigilance New vigilance value (0.0 to 1.0)
     */
    void setVigilance(double vigilance);

    // === Attention Control ===

    /**
     * Focus attention on specific features.
     *
     * @param features Feature indices to attend
     */
    void focusAttention(int[] features);

    /**
     * Get current attention weights.
     *
     * @return Attention weight pattern
     */
    Pattern getAttentionWeights();

    /**
     * Apply attention gain to a pattern.
     *
     * @param input Input pattern
     * @return Attention-modulated pattern
     */
    Pattern applyAttention(Pattern input);

    // === Reset Mechanism ===

    /**
     * Check if reset should occur.
     *
     * @param matchScore Current match score
     * @return true if reset needed
     */
    boolean shouldReset(double matchScore);

    /**
     * Perform reset operation.
     *
     * @param categoryIndex Category to reset
     */
    void reset(int categoryIndex);

    /**
     * Get reset history for debugging.
     *
     * @return List of reset events
     */
    List<ResetEvent> getResetHistory();

    // === Search Control ===

    /**
     * Get next category to try after reset.
     *
     * @param excludedCategories Already tried categories
     * @return Next category index, or -1 if none
     */
    int getNextCategory(Set<Integer> excludedCategories);

    /**
     * Update search order based on success.
     *
     * @param categoryIndex Successful category
     */
    void reinforceSearchOrder(int categoryIndex);
}
```

## Parameter Interfaces

### 1. ILaminarParameters Interface

```java
package com.hellblazer.art.laminar.parameters;

import java.io.Serializable;

/**
 * Base interface for laminar circuit parameters.
 *
 * @author Hal Hildebrand
 */
public interface ILaminarParameters extends Serializable {

    /**
     * Get parameters for specific layer.
     *
     * @param layerId Layer identifier
     * @return Layer parameters
     */
    ILayerParameters getLayerParameters(String layerId);

    /**
     * Get parameters for specific pathway.
     *
     * @param pathwayId Pathway identifier
     * @return Pathway parameters
     */
    IPathwayParameters getPathwayParameters(String pathwayId);

    /**
     * Get resonance control parameters.
     *
     * @return Resonance parameters
     */
    IResonanceParameters getResonanceParameters();

    /**
     * Get shunting equation parameters.
     *
     * @return Shunting parameters
     */
    IShuntingParameters getShuntingParameters();

    /**
     * Get learning parameters.
     *
     * @return Learning parameters
     */
    ILearningParameters getLearningParameters();

    /**
     * Validate parameter consistency.
     *
     * @return true if parameters are valid
     */
    boolean validate();

    /**
     * Create a deep copy of parameters.
     *
     * @return Parameter copy
     */
    ILaminarParameters copy();
}
```

### 2. ILayerParameters Interface

```java
package com.hellblazer.art.laminar.parameters;

/**
 * Parameters for layer dynamics and processing.
 *
 * @author Hal Hildebrand
 */
public interface ILayerParameters {

    // Shunting equation parameters
    double getDecayRate();           // A in dx/dt = -Ax + ...
    double getUpperBound();           // B in (B-x)E
    double getLowerBound();           // C in (x+C)I

    // Processing parameters
    double getActivationThreshold();
    double getSaturationLevel();
    boolean useNormalization();
    NormalizationType getNormalizationType();

    // Temporal parameters
    double getTimeConstant();
    int getIntegrationSteps();

    // Noise parameters
    double getNoiseLevel();
    NoiseType getNoiseType();
}
```

### 3. IPathwayParameters Interface

```java
package com.hellblazer.art.laminar.parameters;

/**
 * Parameters for pathway connections.
 *
 * @author Hal Hildebrand
 */
public interface IPathwayParameters {

    // Connection strength
    double getConnectionStrength();
    double getInitialGain();

    // Propagation parameters
    int getPropagationDelay();
    double getSignalAttenuation();

    // Learning parameters
    double getLearningRate();
    boolean isHebbian();
    boolean isCompetitive();

    // Modulation parameters
    double getMaxGain();
    double getMinGain();
    double getGainDecay();
}
```

## Factory and Builder Patterns

### 1. LaminarCircuitBuilder

```java
package com.hellblazer.art.laminar.builders;

import com.hellblazer.art.laminar.core.*;
import com.hellblazer.art.laminar.parameters.ILaminarParameters;

/**
 * Fluent builder for constructing laminar circuits.
 *
 * @author Hal Hildebrand
 */
public class LaminarCircuitBuilder<P extends ILaminarParameters> {

    private final Map<Integer, ILayer> layers = new HashMap<>();
    private final List<IPathway> pathways = new ArrayList<>();
    private IResonanceController resonanceController;
    private P parameters;
    private final List<ICircuitEventListener> listeners = new ArrayList<>();

    // === Layer Configuration ===

    public LaminarCircuitBuilder<P> withInputLayer(int size) {
        layers.put(0, LayerFactory.createInputLayer(size));
        return this;
    }

    public LaminarCircuitBuilder<P> withFeatureLayer(int size, int depth) {
        layers.put(depth, LayerFactory.createFeatureLayer(size));
        return this;
    }

    public LaminarCircuitBuilder<P> withCategoryLayer(int size, int depth) {
        layers.put(depth, LayerFactory.createCategoryLayer(size));
        return this;
    }

    public LaminarCircuitBuilder<P> withCustomLayer(ILayer layer, int depth) {
        layers.put(depth, layer);
        return this;
    }

    // === Pathway Configuration ===

    public LaminarCircuitBuilder<P> connectBottomUp(String sourceId, String targetId) {
        pathways.add(PathwayBuilder.bottomUp()
            .from(sourceId)
            .to(targetId)
            .build());
        return this;
    }

    public LaminarCircuitBuilder<P> connectTopDown(String sourceId, String targetId) {
        pathways.add(PathwayBuilder.topDown()
            .from(sourceId)
            .to(targetId)
            .build());
        return this;
    }

    public LaminarCircuitBuilder<P> connectLateral(String layerId) {
        pathways.add(PathwayBuilder.horizontal()
            .withinLayer(layerId)
            .build());
        return this;
    }

    public LaminarCircuitBuilder<P> withPathway(IPathway pathway) {
        pathways.add(pathway);
        return this;
    }

    // === Resonance Configuration ===

    public LaminarCircuitBuilder<P> withResonanceController(IResonanceController controller) {
        this.resonanceController = controller;
        return this;
    }

    public LaminarCircuitBuilder<P> withVigilance(double vigilance) {
        if (resonanceController == null) {
            resonanceController = new DefaultResonanceController();
        }
        resonanceController.setVigilance(vigilance);
        return this;
    }

    // === Parameters ===

    public LaminarCircuitBuilder<P> withParameters(P parameters) {
        this.parameters = parameters;
        return this;
    }

    // === Event Listeners ===

    public LaminarCircuitBuilder<P> withListener(ICircuitEventListener listener) {
        listeners.add(listener);
        return this;
    }

    // === Build Methods ===

    public ILaminarCircuit<P> build() {
        validate();

        var circuit = createCircuit();

        // Add layers
        layers.forEach((depth, layer) -> circuit.addLayer(layer, depth));

        // Add pathways
        pathways.forEach(circuit::connectLayers);

        // Set resonance controller
        if (resonanceController != null) {
            circuit.setResonanceController(resonanceController);
        }

        // Add listeners
        listeners.forEach(circuit::addListener);

        return circuit;
    }

    public VectorizedLaminarCircuit<P> buildVectorized() {
        validate();

        var circuit = createVectorizedCircuit();

        // Configuration similar to build()
        // but with vectorized implementations

        return circuit;
    }

    private void validate() {
        if (layers.isEmpty()) {
            throw new IllegalStateException("Circuit must have at least one layer");
        }
        if (parameters == null) {
            throw new IllegalStateException("Parameters must be specified");
        }
        if (!parameters.validate()) {
            throw new IllegalStateException("Invalid parameters");
        }
    }

    private ILaminarCircuit<P> createCircuit() {
        return new StandardLaminarCircuit<>(parameters);
    }

    private VectorizedLaminarCircuit<P> createVectorizedCircuit() {
        return new VectorizedLaminarCircuit<>(parameters);
    }
}
```

### 2. LayerFactory

```java
package com.hellblazer.art.laminar.factories;

import com.hellblazer.art.laminar.core.*;

/**
 * Factory for creating different types of layers.
 *
 * @author Hal Hildebrand
 */
public class LayerFactory {

    public static ILayer createInputLayer(int size) {
        return new InputLayer(size);
    }

    public static ILayer createFeatureLayer(int size) {
        return new FeatureLayer(size);
    }

    public static ILayer createCategoryLayer(int size) {
        return new CategoryLayer(size);
    }

    public static ILayer createAttentionLayer(int size) {
        return new AttentionLayer(size);
    }

    public static ILayer createBoundaryLayer(int width, int height) {
        return new BoundaryLayer(width, height);
    }

    public static ILayer createSurfaceLayer(int width, int height) {
        return new SurfaceLayer(width, height);
    }

    public static ILayer createCustomLayer(LayerType type, int size,
                                          LayerConfiguration config) {
        return new CustomLayer(type, size, config);
    }

    // Vectorized versions

    public static ILayer createVectorizedFeatureLayer(int size) {
        return new VectorizedFeatureLayer(size);
    }

    public static ILayer createVectorizedCategoryLayer(int size) {
        return new VectorizedCategoryLayer(size);
    }
}
```

## Event and Callback Interfaces

### 1. ICircuitEventListener

```java
package com.hellblazer.art.laminar.events;

/**
 * Listener for circuit-level events.
 *
 * @author Hal Hildebrand
 */
public interface ICircuitEventListener {

    /**
     * Called when resonance is achieved.
     *
     * @param event Resonance event details
     */
    void onResonance(ResonanceEvent event);

    /**
     * Called when a reset occurs.
     *
     * @param event Reset event details
     */
    void onReset(ResetEvent event);

    /**
     * Called when a new category is created.
     *
     * @param event Category creation event
     */
    void onCategoryCreated(CategoryEvent event);

    /**
     * Called when attention focus changes.
     *
     * @param event Attention shift event
     */
    void onAttentionShift(AttentionEvent event);

    /**
     * Called after each processing cycle.
     *
     * @param event Cycle completion event
     */
    void onCycleComplete(CycleEvent event);
}
```

### 2. ILayerActivationListener

```java
package com.hellblazer.art.laminar.events;

import com.hellblazer.art.core.Pattern;

/**
 * Listener for layer activation events.
 *
 * @author Hal Hildebrand
 */
public interface ILayerActivationListener {

    /**
     * Called when layer activation changes.
     *
     * @param layerId Layer identifier
     * @param oldActivation Previous activation
     * @param newActivation New activation
     * @param timestamp Event timestamp
     */
    void onActivationChange(String layerId, Pattern oldActivation,
                           Pattern newActivation, long timestamp);

    /**
     * Called when layer reaches threshold.
     *
     * @param layerId Layer identifier
     * @param activation Current activation
     */
    void onThresholdReached(String layerId, Pattern activation);

    /**
     * Called when layer saturates.
     *
     * @param layerId Layer identifier
     * @param saturationLevel Saturation amount
     */
    void onSaturation(String layerId, double saturationLevel);
}
```

### 3. ILearningProgressListener

```java
package com.hellblazer.art.laminar.events;

/**
 * Listener for monitoring learning progress.
 *
 * @author Hal Hildebrand
 */
public interface ILearningProgressListener {

    /**
     * Called when weights are updated.
     *
     * @param layerId Layer or pathway ID
     * @param weightChange Magnitude of change
     * @param learningRate Current learning rate
     */
    void onWeightUpdate(String layerId, double weightChange, double learningRate);

    /**
     * Called when learning converges.
     *
     * @param iterations Number of iterations to convergence
     * @param finalError Final error measure
     */
    void onConvergence(int iterations, double finalError);

    /**
     * Called periodically with learning metrics.
     *
     * @param metrics Current learning metrics
     */
    void onLearningMetrics(LearningMetrics metrics);
}
```

## Adapter Interfaces

### 1. IARTToLaminarAdapter

```java
package com.hellblazer.art.laminar.adapters;

import com.hellblazer.art.core.ARTAlgorithm;
import com.hellblazer.art.laminar.core.ILaminarCircuit;

/**
 * Adapter for using existing ART algorithms in laminar circuits.
 *
 * @author Hal Hildebrand
 */
public interface IARTToLaminarAdapter<P> {

    /**
     * Wrap an ART algorithm as a laminar circuit.
     *
     * @param algorithm The ART algorithm to wrap
     * @return Laminar circuit adapter
     */
    ILaminarCircuit<P> adaptToLaminar(ARTAlgorithm<P> algorithm);

    /**
     * Extract layer representation from ART algorithm.
     *
     * @param algorithm The ART algorithm
     * @param layerType Type of layer to extract
     * @return Layer representation
     */
    ILayer extractLayer(ARTAlgorithm<P> algorithm, LayerType layerType);

    /**
     * Map ART parameters to laminar parameters.
     *
     * @param artParameters Original parameters
     * @return Laminar parameters
     */
    ILaminarParameters mapParameters(P artParameters);
}
```

### 2. ILaminarToVectorizedAdapter

```java
package com.hellblazer.art.laminar.adapters;

import com.hellblazer.art.performance.VectorizedARTAlgorithm;
import com.hellblazer.art.laminar.core.ILaminarCircuit;

/**
 * Adapter for vectorizing laminar circuits.
 *
 * @author Hal Hildebrand
 */
public interface ILaminarToVectorizedAdapter<T, P> {

    /**
     * Convert laminar circuit to vectorized implementation.
     *
     * @param circuit The laminar circuit
     * @return Vectorized implementation
     */
    VectorizedARTAlgorithm<T, P> vectorize(ILaminarCircuit<P> circuit);

    /**
     * Optimize circuit for SIMD operations.
     *
     * @param circuit The circuit to optimize
     * @return Optimization report
     */
    OptimizationReport optimizeForSIMD(ILaminarCircuit<P> circuit);

    /**
     * Check if circuit can be vectorized.
     *
     * @param circuit The circuit to check
     * @return true if vectorizable
     */
    boolean canVectorize(ILaminarCircuit<P> circuit);
}
```

## Extension Points

### 1. Custom Layer Implementation

```java
package com.hellblazer.art.laminar.extensions;

/**
 * Base class for custom layer implementations.
 *
 * @author Hal Hildebrand
 */
public abstract class CustomLayerBase implements ILayer {

    protected final String id;
    protected final LayerType type;
    protected final int size;
    protected Pattern activation;
    protected WeightMatrix weights;
    protected boolean plastic = true;
    protected final List<ILayerActivationListener> listeners = new ArrayList<>();

    protected CustomLayerBase(String id, LayerType type, int size) {
        this.id = id;
        this.type = type;
        this.size = size;
        this.activation = new DenseVector(size);
        this.weights = new WeightMatrix(size, size);
    }

    // Template method for custom processing
    protected abstract Pattern customProcess(Pattern input);

    // Hook for custom learning rules
    protected abstract void customLearn(Pattern signal, double rate);

    // Extension point for custom dynamics
    protected abstract void customDynamics(double dt);
}
```

### 2. Plugin Architecture

```java
package com.hellblazer.art.laminar.plugins;

/**
 * Plugin interface for extending laminar circuits.
 *
 * @author Hal Hildebrand
 */
public interface ILaminarPlugin {

    /**
     * Get plugin name.
     *
     * @return Plugin name
     */
    String getName();

    /**
     * Get plugin version.
     *
     * @return Version string
     */
    String getVersion();

    /**
     * Initialize plugin with circuit.
     *
     * @param circuit The host circuit
     */
    void initialize(ILaminarCircuit<?> circuit);

    /**
     * Called before each processing cycle.
     *
     * @param input Input pattern
     * @return Modified input (or original)
     */
    Pattern preprocessInput(Pattern input);

    /**
     * Called after each processing cycle.
     *
     * @param result Processing result
     */
    void postprocessResult(LaminarActivationResult result);

    /**
     * Clean up plugin resources.
     */
    void shutdown();
}
```

## Usage Examples

### Example 1: Basic Laminar Circuit Construction

```java
// Create a simple laminar ART circuit
var builder = new LaminarCircuitBuilder<LaminarParameters>();

var circuit = builder
    // Define layers
    .withInputLayer(100)
    .withFeatureLayer(50, 1)
    .withCategoryLayer(20, 2)

    // Connect layers
    .connectBottomUp("input", "feature")
    .connectBottomUp("feature", "category")
    .connectTopDown("category", "feature")
    .connectLateral("feature")

    // Configure resonance
    .withVigilance(0.9)

    // Set parameters
    .withParameters(new LaminarParameters.Builder()
        .withLayerDecayRate(0.1)
        .withLearningRate(0.01)
        .withTimeConstant(10.0)
        .build())

    // Add monitoring
    .withListener(new CircuitMonitor())

    // Build the circuit
    .build();

// Use the circuit
var input = new DenseVector(100);
var result = circuit.learn(input, circuit.getParameters());
```

### Example 2: Vectorized Circuit with Custom Layers

```java
// Create vectorized circuit with custom layers
var builder = new LaminarCircuitBuilder<VectorizedLaminarParameters>();

var circuit = builder
    // Use vectorized layers
    .withCustomLayer(LayerFactory.createVectorizedFeatureLayer(256), 1)
    .withCustomLayer(new CustomAttentionLayer(256), 2)
    .withCategoryLayer(64, 3)

    // Complex connectivity
    .connectBottomUp("input", "feature")
    .connectBottomUp("feature", "attention")
    .connectBottomUp("attention", "category")
    .connectTopDown("category", "attention")
    .connectTopDown("attention", "feature")
    .withPathway(new AdaptivePathway("feature", "category", PathwayType.DIAGONAL))

    // Advanced resonance control
    .withResonanceController(new AdaptiveResonanceController()
        .withDualVigilance(0.8, 0.95)
        .withAttentionMechanism(AttentionType.COMPETITIVE))

    // Build vectorized version
    .buildVectorized();

// Batch processing
var inputs = List.of(pattern1, pattern2, pattern3);
var results = circuit.learnBatch(inputs, parameters);
```

### Example 3: Integration with Existing ART

```java
// Adapt existing FuzzyART to laminar circuit
var fuzzyART = new FuzzyART(inputSize);
var adapter = new ARTToLaminarAdapter();

var laminarCircuit = adapter.adaptToLaminar(fuzzyART);

// Enhance with laminar features
laminarCircuit
    .addLayer(new AttentionLayer(inputSize), 3)
    .connectLayers(new ModulatoryPathway("attention", "feature"));

// Use with SklearnWrapper
var sklearn = new SklearnWrapper(laminarCircuit, parameters);
sklearn.fit(trainingData);
var predictions = sklearn.predict(testData);
```

### Example 4: Event-Driven Monitoring

```java
// Create circuit with comprehensive monitoring
var circuit = new LaminarCircuitBuilder<LaminarParameters>()
    .withInputLayer(100)
    .withFeatureLayer(50, 1)
    .withCategoryLayer(20, 2)
    .connectBottomUp("input", "feature")
    .connectBottomUp("feature", "category")
    .withParameters(defaultParameters)
    .build();

// Add event listeners
circuit.addListener(new ICircuitEventListener() {
    @Override
    public void onResonance(ResonanceEvent event) {
        System.out.printf("Resonance achieved: category=%d, match=%.3f%n",
            event.getCategoryIndex(), event.getMatchScore());
    }

    @Override
    public void onReset(ResetEvent event) {
        System.out.printf("Reset occurred: category=%d, reason=%s%n",
            event.getCategoryIndex(), event.getReason());
    }

    @Override
    public void onCycleComplete(CycleEvent event) {
        if (event.getCycleNumber() % 100 == 0) {
            System.out.printf("Cycle %d: resonance=%.3f%n",
                event.getCycleNumber(), event.getResonanceScore());
        }
    }
});

// Layer-specific monitoring
circuit.getLayer(1).addActivationListener(new ILayerActivationListener() {
    @Override
    public void onActivationChange(String layerId, Pattern oldActivation,
                                  Pattern newActivation, long timestamp) {
        var change = newActivation.subtract(oldActivation).norm();
        if (change > 0.1) {
            System.out.printf("Significant activation change in %s: %.3f%n",
                layerId, change);
        }
    }
});
```

## Performance Considerations

### Vectorization Strategy
- All core operations implement SIMD-friendly algorithms
- Layer activations use vector operations when size >= 64
- Pathway propagation uses matrix-vector multiplication
- Batch processing amortizes overhead

### Memory Management
- Object pooling for frequently allocated patterns
- Lazy initialization of large matrices
- Copy-on-write for parameter updates
- Weak references for event listeners

### Parallelization Opportunities
- Layer updates can be parallel when independent
- Multiple pathways can propagate simultaneously
- Category search parallelizable across candidates
- Event dispatching on separate thread

## Testing Strategy

### Unit Test Coverage
- Each interface has corresponding test contract
- Factory methods have creation tests
- Builders validate construction logic
- Event dispatch verified with mock listeners

### Integration Testing
- Full circuit construction and operation
- Adapter compatibility with existing ART
- SklearnWrapper integration
- Performance benchmarks with JMH

### Example Test

```java
@Test
public void testLaminarCircuitResonance() {
    // Arrange
    var circuit = new LaminarCircuitBuilder<LaminarParameters>()
        .withInputLayer(10)
        .withFeatureLayer(5, 1)
        .withCategoryLayer(2, 2)
        .connectBottomUp("input", "feature")
        .connectBottomUp("feature", "category")
        .connectTopDown("category", "feature")
        .withVigilance(0.8)
        .withParameters(testParameters)
        .build();

    var resonanceListener = mock(ICircuitEventListener.class);
    circuit.addListener(resonanceListener);

    // Act
    var input = new DenseVector(new double[]{1, 0, 1, 0, 1, 0, 1, 0, 1, 0});
    var result = circuit.learn(input, testParameters);

    // Assert
    assertTrue(circuit.isResonant());
    assertTrue(result instanceof LaminarActivationResult.Success);
    verify(resonanceListener, times(1)).onResonance(any(ResonanceEvent.class));
    assertEquals(0, result.getCategoryIndex());
    assertTrue(result.getResonanceScore() > 0.8);
}
```