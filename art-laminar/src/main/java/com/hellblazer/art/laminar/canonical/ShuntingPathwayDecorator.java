package com.hellblazer.art.laminar.canonical;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.core.Pathway;
import com.hellblazer.art.laminar.core.PathwayType;
import com.hellblazer.art.laminar.parameters.PathwayParameters;
import com.hellblazer.art.temporal.core.ShuntingDynamics;
import com.hellblazer.art.temporal.core.ShuntingParameters;
import com.hellblazer.art.temporal.core.ShuntingState;
import com.hellblazer.art.temporal.core.TransmitterDynamics;
import com.hellblazer.art.temporal.core.TransmitterParameters;
import com.hellblazer.art.temporal.core.TransmitterState;

/**
 * Decorator that wraps existing laminar pathways with sophisticated temporal dynamics.
 * Implements the integration of shunting and transmitter dynamics into pathway processing
 * without modifying the original pathway implementation.
 *
 * This follows the decorator pattern to preserve existing well-tested pathway implementations
 * while adding canonical laminar circuit temporal dynamics.
 *
 * Implements equations from Grossberg's canonical laminar circuit:
 * - Shunting: dX_i/dt = -A_i * X_i + (B - X_i) * S_i - X_i * Σ(j≠i) I_ij
 * - Transmitter: dZ_i/dt = ε(1 - Z_i) - Z_i(λ * S_i + μ * S_i²)
 *
 * @author Hal Hildebrand
 */
public class ShuntingPathwayDecorator implements TemporallyIntegratedPathway {

    private final Pathway delegate;
    private final ShuntingDynamics shuntingDynamics;
    private final TransmitterDynamics transmitterDynamics;
    private final ShuntingParameters shuntingParameters;
    private final TransmitterParameters transmitterParameters;
    private final SimpleIntegrator<ShuntingState, ShuntingParameters> shuntingIntegrator;
    private final SimpleIntegrator<TransmitterState, TransmitterParameters> transmitterIntegrator;
    private final TimeScale timeScale;

    private ShuntingState currentShuntingState;
    private TransmitterState currentTransmitterState;
    private boolean temporalDynamicsEnabled;
    private double convergenceThreshold;

    /**
     * Create a shunting pathway decorator.
     *
     * @param delegate the pathway to wrap
     * @param shuntingParams shunting dynamics parameters
     * @param transmitterParams transmitter dynamics parameters
     * @param timeScale the time scale at which this pathway operates
     */
    public ShuntingPathwayDecorator(
        Pathway delegate,
        ShuntingParameters shuntingParams,
        TransmitterParameters transmitterParams,
        TimeScale timeScale
    ) {
        this.delegate = delegate;
        this.shuntingDynamics = new ShuntingDynamics();
        this.transmitterDynamics = new TransmitterDynamics();
        this.shuntingParameters = shuntingParams;
        this.transmitterParameters = transmitterParams;
        this.shuntingIntegrator = new SimpleIntegrator<>(shuntingDynamics);
        this.transmitterIntegrator = new SimpleIntegrator<>(transmitterDynamics);
        this.timeScale = timeScale;
        this.temporalDynamicsEnabled = true;
        this.convergenceThreshold = 1e-5;

        // Initialize states (will be properly sized on first use)
        this.currentShuntingState = null;
        this.currentTransmitterState = null;
    }

    @Override
    public Pattern propagate(Pattern signal, PathwayParameters parameters) {
        if (!temporalDynamicsEnabled) {
            // Fall back to original pathway behavior
            return delegate.propagate(signal, parameters);
        }

        // Ensure states are initialized
        ensureStatesInitialized(signal.dimension());

        // Convert pattern to shunting state
        var shuntingInput = convertPatternToShuntingState(signal);

        // Integrate shunting dynamics
        var integratedShunting = shuntingIntegrator.integrate(
            shuntingInput,
            shuntingParameters,
            0.0,
            timeScale.getTypicalTimeStep()
        );

        // Update current shunting state
        currentShuntingState = integratedShunting;

        // Apply transmitter gating
        var shuntingActivations = integratedShunting.getActivations();
        var gatedActivations = currentTransmitterState.applyGating(shuntingActivations);

        // Update transmitter state based on signal usage
        updateTransmitterState(signal, timeScale.getTypicalTimeStep());

        // Convert back to pattern and apply through delegate
        var gatedPattern = new DenseVector(gatedActivations);

        // Allow delegate to apply weights, gain, etc.
        return delegate.propagate(gatedPattern, parameters);
    }

    /**
     * Update transmitter dynamics based on signal usage.
     */
    private void updateTransmitterState(Pattern signal, double timeStep) {
        // Set presynaptic signals from input pattern
        var signalArray = signal.toArray();
        var transmitterInput = new TransmitterState(
            currentTransmitterState.getTransmitterLevels(),
            signalArray,
            currentTransmitterState.getDepletionHistory()
        );

        // Integrate transmitter dynamics
        var integratedTransmitter = transmitterIntegrator.integrate(
            transmitterInput,
            transmitterParameters,
            0.0,
            timeStep
        );

        currentTransmitterState = integratedTransmitter;
    }

    /**
     * Convert pattern to shunting state with excitatory inputs.
     */
    private ShuntingState convertPatternToShuntingState(Pattern pattern) {
        var activations = currentShuntingState != null ?
            currentShuntingState.getActivations() :
            new double[pattern.dimension()];

        var excitatoryInputs = pattern.toArray();
        return new ShuntingState(activations, excitatoryInputs);
    }

    /**
     * Ensure states are initialized to correct dimension.
     */
    private void ensureStatesInitialized(int dimension) {
        if (currentShuntingState == null || currentShuntingState.dimension() != dimension) {
            var initialActivations = new double[dimension];
            var initialInputs = new double[dimension];
            currentShuntingState = new ShuntingState(initialActivations, initialInputs);
        }

        if (currentTransmitterState == null || currentTransmitterState.dimension() != dimension) {
            currentTransmitterState = new TransmitterState(dimension);
        }
    }

    @Override
    public void updateDynamics(double timeStep) {
        if (!temporalDynamicsEnabled || currentShuntingState == null) {
            return;
        }

        // Integrate shunting dynamics forward
        currentShuntingState = shuntingIntegrator.integrate(
            currentShuntingState,
            shuntingParameters,
            0.0,
            timeStep
        );

        // Integrate transmitter dynamics forward
        currentTransmitterState = transmitterIntegrator.integrate(
            currentTransmitterState,
            transmitterParameters,
            0.0,
            timeStep
        );
    }

    @Override
    public boolean hasReachedEquilibrium(double threshold) {
        if (currentShuntingState == null) {
            return true;
        }

        // Check if shunting dynamics have converged
        return shuntingDynamics.hasConverged(
            currentShuntingState,
            getPreviousShuntingState(),
            threshold
        );
    }

    /**
     * Get previous shunting state for convergence check.
     * In a real implementation, this would maintain state history.
     */
    private ShuntingState getPreviousShuntingState() {
        // Simulate one step backward for convergence check
        if (currentShuntingState == null) {
            return null;
        }

        // In production, we'd maintain a history buffer
        // For now, we'll compute a small backward step
        var derivative = shuntingDynamics.computeDerivative(
            currentShuntingState,
            shuntingParameters,
            0.0
        );

        var activations = currentShuntingState.getActivations();
        var previousActivations = new double[activations.length];
        for (int i = 0; i < activations.length; i++) {
            previousActivations[i] = activations[i] - derivative.getActivations()[i] * 0.001;
        }

        return new ShuntingState(previousActivations, currentShuntingState.getExcitatoryInputs());
    }

    @Override
    public void resetDynamics() {
        if (currentShuntingState != null) {
            var dimension = currentShuntingState.dimension();
            currentShuntingState = new ShuntingState(
                new double[dimension],
                new double[dimension]
            );
            currentTransmitterState = new TransmitterState(dimension);
        }
    }

    // ============ TemporallyIntegratedPathway Implementation ============

    @Override
    public ShuntingState getShuntingState() {
        return currentShuntingState;
    }

    @Override
    public TransmitterState getTransmitterState() {
        return currentTransmitterState;
    }

    @Override
    public TimeScale getTimeScale() {
        return timeScale;
    }

    @Override
    public void setTemporalDynamicsEnabled(boolean enabled) {
        this.temporalDynamicsEnabled = enabled;
    }

    @Override
    public boolean isTemporalDynamicsEnabled() {
        return temporalDynamicsEnabled;
    }

    // ============ Delegate Pathway Methods ============

    @Override
    public String getId() {
        return delegate.getId();
    }

    @Override
    public String getSourceLayerId() {
        return delegate.getSourceLayerId();
    }

    @Override
    public String getTargetLayerId() {
        return delegate.getTargetLayerId();
    }

    @Override
    public PathwayType getType() {
        return delegate.getType();
    }

    @Override
    public double getGain() {
        return delegate.getGain();
    }

    @Override
    public void setGain(double gain) {
        delegate.setGain(gain);
    }

    @Override
    public boolean isAdaptive() {
        return delegate.isAdaptive();
    }

    @Override
    public void updateWeights(Pattern sourceActivation, Pattern targetActivation, double learningRate) {
        delegate.updateWeights(sourceActivation, targetActivation, learningRate);
    }

    @Override
    public void reset() {
        delegate.reset();
        resetDynamics();
    }

    // ============ Configuration ============

    /**
     * Set convergence threshold for equilibrium detection.
     */
    public void setConvergenceThreshold(double threshold) {
        this.convergenceThreshold = threshold;
    }

    /**
     * Get shunting dynamics parameters.
     */
    public ShuntingParameters getShuntingParameters() {
        return shuntingParameters;
    }

    /**
     * Get transmitter dynamics parameters.
     */
    public TransmitterParameters getTransmitterParameters() {
        return transmitterParameters;
    }

    /**
     * Get the wrapped pathway.
     */
    public Pathway getDelegate() {
        return delegate;
    }
}