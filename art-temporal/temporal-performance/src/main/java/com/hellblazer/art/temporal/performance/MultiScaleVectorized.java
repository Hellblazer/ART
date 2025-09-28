package com.hellblazer.art.temporal.performance;

import com.hellblazer.art.temporal.dynamics.MultiScaleParameters;
import com.hellblazer.art.temporal.core.ActivationState;
import jdk.incubator.vector.*;
import java.util.concurrent.*;
import java.util.Arrays;

/**
 * Vectorized multi-scale dynamics with parallel time scale processing.
 */
public class MultiScaleVectorized {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private final MultiScaleParameters parameters;
    private final int dimension;
    private final int vectorLength;
    private final int loopBound;

    // Time scales (fast, medium, slow)
    private final VectorizedShuntingDynamics[] scales;
    private final double[] timeConstants;
    private final double[] scaleWeights;

    // Cross-scale coupling
    private double[][] crossScaleWeights;
    private double[] couplingStrength;

    // State arrays for each scale
    private double[][] scaleStates;
    private double[][] scaleDerivatives;

    // Combined output
    private double[] combinedOutput;

    // Thread pool for parallel scale processing
    private final ForkJoinPool scalePool;

    public MultiScaleVectorized(MultiScaleParameters parameters) {
        this.parameters = parameters;
        this.dimension = parameters.getDimension();
        this.vectorLength = SPECIES.length();
        this.loopBound = SPECIES.loopBound(dimension);

        int numScales = parameters.getNumberOfScales();
        this.scales = new VectorizedShuntingDynamics[numScales];
        this.timeConstants = parameters.getTimeConstants();
        this.scaleWeights = parameters.getScaleWeights();

        // Initialize scales with different time constants
        for (int i = 0; i < numScales; i++) {
            var scaleParams = (com.hellblazer.art.temporal.dynamics.ShuntingParameters) parameters.getScaleParameters(i);
            scales[i] = new VectorizedShuntingDynamics(scaleParams, dimension);
        }

        // Initialize cross-scale coupling
        initializeCrossScaleCoupling();

        // State arrays
        this.scaleStates = new double[numScales][dimension];
        this.scaleDerivatives = new double[numScales][dimension];
        this.combinedOutput = new double[dimension];

        // Thread pool for parallel processing
        this.scalePool = ForkJoinPool.commonPool();
    }

    /**
     * Update multi-scale dynamics with parallel + vectorized computation.
     */
    public ActivationState update(double[] input, double deltaT) {
        // Process scales in parallel
        var futures = new CompletableFuture[scales.length];

        for (int s = 0; s < scales.length; s++) {
            final int scale = s;
            futures[s] = CompletableFuture.runAsync(() ->
                updateScaleVectorized(scale, input, deltaT), scalePool);
        }

        // Wait for all scales
        CompletableFuture.allOf(futures).join();

        // Apply cross-scale coupling (vectorized)
        applyCrossScaleCouplingVectorized();

        // Combine outputs (vectorized)
        combineOutputsVectorized();

        return new ActivationState(combinedOutput.clone());
    }

    /**
     * Update individual scale with vectorization.
     */
    private void updateScaleVectorized(int scaleIdx, double[] input, double deltaT) {
        var scale = scales[scaleIdx];
        double timeConstant = timeConstants[scaleIdx];

        // Adjust time step for this scale
        double scaleDeltaT = deltaT / timeConstant;

        // Set input with scale-specific processing
        double[] scaledInput = preprocessInputForScale(input, scaleIdx);
        scale.setExcitatoryInput(scaledInput);

        // Get current state
        var currentState = new ActivationState(scaleStates[scaleIdx]);

        // Evolve dynamics
        var newState = scale.evolve(currentState, scaleDeltaT);

        // Store new state
        System.arraycopy(newState.getActivations(), 0,
                        scaleStates[scaleIdx], 0, dimension);

        // Compute derivatives for coupling
        computeScaleDerivativesVectorized(scaleIdx);
    }

    /**
     * Preprocess input for specific scale (vectorized).
     */
    private double[] preprocessInputForScale(double[] input, int scaleIdx) {
        double[] result = new double[dimension];

        // Apply scale-specific filtering (vectorized)
        double filterStrength = parameters.getFilterStrength(scaleIdx);
        double filterCutoff = parameters.getFilterCutoff(scaleIdx);

        int i = 0;
        for (; i < loopBound; i += vectorLength) {
            var vInput = DoubleVector.fromArray(SPECIES, input, i);

            // Apply low-pass filter for slower scales
            if (scaleIdx > 0) {
                var vPrev = DoubleVector.fromArray(SPECIES,
                    scaleStates[scaleIdx - 1], i);
                vInput = vInput.mul(1.0 - filterStrength)
                              .add(vPrev.mul(filterStrength));
            }

            // Apply gain
            vInput = vInput.mul(parameters.getInputGain(scaleIdx));

            vInput.intoArray(result, i);
        }

        // Scalar tail
        for (; i < dimension; i++) {
            if (scaleIdx > 0) {
                result[i] = input[i] * (1.0 - filterStrength) +
                           scaleStates[scaleIdx - 1][i] * filterStrength;
            } else {
                result[i] = input[i];
            }
            result[i] *= parameters.getInputGain(scaleIdx);
        }

        return result;
    }

    /**
     * Compute derivatives for cross-scale coupling (vectorized).
     */
    private void computeScaleDerivativesVectorized(int scaleIdx) {
        var scale = scales[scaleIdx];
        var state = scaleStates[scaleIdx];
        var derivatives = scaleDerivatives[scaleIdx];

        // Use scale's internal derivative computation
        var currentState = new ActivationState(state);
        var nextState = scale.evolve(currentState, 0.001);  // Small dt for derivative

        // Compute derivative: (next - current) / dt
        int i = 0;
        for (; i < loopBound; i += vectorLength) {
            var vCurrent = DoubleVector.fromArray(SPECIES, state, i);
            var vNext = DoubleVector.fromArray(SPECIES,
                nextState.getActivations(), i);
            var vDeriv = vNext.sub(vCurrent).div(0.001);
            vDeriv.intoArray(derivatives, i);
        }

        for (; i < dimension; i++) {
            derivatives[i] = (nextState.getActivations()[i] - state[i]) / 0.001;
        }
    }

    /**
     * Apply cross-scale coupling (vectorized).
     */
    private void applyCrossScaleCouplingVectorized() {
        int numScales = scales.length;

        // Temporary array for coupling updates
        double[][] updates = new double[numScales][dimension];

        // Compute coupling from each scale to others
        for (int source = 0; source < numScales; source++) {
            for (int target = 0; target < numScales; target++) {
                if (source == target) continue;

                double coupling = crossScaleWeights[source][target];
                if (Math.abs(coupling) < 1e-10) continue;

                // Vectorized coupling computation
                computeCouplingVectorized(
                    scaleStates[source],
                    scaleDerivatives[source],
                    updates[target],
                    coupling
                );
            }
        }

        // Apply updates to states (vectorized)
        for (int s = 0; s < numScales; s++) {
            int i = 0;
            for (; i < loopBound; i += vectorLength) {
                var vState = DoubleVector.fromArray(SPECIES, scaleStates[s], i);
                var vUpdate = DoubleVector.fromArray(SPECIES, updates[s], i);
                vState = vState.add(vUpdate);

                // Apply bounds
                vState = vState.max(0.0).min(1.0);
                vState.intoArray(scaleStates[s], i);
            }

            for (; i < dimension; i++) {
                scaleStates[s][i] += updates[s][i];
                scaleStates[s][i] = Math.max(0.0, Math.min(1.0, scaleStates[s][i]));
            }
        }
    }

    /**
     * Compute coupling between scales (vectorized).
     */
    private void computeCouplingVectorized(double[] source, double[] sourceDeriv,
                                          double[] targetUpdate, double coupling) {
        // Coupling based on source activity and derivatives
        int i = 0;
        for (; i < loopBound; i += vectorLength) {
            var vSource = DoubleVector.fromArray(SPECIES, source, i);
            var vDeriv = DoubleVector.fromArray(SPECIES, sourceDeriv, i);
            var vUpdate = DoubleVector.fromArray(SPECIES, targetUpdate, i);

            // Coupling: target += coupling * (source * deriv)
            var vCoupling = vSource.mul(vDeriv).mul(coupling);
            vUpdate = vUpdate.add(vCoupling);
            vUpdate.intoArray(targetUpdate, i);
        }

        for (; i < dimension; i++) {
            targetUpdate[i] += coupling * source[i] * sourceDeriv[i];
        }
    }

    /**
     * Combine outputs from all scales (vectorized).
     */
    private void combineOutputsVectorized() {
        // Reset combined output
        Arrays.fill(combinedOutput, 0.0);

        // Weighted combination of scales
        for (int s = 0; s < scales.length; s++) {
            double weight = scaleWeights[s];

            int i = 0;
            for (; i < loopBound; i += vectorLength) {
                var vScale = DoubleVector.fromArray(SPECIES, scaleStates[s], i);
                var vOutput = DoubleVector.fromArray(SPECIES, combinedOutput, i);
                vOutput = vOutput.add(vScale.mul(weight));
                vOutput.intoArray(combinedOutput, i);
            }

            for (; i < dimension; i++) {
                combinedOutput[i] += scaleStates[s][i] * weight;
            }
        }

        // Normalize if weights don't sum to 1
        double weightSum = Arrays.stream(scaleWeights).sum();
        if (Math.abs(weightSum - 1.0) > 1e-6) {
            var vNorm = DoubleVector.broadcast(SPECIES, 1.0 / weightSum);

            int i = 0;
            for (; i < loopBound; i += vectorLength) {
                var vOutput = DoubleVector.fromArray(SPECIES, combinedOutput, i);
                vOutput = vOutput.mul(vNorm);
                vOutput.intoArray(combinedOutput, i);
            }

            for (; i < dimension; i++) {
                combinedOutput[i] /= weightSum;
            }
        }
    }

    /**
     * Initialize cross-scale coupling weights.
     */
    private void initializeCrossScaleCoupling() {
        int numScales = scales.length;
        crossScaleWeights = new double[numScales][numScales];
        couplingStrength = new double[numScales];

        // Default coupling pattern: fast -> medium -> slow
        for (int i = 0; i < numScales - 1; i++) {
            // Forward coupling (fast to slow)
            crossScaleWeights[i][i + 1] = parameters.getForwardCoupling();
            // Backward coupling (slow to fast)
            crossScaleWeights[i + 1][i] = parameters.getBackwardCoupling();
        }

        // Coupling strength based on time scale separation
        for (int i = 0; i < numScales; i++) {
            couplingStrength[i] = parameters.getCouplingStrength(i);
        }
    }

    /**
     * Get energy across all scales (vectorized).
     */
    public double computeTotalEnergy() {
        double totalEnergy = 0.0;

        // Compute energy for each scale in parallel
        var futures = new CompletableFuture[scales.length];

        for (int s = 0; s < scales.length; s++) {
            final int scale = s;
            futures[s] = CompletableFuture.supplyAsync(() -> {
                scales[scale].setState(new ActivationState(scaleStates[scale]));
                return scales[scale].computeEnergyVectorized();
            }, scalePool);
        }

        // Sum energies
        for (var future : futures) {
            try {
                totalEnergy += (Double) future.get();
            } catch (Exception e) {
                // Handle error
            }
        }

        return totalEnergy;
    }

    /**
     * Check convergence across all scales.
     */
    public boolean hasConverged(double tolerance) {
        for (int s = 0; s < scales.length; s++) {
            scales[s].setState(new ActivationState(scaleStates[s]));
            if (!scales[s].hasConvergedVectorized(tolerance * timeConstants[s])) {
                return false;
            }
        }
        return true;
    }

    public void reset() {
        for (var scale : scales) {
            scale.reset();
        }

        for (var state : scaleStates) {
            Arrays.fill(state, 0.0);
        }

        for (var deriv : scaleDerivatives) {
            Arrays.fill(deriv, 0.0);
        }

        Arrays.fill(combinedOutput, 0.0);
    }

    // Getters
    public double[] getCombinedOutput() {
        return combinedOutput.clone();
    }

    public double[][] getScaleStates() {
        double[][] copy = new double[scaleStates.length][];
        for (int i = 0; i < scaleStates.length; i++) {
            copy[i] = scaleStates[i].clone();
        }
        return copy;
    }

    public MultiScaleParameters getParameters() {
        return parameters;
    }
}