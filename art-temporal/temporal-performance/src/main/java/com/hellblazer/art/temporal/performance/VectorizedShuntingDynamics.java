package com.hellblazer.art.temporal.performance;

import com.hellblazer.art.temporal.core.ActivationState;
import com.hellblazer.art.temporal.dynamics.ShuntingParameters;
import jdk.incubator.vector.*;

/**
 * High-performance vectorized implementation of shunting dynamics.
 * Uses Java Vector API for SIMD acceleration.
 */
public class VectorizedShuntingDynamics {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private final ShuntingParameters parameters;
    private final int dimension;
    private final int vectorLength;
    private final int loopBound;

    private double[] activations;
    private double[] excitatory;
    private double[] inhibitory;
    private double[] derivatives;

    // Pre-computed weight matrices for efficiency
    private double[] excitatoryWeights;
    private double[] inhibitoryWeights;

    public VectorizedShuntingDynamics(ShuntingParameters parameters, int dimension) {
        this.parameters = parameters;
        this.dimension = dimension;
        this.vectorLength = SPECIES.length();
        this.loopBound = SPECIES.loopBound(dimension);

        // Allocate arrays aligned for vector operations
        this.activations = new double[dimension];
        this.excitatory = new double[dimension];
        this.inhibitory = new double[dimension];
        this.derivatives = new double[dimension];

        // Pre-compute weight matrices
        precomputeWeights();
    }

    /**
     * Vectorized evolution of shunting dynamics.
     */
    public ActivationState evolve(ActivationState currentState, double deltaT) {
        var current = currentState.getActivations();
        System.arraycopy(current, 0, activations, 0, dimension);

        // Compute derivatives using vectorized operations
        computeDerivativesVectorized();

        // Update state using Euler integration (vectorized)
        var result = new double[dimension];
        int i = 0;

        // Main vectorized loop
        for (; i < loopBound; i += vectorLength) {
            var vActivations = DoubleVector.fromArray(SPECIES, activations, i);
            var vDerivatives = DoubleVector.fromArray(SPECIES, derivatives, i);

            // Euler step: x_new = x + dt * dx/dt
            var vDelta = vDerivatives.mul(deltaT);
            var vResult = vActivations.add(vDelta);

            // Apply bounds
            vResult = vResult.max(parameters.getFloor())
                            .min(parameters.getCeiling());

            vResult.intoArray(result, i);
        }

        // Scalar tail
        for (; i < dimension; i++) {
            result[i] = activations[i] + deltaT * derivatives[i];
            result[i] = Math.max(parameters.getFloor(),
                                Math.min(parameters.getCeiling(), result[i]));
        }

        return new ActivationState(result);
    }

    /**
     * Compute derivatives using vectorized operations.
     */
    private void computeDerivativesVectorized() {
        // Reset derivatives
        var zeros = DoubleVector.zero(SPECIES);
        for (int i = 0; i < loopBound; i += vectorLength) {
            zeros.intoArray(derivatives, i);
        }

        // Compute excitation and inhibition in parallel
        computeExcitationVectorized();
        computeInhibitionVectorized();

        // Apply shunting equation (vectorized)
        double ceiling = parameters.getCeiling();
        double floor = parameters.getFloor();

        int i = 0;
        for (; i < loopBound; i += vectorLength) {
            var vX = DoubleVector.fromArray(SPECIES, activations, i);
            var vExcite = DoubleVector.fromArray(SPECIES, excitatory, i);
            var vInhibit = DoubleVector.fromArray(SPECIES, inhibitory, i);
            var vDecay = DoubleVector.fromArray(SPECIES, parameters.getDecayRates(), i);

            // dx/dt = -A*x + (B-x)*E - (x-floor)*I
            var vDecayTerm = vX.mul(vDecay).neg();
            var vExciteTerm = vExcite.mul(ceiling - vX.reduceLanes(VectorOperators.ADD) / vectorLength);
            var vInhibitTerm = vInhibit.mul(vX.reduceLanes(VectorOperators.ADD) / vectorLength - floor);

            var vDerivative = vDecayTerm.add(vExciteTerm).sub(vInhibitTerm);
            vDerivative.intoArray(derivatives, i);
        }

        // Scalar tail
        for (; i < dimension; i++) {
            derivatives[i] = -parameters.getDecayRate(i) * activations[i] +
                            (ceiling - activations[i]) * excitatory[i] -
                            (activations[i] - floor) * inhibitory[i];
        }
    }

    /**
     * Vectorized computation of excitation.
     */
    private void computeExcitationVectorized() {
        // Use pre-computed weights for matrix-vector multiplication
        matrixVectorMultiplyVectorized(activations, excitatoryWeights, excitatory);

        // Add self-excitation
        if (parameters.getSelfExcitation() > 0) {
            double selfExcite = parameters.getSelfExcitation();

            int i = 0;
            for (; i < loopBound; i += vectorLength) {
                var vAct = DoubleVector.fromArray(SPECIES, activations, i);
                var vExcite = DoubleVector.fromArray(SPECIES, excitatory, i);
                var vResult = vExcite.add(vAct.mul(selfExcite));
                vResult.intoArray(excitatory, i);
            }

            for (; i < dimension; i++) {
                excitatory[i] += activations[i] * selfExcite;
            }
        }
    }

    /**
     * Vectorized computation of inhibition.
     */
    private void computeInhibitionVectorized() {
        matrixVectorMultiplyVectorized(activations, inhibitoryWeights, inhibitory);
    }

    /**
     * Vectorized matrix-vector multiplication.
     */
    private void matrixVectorMultiplyVectorized(double[] vector, double[] matrix, double[] result) {
        // Reset result
        var zeros = DoubleVector.zero(SPECIES);
        for (int i = 0; i < loopBound; i += vectorLength) {
            zeros.intoArray(result, i);
        }

        // Perform multiplication
        for (int i = 0; i < dimension; i++) {
            double vi = vector[i];
            if (Math.abs(vi) < 1e-10) continue;  // Skip near-zero values

            int j = 0;
            for (; j < loopBound; j += vectorLength) {
                var vWeight = DoubleVector.fromArray(SPECIES, matrix, i * dimension + j);
                var vResult = DoubleVector.fromArray(SPECIES, result, j);
                vResult = vResult.add(vWeight.mul(vi));
                vResult.intoArray(result, j);
            }

            for (; j < dimension; j++) {
                result[j] += matrix[i * dimension + j] * vi;
            }
        }
    }

    /**
     * Pre-compute weight matrices for efficiency.
     */
    private void precomputeWeights() {
        excitatoryWeights = new double[dimension * dimension];
        inhibitoryWeights = new double[dimension * dimension];

        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if (i != j) {
                    double distance = Math.abs(i - j);

                    // Excitatory weight (Gaussian)
                    double exciteSigma = parameters.getExcitatoryRange();
                    excitatoryWeights[i * dimension + j] =
                        parameters.getExcitatoryStrength() *
                        Math.exp(-distance * distance / (2.0 * exciteSigma * exciteSigma));

                    // Inhibitory weight (broader Gaussian)
                    double inhibitSigma = parameters.getInhibitoryRange();
                    inhibitoryWeights[i * dimension + j] =
                        parameters.getInhibitoryStrength() *
                        Math.exp(-distance * distance / (2.0 * inhibitSigma * inhibitSigma));
                }
            }
        }
    }

    /**
     * Vectorized energy computation for Lyapunov analysis.
     */
    public double computeEnergyVectorized() {
        double energy = 0.0;

        // Self-energy terms (vectorized)
        int i = 0;
        for (; i < loopBound; i += vectorLength) {
            var vAct = DoubleVector.fromArray(SPECIES, activations, i);
            var vDecay = DoubleVector.fromArray(SPECIES, parameters.getDecayRates(), i);
            var vEnergy = vAct.mul(vAct).mul(vDecay).mul(0.5);
            energy += vEnergy.reduceLanes(VectorOperators.ADD);
        }

        // Scalar tail
        for (; i < dimension; i++) {
            energy += 0.5 * parameters.getDecayRate(i) * activations[i] * activations[i];
        }

        // Interaction energy
        for (i = 0; i < dimension; i++) {
            for (int j = i + 1; j < dimension; j++) {
                double excWeight = excitatoryWeights[i * dimension + j];
                double inhWeight = inhibitoryWeights[i * dimension + j];
                energy -= excWeight * activations[i] * activations[j];
                energy += inhWeight * activations[i] * activations[j];
            }
        }

        return energy;
    }

    /**
     * Fast convergence check using vectorized norm.
     */
    public boolean hasConvergedVectorized(double tolerance) {
        computeDerivativesVectorized();

        double maxChange = 0.0;

        // Vectorized max reduction
        int i = 0;
        for (; i < loopBound; i += vectorLength) {
            var vDeriv = DoubleVector.fromArray(SPECIES, derivatives, i);
            var vAbs = vDeriv.abs();
            maxChange = Math.max(maxChange, vAbs.reduceLanes(VectorOperators.MAX));
        }

        // Scalar tail
        for (; i < dimension; i++) {
            maxChange = Math.max(maxChange, Math.abs(derivatives[i]));
        }

        return maxChange < tolerance;
    }

    // Standard interface methods

    public void setExcitatoryInput(double[] input) {
        System.arraycopy(input, 0, excitatory, 0, Math.min(input.length, dimension));
    }

    public void setInhibitoryInput(double[] input) {
        System.arraycopy(input, 0, inhibitory, 0, Math.min(input.length, dimension));
    }

    public void clearInputs() {
        var zeros = DoubleVector.zero(SPECIES);
        for (int i = 0; i < loopBound; i += vectorLength) {
            zeros.intoArray(excitatory, i);
            zeros.intoArray(inhibitory, i);
        }
        for (int i = loopBound; i < dimension; i++) {
            excitatory[i] = 0.0;
            inhibitory[i] = 0.0;
        }
    }

    public void reset() {
        double initial = parameters.getInitialActivation();
        var vInitial = DoubleVector.broadcast(SPECIES, initial);

        for (int i = 0; i < loopBound; i += vectorLength) {
            vInitial.intoArray(activations, i);
        }
        for (int i = loopBound; i < dimension; i++) {
            activations[i] = initial;
        }

        clearInputs();
    }

    public ActivationState getState() {
        return new ActivationState(activations.clone());
    }

    public void setState(ActivationState state) {
        System.arraycopy(state.getActivations(), 0, activations, 0, dimension);
    }

    public double[] getActivations() {
        return activations.clone();
    }

    public int getDimension() {
        return dimension;
    }

    public ShuntingParameters getParameters() {
        return parameters;
    }
}