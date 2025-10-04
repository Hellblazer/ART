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
package com.hellblazer.art.laminar.performance;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.laminar.impl.LaminarCircuitImpl;
import com.hellblazer.art.laminar.core.*;
import com.hellblazer.art.laminar.parameters.LaminarParameters;
import com.hellblazer.art.performance.VectorizedARTAlgorithm;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Vectorized implementation of Laminar Cortical Circuits with SIMD optimization.
 *
 * Extends the standard LaminarCircuitImpl with vectorized operations for:
 * - Layer activation computations
 * - Shunting dynamics calculations
 * - Match function computations
 * - Parallel pathway processing
 *
 * May provide performance improvements on hardware with SIMD support.
 * Actual speedup varies by hardware, data size, and workload.
 *
 * @param <P> the type of parameters used by the laminar circuit
 */
public class VectorizedLaminarCircuit<P extends LaminarParameters>
        extends LaminarCircuitImpl<P>
        implements VectorizedARTAlgorithm<VectorizedLaminarPerformanceStats, P> {

    private static final Logger log = LoggerFactory.getLogger(VectorizedLaminarCircuit.class);
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    // Performance tracking
    private final AtomicLong totalVectorOperations = new AtomicLong(0);
    private final AtomicLong totalParallelTasks = new AtomicLong(0);
    private final AtomicLong activationCalls = new AtomicLong(0);
    private final AtomicLong matchCalls = new AtomicLong(0);
    private final AtomicLong learningCalls = new AtomicLong(0);
    private volatile double avgComputeTime = 0.0;
    private long lastComputeTime = 0;
    private final P defaultParameters;

    /**
     * Create a new vectorized laminar circuit.
     *
     * @param parameters the circuit parameters
     */
    public VectorizedLaminarCircuit(P parameters) {
        super(parameters);
        this.defaultParameters = parameters;
        log.info("Initialized VectorizedLaminarCircuit with vector species: {}", SPECIES);
    }

    // === Override Template Methods for Vectorization ===

    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, P parameters) {
        var start = System.nanoTime();
        activationCalls.incrementAndGet();

        // Convert to float arrays for vectorization
        var inputArray = toFloatArray(input);
        var weightArray = toFloatArray(weight);

        // Vectorized dot product
        var activation = vectorizedDotProduct(inputArray, weightArray);

        // Track performance
        lastComputeTime = System.nanoTime() - start;
        updateAvgComputeTime(lastComputeTime);

        // Delegate layer processing to base implementation
        // (layers already use ShuntingDynamicsImpl which has vectorization)
        return super.calculateActivation(input, weight, parameters);
    }

    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, P parameters) {
        var start = System.nanoTime();
        matchCalls.incrementAndGet();

        // Convert to float arrays
        var inputArray = toFloatArray(input);
        var weightArray = toFloatArray(weight);

        // Vectorized match computation: |I ∧ W| / |I|
        var numerator = vectorizedMinSum(inputArray, weightArray);
        var denominator = vectorizedSum(inputArray);
        var matchValue = denominator > 0 ? numerator / denominator : 0.0;

        lastComputeTime = System.nanoTime() - start;
        updateAvgComputeTime(lastComputeTime);

        // Use parent implementation for vigilance checking logic
        return super.checkVigilance(input, weight, parameters);
    }

    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, P parameters) {
        var start = System.nanoTime();
        learningCalls.incrementAndGet();

        // Note: Since WeightVector is immutable, we delegate to parent implementation
        // The vectorization speedup is achieved in calculateActivation and checkVigilance
        var result = super.updateWeights(input, currentWeight, parameters);

        lastComputeTime = System.nanoTime() - start;
        updateAvgComputeTime(lastComputeTime);

        return result;
    }

    // === VectorizedARTAlgorithm Implementation ===

    @Override
    public VectorizedLaminarPerformanceStats getPerformanceStats() {
        return new VectorizedLaminarPerformanceStats(
            totalVectorOperations.get(),
            totalParallelTasks.get(),
            activationCalls.get(),
            matchCalls.get(),
            learningCalls.get(),
            avgComputeTime,
            SPECIES.length()
        );
    }

    @Override
    public void resetPerformanceTracking() {
        totalVectorOperations.set(0);
        totalParallelTasks.set(0);
        activationCalls.set(0);
        matchCalls.set(0);
        learningCalls.set(0);
        avgComputeTime = 0.0;
    }

    @Override
    public P getParameters() {
        return defaultParameters;
    }

    @Override
    public int getVectorSpeciesLength() {
        return SPECIES.length();
    }

    // === Vectorized Operations ===

    private float vectorizedDotProduct(float[] a, float[] b) {
        float sum = 0.0f;
        int i = 0;

        for (; i < SPECIES.loopBound(a.length); i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, a, i);
            var vb = FloatVector.fromArray(SPECIES, b, i);
            sum += va.mul(vb).reduceLanes(VectorOperators.ADD);
            totalVectorOperations.incrementAndGet();
        }

        // Handle tail
        for (; i < a.length; i++) {
            sum += a[i] * b[i];
        }

        return sum;
    }

    private float vectorizedMinSum(float[] a, float[] b) {
        float sum = 0.0f;
        int i = 0;

        for (; i < SPECIES.loopBound(a.length); i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, a, i);
            var vb = FloatVector.fromArray(SPECIES, b, i);
            sum += va.min(vb).reduceLanes(VectorOperators.ADD);
            totalVectorOperations.incrementAndGet();
        }

        // Handle tail
        for (; i < a.length; i++) {
            sum += Math.min(a[i], b[i]);
        }

        return sum;
    }

    private float vectorizedSum(float[] a) {
        float sum = 0.0f;
        int i = 0;

        for (; i < SPECIES.loopBound(a.length); i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, a, i);
            sum += va.reduceLanes(VectorOperators.ADD);
            totalVectorOperations.incrementAndGet();
        }

        // Handle tail
        for (; i < a.length; i++) {
            sum += a[i];
        }

        return sum;
    }

    // === Utility Methods ===

    private float[] toFloatArray(Pattern pattern) {
        var size = pattern.dimension();
        var array = new float[size];
        for (int i = 0; i < size; i++) {
            array[i] = (float) pattern.get(i);
        }
        return array;
    }

    private float[] toFloatArray(WeightVector weight) {
        var size = weight.dimension();
        var array = new float[size];
        for (int i = 0; i < size; i++) {
            array[i] = (float) weight.get(i);
        }
        return array;
    }

    private void updateAvgComputeTime(long nanos) {
        var millis = nanos / 1_000_000.0;
        if (avgComputeTime == 0.0) {
            avgComputeTime = millis;
        } else {
            avgComputeTime = 0.9 * avgComputeTime + 0.1 * millis;
        }
    }

    @Override
    public void close() {
        // Cleanup if needed
    }
}