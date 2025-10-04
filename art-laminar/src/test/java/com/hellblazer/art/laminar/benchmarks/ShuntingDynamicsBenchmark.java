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
package com.hellblazer.art.laminar.benchmarks;

import com.hellblazer.art.laminar.core.LayerType;
import com.hellblazer.art.laminar.impl.AbstractLayer;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * Microbenchmark for shunting dynamics computations in laminar layers.
 *
 * Tests the performance of the delegated ShuntingDynamicsImpl across
 * different layer sizes and activation patterns.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgs = {
    "--add-modules=jdk.incubator.vector",
    "--enable-preview",
    "-Xmx1g"
})
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 2)
public class ShuntingDynamicsBenchmark {

    @Param({"10", "100", "1000"})
    private int layerSize;

    @Param({"0.1", "0.5", "0.9"})
    private double activationLevel;

    private TestLayer layer;
    private double[] input;
    private Random random;

    @Setup(Level.Trial)
    public void setup() {
        random = new Random(42);

        // Create test layer
        layer = new TestLayer("test", layerSize, LayerType.FEATURE);

        // Generate input pattern
        input = new double[layerSize];
        for (int i = 0; i < layerSize; i++) {
            if (random.nextDouble() < activationLevel) {
                input[i] = random.nextDouble();
            }
        }

        // Initialize layer with random activations
        layer.setActivations(input);
    }

    @Benchmark
    public void updateActivations(Blackhole bh) {
        layer.updateActivations(0.01); // Small timestep
        bh.consume(layer.getActivations());
    }

    @Benchmark
    public void computeShuntingStep(Blackhole bh) {
        var excitation = generateSignal(layerSize, 0.5);
        var inhibition = generateSignal(layerSize, 0.3);

        layer.applyShuntingDynamics(excitation, inhibition, 0.01);
        bh.consume(layer.getActivations());
    }

    @Benchmark
    public void processFullCycle(Blackhole bh) {
        for (int i = 0; i < 10; i++) {
            layer.updateActivations(0.01);
            var excitation = generateSignal(layerSize, 0.5);
            var inhibition = generateSignal(layerSize, 0.3);
            layer.applyShuntingDynamics(excitation, inhibition, 0.01);
        }
        bh.consume(layer.getActivations());
    }

    private double[] generateSignal(int size, double strength) {
        var signal = new double[size];
        for (int i = 0; i < size; i++) {
            signal[i] = random.nextDouble() * strength;
        }
        return signal;
    }

    /**
     * Test layer implementation for benchmarking.
     */
    static class TestLayer extends AbstractLayer {
        private double[] currentActivations;

        public TestLayer(String id, int size, LayerType type) {
            super(id, size, type);
            this.currentActivations = new double[size];
        }

        public void setActivations(double[] activations) {
            System.arraycopy(activations, 0, currentActivations, 0, activations.length);
        }

        public double[] getActivations() {
            return currentActivations;
        }

        public void updateActivations(double timeStep) {
            // Simulate shunting dynamics update
            for (int i = 0; i < currentActivations.length; i++) {
                // Simplified shunting equation: dx/dt = -Ax + (B-x)E - (x+C)I
                double decay = -0.1 * currentActivations[i];
                currentActivations[i] += decay * timeStep;
                currentActivations[i] = Math.max(0, Math.min(1, currentActivations[i]));
            }
        }

        public void applyShuntingDynamics(double[] excitation, double[] inhibition, double timeStep) {
            for (int i = 0; i < currentActivations.length; i++) {
                double x = currentActivations[i];
                double e = excitation[i];
                double in = inhibition[i];

                // Shunting equation
                double dx = -0.1 * x + (1.0 - x) * e - (x + 0.1) * in;
                currentActivations[i] = x + dx * timeStep;
                currentActivations[i] = Math.max(0, Math.min(1, currentActivations[i]));
            }
        }
    }
}