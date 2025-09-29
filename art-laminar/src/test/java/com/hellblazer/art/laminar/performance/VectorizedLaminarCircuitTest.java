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

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.laminar.builders.LaminarCircuitBuilder;
import com.hellblazer.art.laminar.impl.DefaultLaminarParameters;
import com.hellblazer.art.laminar.impl.DefaultLearningParameters;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import static org.junit.jupiter.api.Assertions.*;
import com.hellblazer.art.core.results.ActivationResult;

import com.hellblazer.art.laminar.core.LayerType;
import com.hellblazer.art.laminar.core.PathwayType;
import com.hellblazer.art.laminar.impl.AbstractLayer;
import com.hellblazer.art.laminar.impl.AbstractPathway;

/**
 * Test suite for VectorizedLaminarCircuit.
 */
class VectorizedLaminarCircuitTest {

    private VectorizedLaminarCircuit<DefaultLaminarParameters> circuit;
    private DefaultLaminarParameters parameters;

    @BeforeEach
    void setUp() {
        parameters = DefaultLaminarParameters.builder()
            .withLearningParameters(new DefaultLearningParameters(0.5, 0.0, false, 0.0))
            .withVigilance(0.8)
            .build();

        circuit = new VectorizedLaminarCircuit<>(parameters);

        // Add layers and connections directly
        circuit.addLayer(new TestInputLayer("input", 4, false), 0);
        circuit.addLayer(new TestFeatureLayer("feature", 4), 1);
        circuit.addLayer(new TestCategoryLayer("category", 10), 2);

        circuit.connectLayers(new TestBottomUpPathway("bu-input-feature", "input", "feature"));
        circuit.connectLayers(new TestBottomUpPathway("bu-feature-category", "feature", "category"));
        circuit.connectLayers(new TestTopDownPathway("td-category-feature", "category", "feature"));
        circuit.connectLayers(new TestLateralPathway("lat-feature", "feature", "feature"));
    }

    @Test
    void testVectorizedOperations() {
        var pattern = new DenseVector(new double[]{0.5, 0.5, 0.5, 0.5});

        // Perform learning
        var result = circuit.learn(pattern, parameters);

        assertNotNull(result);
        assertEquals(1, circuit.getCategoryCount());

        // Verify the circuit works correctly
        var predictResult = circuit.predict(pattern, parameters);
        assertNotNull(predictResult);

        // Check if it's a successful prediction
        if (predictResult instanceof ActivationResult.Success success) {
            assertEquals(0, success.categoryIndex());
        } else {
            fail("Expected successful prediction");
        }
    }

    @Test
    void testPerformanceMetrics() {
        // Process multiple patterns
        for (int i = 0; i < 10; i++) {
            var pattern = new DenseVector(new double[]{
                Math.random(), Math.random(), Math.random(), Math.random()
            });
            circuit.learn(pattern, parameters);
        }

        var stats = circuit.getPerformanceStats();

        // Basic sanity checks
        assertNotNull(stats);
        assertTrue(stats.vectorLaneWidth() > 0, "Vector lane width should be positive");

        // The circuit should have learned multiple categories
        assertTrue(circuit.getCategoryCount() > 0, "Should have learned categories");
    }

    @Test
    void testSpeedupCalculation() {
        circuit.resetPerformanceTracking();

        // Process a pattern
        var pattern = new DenseVector(new double[]{0.7, 0.3, 0.2, 0.1});
        circuit.learn(pattern, parameters);

        var stats = circuit.getPerformanceStats();
        double speedup = stats.getEstimatedSpeedup();

        // Speedup should be >= 1 (at least as fast as scalar)
        assertTrue(speedup >= 1.0, "Speedup should be at least 1x");
    }

    @Test
    void testResourceCleanup() {
        // Verify close doesn't throw
        assertDoesNotThrow(() -> circuit.close());
    }

    @Test
    void testCompatibilityWithBaseImplementation() {
        // Create standard circuit for comparison
        var standardParams = DefaultLaminarParameters.builder()
            .withLearningParameters(new DefaultLearningParameters(0.5, 0.0, false, 0.0))
            .withVigilance(0.8)
            .build();

        var standardCircuit = new LaminarCircuitBuilder<DefaultLaminarParameters>()
            .withParameters(standardParams)
            .withInputLayer(4, false)
            .withFeatureLayer(4)
            .withCategoryLayer(10)
            .withStandardConnections()
            .build();

        // Process same pattern on both
        var pattern = new DenseVector(new double[]{0.6, 0.4, 0.3, 0.1});

        var vectorizedResult = circuit.learn(pattern, parameters);
        var standardResult = standardCircuit.learn(pattern, standardParams);

        // Both should learn the pattern
        assertNotNull(vectorizedResult);
        assertNotNull(standardResult);
        assertEquals(1, circuit.getCategoryCount());
        assertEquals(1, standardCircuit.getCategoryCount());
    }

    // Test helper classes
    static class TestInputLayer extends AbstractLayer {
        private final boolean complementCoding;

        public TestInputLayer(String id, int size, boolean complementCoding) {
            super(id, complementCoding ? size * 2 : size, LayerType.INPUT);
            this.complementCoding = complementCoding;
        }
    }

    static class TestFeatureLayer extends AbstractLayer {
        public TestFeatureLayer(String id, int size) {
            super(id, size, LayerType.FEATURE);
        }
    }

    static class TestCategoryLayer extends AbstractLayer {
        public TestCategoryLayer(String id, int size) {
            super(id, size, LayerType.CATEGORY);
        }
    }

    static class TestBottomUpPathway extends AbstractPathway {
        public TestBottomUpPathway(String id, String sourceId, String targetId) {
            super(id, sourceId, targetId, PathwayType.BOTTOM_UP);
        }
    }

    static class TestTopDownPathway extends AbstractPathway {
        public TestTopDownPathway(String id, String sourceId, String targetId) {
            super(id, sourceId, targetId, PathwayType.TOP_DOWN);
        }
    }

    static class TestLateralPathway extends AbstractPathway {
        public TestLateralPathway(String id, String sourceId, String targetId) {
            super(id, sourceId, targetId, PathwayType.LATERAL);
        }
    }
}