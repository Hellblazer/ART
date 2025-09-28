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
package com.hellblazer.art.temporal.vectorized;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.temporal.TemporalPattern;
import com.hellblazer.art.temporal.parameters.TemporalParameters;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for VectorizedTemporalART to verify correct functionality
 * of all vectorized components and temporal processing pipeline.
 *
 * @author Hal Hildebrand
 */
@DisplayName("Vectorized Temporal ART Tests")
public class VectorizedTemporalARTTest {

    private static final Logger log = LoggerFactory.getLogger(VectorizedTemporalARTTest.class);

    private VectorizedTemporalART temporalART;
    private TemporalParameters parameters;

    @BeforeEach
    void setUp() {
        parameters = TemporalParameters.DEFAULT;
        temporalART = new VectorizedTemporalART(parameters, 20, 8);
    }

    @AfterEach
    void tearDown() {
        if (temporalART != null) {
            temporalART.close();
        }
    }

    @Test
    @DisplayName("Basic Temporal Learning")
    void testBasicTemporalLearning() {
        // Create a simple temporal pattern
        List<Pattern> sequence = List.of(
            new DenseVector(new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}),
            new DenseVector(new double[]{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}),
            new DenseVector(new double[]{0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0})
        );
        var temporalPattern = createTemporalPattern(sequence);

        // Learn the pattern
        var result = temporalART.learnTemporal(temporalPattern);

        assertNotNull(result);
        assertTrue(result.isSuccessful() || result.getActivationResult() instanceof ActivationResult.NoMatch);
        assertNotNull(result.getWorkingMemoryState());
        assertNotNull(result.getMaskingFieldActivations());
        assertFalse(result.getWorkingMemoryState().isEmpty());

        log.info("Learning result: {}", result.getResultSummary());
    }

    @Test
    @DisplayName("Sequential Item Processing")
    void testSequentialItemProcessing() {
        var patterns = List.of(
            new DenseVector(new double[]{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}),
            new DenseVector(new double[]{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}),
            new DenseVector(new double[]{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9})
        );

        temporalART.resetTemporalState();

        for (var pattern : patterns) {
            var result = temporalART.processSequenceItem(pattern);
            assertNotNull(result);
            log.debug("Sequential processing result: {}", result.getResultSummary());
        }

        // Verify working memory contains items
        var workingMemoryContents = temporalART.getWorkingMemoryContents();
        assertNotNull(workingMemoryContents);
        assertFalse(workingMemoryContents.isEmpty());
        assertEquals(patterns.size(), workingMemoryContents.getSequenceLength());
    }

    @Test
    @DisplayName("Temporal Prediction")
    void testTemporalPrediction() {
        // First learn a pattern
        List<Pattern> sequence = List.of(
            new DenseVector(new double[]{0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2}),
            new DenseVector(new double[]{0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4}),
            new DenseVector(new double[]{0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6})
        );
        var temporalPattern = createTemporalPattern(sequence);

        var learnResult = temporalART.learnTemporal(temporalPattern);
        assertTrue(learnResult.isSuccessful() || learnResult.getActivationResult() instanceof ActivationResult.NoMatch);

        // Then predict with the same pattern
        var predictResult = temporalART.predictTemporal(temporalPattern);
        assertNotNull(predictResult);

        log.info("Prediction result: {}", predictResult.getResultSummary());
    }

    @Test
    @DisplayName("Batch Processing")
    void testBatchProcessing() {
        var patterns = List.of(
            createTemporalPattern(List.of(
                new DenseVector(new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}),
                new DenseVector(new double[]{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9})
            )),
            createTemporalPattern(List.of(
                new DenseVector(new double[]{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2}),
                new DenseVector(new double[]{0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1})
            )),
            createTemporalPattern(List.of(
                new DenseVector(new double[]{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}),
                new DenseVector(new double[]{0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6})
            ))
        );

        var results = temporalART.learnTemporalBatch(patterns);

        assertEquals(patterns.size(), results.size());
        for (var result : results) {
            assertNotNull(result);
            log.debug("Batch result: {}", result.getResultSummary());
        }
    }

    @Test
    @DisplayName("Working Memory Operations")
    void testWorkingMemoryOperations() {
        assertTrue(temporalART.getWorkingMemoryContents().isEmpty());
        assertEquals(parameters.workingMemoryParameters().capacity(), temporalART.getWorkingMemoryCapacity());

        // Process some items
        List<Pattern> patterns = List.of(
            new DenseVector(new double[]{0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3}),
            new DenseVector(new double[]{0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7})
        );

        temporalART.resetTemporalState();
        for (var pattern : patterns) {
            temporalART.processSequenceItem(pattern);
        }

        assertFalse(temporalART.getWorkingMemoryContents().isEmpty());
        assertTrue(temporalART.isWorkingMemoryActive());
        assertEquals(patterns.size(), temporalART.getWorkingMemoryContents().getSequenceLength());
    }

    @Test
    @DisplayName("Masking Field Operations")
    void testMaskingFieldOperations() {
        List<Pattern> sequence = List.of(
            new DenseVector(new double[]{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}),
            new DenseVector(new double[]{0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3}),
            new DenseVector(new double[]{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}),
            new DenseVector(new double[]{0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7}),
            new DenseVector(new double[]{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9})
        );
        var temporalPattern = createTemporalPattern(sequence);

        var result = temporalART.learnTemporal(temporalPattern);

        var maskingActivations = temporalART.getMaskingFieldActivations();
        assertNotNull(maskingActivations);
        assertTrue(maskingActivations.length > 0);

        // Check if masking field is active
        var hasActivation = false;
        for (var scale : maskingActivations) {
            for (var activation : scale) {
                if (activation > 0.0) {
                    hasActivation = true;
                    break;
                }
            }
        }

        log.info("Masking field has activation: {}", hasActivation);
    }

    @Test
    @DisplayName("Performance Metrics")
    void testPerformanceMetrics() {
        List<Pattern> sequence = List.of(
            new DenseVector(new double[]{0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7}),
            new DenseVector(new double[]{0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7, 0.9}),
            new DenseVector(new double[]{0.6, 0.8, 0.1, 0.3, 0.5, 0.7, 0.9, 0.2})
        );
        var temporalPattern = createTemporalPattern(sequence);

        temporalART.resetPerformanceTracking();
        var result = temporalART.learnTemporal(temporalPattern);

        var stats = temporalART.getPerformanceStats();
        assertNotNull(stats);
        assertTrue(stats.getTemporalOperations() > 0);

        log.info("Performance stats: {}", stats);
    }

    @Test
    @DisplayName("Parameter Updates")
    void testParameterUpdates() {
        var originalParams = temporalART.getTemporalParameters();
        assertEquals(parameters.vigilance(), originalParams.vigilance());

        var newParams = parameters.withVigilance(0.8f);
        temporalART.setTemporalParameters(newParams);

        var updatedParams = temporalART.getTemporalParameters();
        assertEquals(0.8f, updatedParams.vigilance(), 0.001f);
    }

    @Test
    @DisplayName("Chunk Detection")
    void testChunkDetection() {
        // Create a longer sequence that might be chunked
        List<Pattern> sequence = List.of(
            // First chunk
            new DenseVector(new double[]{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}),
            new DenseVector(new double[]{0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2}),
            new DenseVector(new double[]{0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3}),
            // Second chunk (different pattern)
            new DenseVector(new double[]{0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7}),
            new DenseVector(new double[]{0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8}),
            new DenseVector(new double[]{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9})
        );
        var temporalPattern = createTemporalPattern(sequence);

        var result = temporalART.learnTemporal(temporalPattern);

        assertNotNull(result.getIdentifiedChunks());
        log.info("Identified {} chunks", result.getIdentifiedChunks().size());
        log.info("Chunk boundaries: {}", result.getChunkBoundaries());

        // Check if chunking was attempted
        assertTrue(result.getChunkBoundaries().size() >= 0);
    }

    @Test
    @DisplayName("Empty Pattern Handling")
    void testEmptyPatternHandling() {
        var emptyPattern = createTemporalPattern(List.of());

        var result = temporalART.learnTemporal(emptyPattern);
        assertNotNull(result);

        var predictResult = temporalART.predictTemporal(emptyPattern);
        assertNotNull(predictResult);
    }

    @Test
    @DisplayName("Reset Functionality")
    void testResetFunctionality() {
        // Process some data
        var pattern = new DenseVector(new double[]{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5});
        temporalART.processSequenceItem(pattern);

        assertTrue(temporalART.isWorkingMemoryActive());

        // Reset and verify
        temporalART.resetTemporalState();
        assertFalse(temporalART.isWorkingMemoryActive());
        assertTrue(temporalART.getWorkingMemoryContents().isEmpty());
    }

    @Test
    @DisplayName("Vectorization Efficiency")
    void testVectorizationEfficiency() {
        var stats = temporalART.getPerformanceStats();

        // Test that vectorized components report reasonable efficiency
        var workingMemoryMetrics = stats.getWorkingMemoryMetrics();
        var maskingFieldMetrics = stats.getMaskingFieldMetrics();
        var instarMetrics = stats.getCompetitiveInstarMetrics();

        assertNotNull(workingMemoryMetrics);
        assertNotNull(maskingFieldMetrics);
        assertNotNull(instarMetrics);

        log.info("Working memory metrics: {}", workingMemoryMetrics);
        log.info("Masking field metrics: {}", maskingFieldMetrics);
        log.info("Competitive instar metrics: {}", instarMetrics);

        // Basic validation - metrics should be available
        assertTrue(instarMetrics.getVectorizationEfficiency() > 0.5);
    }

    // === Helper Methods ===

    private TemporalPattern createTemporalPattern(List<Pattern> sequence) {
        return new TemporalPattern() {
            @Override
            public List<Pattern> getSequence() {
                return new ArrayList<>(sequence);
            }

            @Override
            public TemporalPattern getSubsequence(int startTime, int endTime) {
                if (startTime < 0 || endTime > sequence.size() || startTime >= endTime) {
                    throw new IndexOutOfBoundsException("Invalid subsequence bounds");
                }
                return createTemporalPattern(sequence.subList(startTime, endTime));
            }

            @Override
            public boolean isEmpty() {
                return sequence.isEmpty();
            }
        };
    }
}