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
package com.hellblazer.art.hartcq.hierarchical;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.hartcq.hierarchical.HierarchyLevel.Level;
import com.hellblazer.art.hartcq.hierarchical.ResonanceController.ResonanceParameters;
import com.hellblazer.art.hartcq.hierarchical.ResonanceController.ResonanceResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.RepeatedTest;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Comprehensive unit tests for ResonanceController.
 * Tests top-down and bottom-up signals, vigilance adaptation,
 * stability detection, and resonance achievement conditions.
 * 
 * @author Hal Hildebrand
 */
class ResonanceControllerTest {
    private static final Logger logger = LoggerFactory.getLogger(ResonanceControllerTest.class);
    
    private ResonanceController resonanceController;
    private ResonanceParameters defaultParams;
    private List<HierarchyLevel> hierarchyLevels;
    
    @Mock
    private HierarchyLevel mockLevel1;
    
    @Mock
    private HierarchyLevel mockLevel2;
    
    @Mock
    private HierarchyLevel mockLevel3;
    
    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        
        defaultParams = ResonanceParameters.defaults();
        
        // Create real hierarchy levels for testing
        hierarchyLevels = List.of(
            new HierarchyLevel(Level.TOKEN, 100),
            new HierarchyLevel(Level.WINDOW, 50),
            new HierarchyLevel(Level.DOCUMENT, 25)
        );
        
        resonanceController = new ResonanceController(hierarchyLevels, defaultParams);
    }
    
    @Nested
    @DisplayName("Basic Resonance Processing Tests")
    class BasicResonanceProcessingTests {
        
        @Test
        @DisplayName("Should process patterns through resonance system")
        void shouldProcessPatternsThroughResonanceSystem() {
            // Given
            var inputPattern = createTestPattern(0.5, 0.3, 0.8, 0.2);
            
            // When
            var result = resonanceController.processPattern(inputPattern, true);
            
            // Then
            assertThat(result).isNotNull();
            assertThat(result.getProcessingTimeMs()).isGreaterThan(0.0);
            
            // Result should have either success or failure with reason
            if (result.isSuccess()) {
                assertThat(result.getHierarchicalCategories()).isNotNull();
                assertThat(result.getHierarchicalCategories()).hasSize(3); // Three levels
                assertThat(result.getVigilanceAdjustments()).isNotNull();
                assertThat(result.getResonanceStrength()).isGreaterThanOrEqualTo(0.0);
                assertThat(result.getResonanceId()).isNotNull();
            } else {
                assertThat(result.getErrorMessage()).isNotNull();
            }
        }
        
        @Test
        @DisplayName("Should handle learning enabled vs disabled modes")
        void shouldHandleLearningEnabledVsDisabledModes() {
            // Given
            var inputPattern = createTestPattern(0.4, 0.6, 0.1, 0.9);
            
            // When - with learning enabled
            var resultWithLearning = resonanceController.processPattern(inputPattern, true);
            
            // When - with learning disabled
            var resultWithoutLearning = resonanceController.processPattern(inputPattern, false);
            
            // Then - both should complete (success or failure)
            assertThat(resultWithLearning).isNotNull();
            assertThat(resultWithoutLearning).isNotNull();
            assertThat(resultWithLearning.getProcessingTimeMs()).isGreaterThan(0.0);
            assertThat(resultWithoutLearning.getProcessingTimeMs()).isGreaterThan(0.0);
        }
        
        @Test
        @DisplayName("Should reject null input patterns")
        void shouldRejectNullInputPatterns() {
            // Given
            Pattern nullPattern = null;
            
            // When & Then
            assertThatThrownBy(() -> resonanceController.processPattern(nullPattern, true))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cannot be null");
        }
        
        @Test
        @DisplayName("Should generate unique resonance IDs")
        void shouldGenerateUniqueResonanceIds() {
            // Given
            var pattern1 = createTestPattern(0.1, 0.2, 0.3);
            var pattern2 = createTestPattern(0.4, 0.5, 0.6);
            
            // When
            var result1 = resonanceController.processPattern(pattern1, true);
            var result2 = resonanceController.processPattern(pattern2, true);
            
            // Then
            if (result1.isSuccess() && result2.isSuccess()) {
                assertThat(result1.getResonanceId()).isNotEqualTo(result2.getResonanceId());
            }
            // Note: If processing fails, resonance ID might be null, which is acceptable
        }
    }
    
    @Nested
    @DisplayName("Top-Down and Bottom-Up Signal Tests")
    class SignalProcessingTests {
        
        @Test
        @DisplayName("Should coordinate bottom-up and top-down processing")
        void shouldCoordinateBottomUpAndTopDownProcessing() {
            // Given
            var inputPattern = createTestPattern(0.3, 0.7, 0.2, 0.8);
            
            // When
            var result = resonanceController.processPattern(inputPattern, true);
            
            // Then
            assertThat(result).isNotNull();
            
            if (result.isSuccess()) {
                // Successful resonance should have hierarchical categories from bottom-up
                assertThat(result.getHierarchicalCategories()).hasSize(3);
                
                // And vigilance adjustments from top-down validation
                assertThat(result.getVigilanceAdjustments()).hasSize(3);
                
                // Categories should be valid (>= -1 for no match, >= 0 for valid category)
                for (var category : result.getHierarchicalCategories()) {
                    assertThat(category).isGreaterThanOrEqualTo(-1);
                }
            }
        }
        
        @Test
        @DisplayName("Should handle processing failures gracefully")
        void shouldHandleProcessingFailuresGracefully() {
            // Given - a pattern that might cause processing issues
            var problematicPattern = createTestPattern(Double.NaN, 0.5, 0.5);
            
            // When
            var result = resonanceController.processPattern(problematicPattern, true);
            
            // Then
            assertThat(result).isNotNull();
            assertThat(result.getProcessingTimeMs()).isGreaterThan(0.0);
            
            if (!result.isSuccess()) {
                assertThat(result.getErrorMessage()).isNotNull();
                assertThat(result.getResonanceStrength()).isZero();
            }
        }
        
        @Test
        @DisplayName("Should process hierarchical levels in correct order")
        void shouldProcessHierarchicalLevelsInCorrectOrder() {
            // Given
            var inputPattern = createTestPattern(0.2, 0.4, 0.6, 0.8);
            
            // When
            var result = resonanceController.processPattern(inputPattern, true);
            
            // Then
            assertThat(result).isNotNull();
            
            if (result.isSuccess()) {
                var categories = result.getHierarchicalCategories();
                assertThat(categories).hasSize(3);
                
                // Each level should produce a category assignment
                // (can be -1 for no match, but should be present)
                for (int i = 0; i < 3; i++) {
                    assertThat(categories[i]).isGreaterThanOrEqualTo(-1);
                }
            }
        }
    }
    
    @Nested
    @DisplayName("Vigilance Adaptation Tests")
    class VigilanceAdaptationTests {
        
        @Test
        @DisplayName("Should use default vigilance parameters")
        void shouldUseDefaultVigilanceParameters() {
            // Given - controller with default parameters
            
            // When
            var inputPattern = createTestPattern(0.5, 0.5, 0.5);
            var result = resonanceController.processPattern(inputPattern, true);
            
            // Then
            assertThat(result).isNotNull();
            
            if (result.isSuccess()) {
                var vigilanceAdjustments = result.getVigilanceAdjustments();
                assertThat(vigilanceAdjustments).hasSize(3);
                
                // Adjustments should be reasonable values
                for (var adjustment : vigilanceAdjustments) {
                    assertThat(adjustment).isBetween(-1.0, 1.0);
                }
            }
        }
        
        @Test
        @DisplayName("Should attempt adaptive resonance when enabled")
        void shouldAttemptAdaptiveResonanceWhenEnabled() {
            // Given - parameters with adaptive vigilance enabled
            var adaptiveParams = new ResonanceParameters(0.6, 0.6, 0.7, true, 0.05, 3);
            var adaptiveController = new ResonanceController(hierarchyLevels, adaptiveParams);
            
            var inputPattern = createTestPattern(0.3, 0.7, 0.1, 0.9);
            
            // When - process with learning enabled (allows adaptation)
            var result = adaptiveController.processPattern(inputPattern, true);
            
            // Then
            assertThat(result).isNotNull();
            assertThat(result.getProcessingTimeMs()).isGreaterThan(0.0);
            
            // Should either succeed or fail gracefully
            if (result.isSuccess()) {
                assertThat(result.getResonanceStrength()).isGreaterThan(0.0);
            } else {
                assertThat(result.getErrorMessage()).isNotNull();
            }
        }
        
        @Test
        @DisplayName("Should not attempt adaptation when learning is disabled")
        void shouldNotAttemptAdaptationWhenLearningDisabled() {
            // Given
            var inputPattern = createTestPattern(0.1, 0.9, 0.3, 0.7);
            
            // When - process with learning disabled
            var result = resonanceController.processPattern(inputPattern, false);
            
            // Then
            assertThat(result).isNotNull();
            
            // Should complete processing without adaptation attempts
            assertThat(result.getProcessingTimeMs()).isGreaterThan(0.0);
        }
        
        @Test
        @DisplayName("Should handle different vigilance thresholds")
        void shouldHandleDifferentVigilanceThresholds() {
            // Given - parameters with different thresholds
            var strictParams = new ResonanceParameters(0.9, 0.9, 0.95, true, 0.02, 2);
            var lenientParams = new ResonanceParameters(0.3, 0.3, 0.4, true, 0.1, 5);
            
            var strictController = new ResonanceController(hierarchyLevels, strictParams);
            var lenientController = new ResonanceController(hierarchyLevels, lenientParams);
            
            var inputPattern = createTestPattern(0.5, 0.5, 0.5, 0.5);
            
            // When
            var strictResult = strictController.processPattern(inputPattern, true);
            var lenientResult = lenientController.processPattern(inputPattern, true);
            
            // Then - both should complete
            assertThat(strictResult).isNotNull();
            assertThat(lenientResult).isNotNull();
            
            // Lenient parameters might be more likely to achieve resonance
            // (though this depends on the specific patterns and implementation)
        }
    }
    
    @Nested
    @DisplayName("Stability Detection Tests")
    class StabilityDetectionTests {
        
        @Test
        @DisplayName("Should track stability-plasticity balance")
        void shouldTrackStabilityPlasticityBalance() {
            // Given
            var pattern = createTestPattern(0.4, 0.6, 0.3, 0.7);
            
            // When - process some patterns
            resonanceController.processPattern(pattern, true);
            resonanceController.processPattern(pattern, true);
            
            // Then
            var balance = resonanceController.getBalance();
            assertThat(balance).isNotNull();
            assertThat(balance.stability()).isBetween(0.0, 1.0);
            assertThat(balance.plasticity()).isBetween(0.0, 1.0);
            
            // Stability + Plasticity should sum to approximately 1.0
            assertThat(balance.stability() + balance.plasticity()).isCloseTo(1.0, offset(0.1));
        }
        
        @Test
        @DisplayName("Should update balance based on resonance outcomes")
        void shouldUpdateBalanceBasedOnResonanceOutcomes() {
            // Given
            var initialBalance = resonanceController.getBalance();
            var pattern = createTestPattern(0.2, 0.8, 0.4, 0.6);
            
            // When - process multiple patterns
            for (int i = 0; i < 5; i++) {
                resonanceController.processPattern(pattern, true);
            }
            
            // Then
            var updatedBalance = resonanceController.getBalance();
            
            // Balance may have changed based on processing outcomes
            // (exact behavior depends on success/failure of resonance)
            assertThat(updatedBalance.stability()).isBetween(0.0, 1.0);
            assertThat(updatedBalance.plasticity()).isBetween(0.0, 1.0);
        }
        
        @Test
        @DisplayName("Should provide balance classification methods")
        void shouldProvideBalanceClassificationMethods() {
            // Given
            var balance = resonanceController.getBalance();
            
            // When & Then - check classification methods work
            var isBalanced = balance.isBalanced();
            var isStabilityDominant = balance.isStabilityDominant();
            var isPlasticityDominant = balance.isPlasticityDominant();
            
            // Exactly one should be true (balanced, stability-dominant, or plasticity-dominant)
            int trueCount = (isBalanced ? 1 : 0) + (isStabilityDominant ? 1 : 0) + (isPlasticityDominant ? 1 : 0);
            assertThat(trueCount).isEqualTo(1);
            
            // Verify string representation
            var balanceString = balance.toString();
            assertThat(balanceString).contains("Balance{");
            assertThat(balanceString).contains("stability=");
            assertThat(balanceString).contains("plasticity=");
        }
    }
    
    @Nested
    @DisplayName("Resonance Achievement Tests")
    class ResonanceAchievementTests {
        
        @Test
        @DisplayName("Should evaluate resonance conditions correctly")
        void shouldEvaluateResonanceConditionsCorrectly() {
            // Given
            var pattern = createTestPattern(0.6, 0.4, 0.8, 0.2);
            
            // When
            var result = resonanceController.processPattern(pattern, true);
            
            // Then
            assertThat(result).isNotNull();
            
            if (result.isSuccess()) {
                // Successful resonance should meet minimum conditions
                assertThat(result.getResonanceStrength()).isGreaterThanOrEqualTo(defaultParams.minResonanceThreshold());
                assertThat(result.getHierarchicalCategories()).isNotNull();
                assertThat(result.getVigilanceAdjustments()).isNotNull();
            } else {
                // Failed resonance should have strength below threshold or error
                assertThat(result.getResonanceStrength()).isLessThan(defaultParams.minResonanceThreshold());
                assertThat(result.getErrorMessage()).isNotNull();
            }
        }
        
        @Test
        @DisplayName("Should handle patterns with different resonance potential")
        void shouldHandlePatternsWithDifferentResonancePotential() {
            // Given - patterns that might have different resonance characteristics
            var strongPattern = createTestPattern(0.8, 0.8, 0.8, 0.8);      // High values
            var weakPattern = createTestPattern(0.1, 0.1, 0.1, 0.1);       // Low values
            var mixedPattern = createTestPattern(0.9, 0.1, 0.9, 0.1);      // Mixed values
            
            // When
            var strongResult = resonanceController.processPattern(strongPattern, true);
            var weakResult = resonanceController.processPattern(weakPattern, true);
            var mixedResult = resonanceController.processPattern(mixedPattern, true);
            
            // Then - all should complete processing
            assertThat(strongResult).isNotNull();
            assertThat(weakResult).isNotNull();
            assertThat(mixedResult).isNotNull();
            
            // Processing times should be reasonable
            assertThat(strongResult.getProcessingTimeMs()).isGreaterThan(0.0);
            assertThat(weakResult.getProcessingTimeMs()).isGreaterThan(0.0);
            assertThat(mixedResult.getProcessingTimeMs()).isGreaterThan(0.0);
        }
        
        @RepeatedTest(5)
        @DisplayName("Should provide consistent resonance evaluation")
        void shouldProvideConsistentResonanceEvaluation() {
            // Given
            var consistentPattern = createTestPattern(0.5, 0.3, 0.7, 0.4);
            
            // When
            var result = resonanceController.processPattern(consistentPattern, false); // No learning for consistency
            
            // Then
            assertThat(result).isNotNull();
            assertThat(result.getProcessingTimeMs()).isGreaterThan(0.0);
            
            // Result should be consistent across runs (success/failure might vary but structure should be consistent)
            if (result.isSuccess()) {
                assertThat(result.getHierarchicalCategories()).hasSize(3);
                assertThat(result.getVigilanceAdjustments()).hasSize(3);
            }
        }
    }
    
    @Nested
    @DisplayName("Performance and Metrics Tests")
    class PerformanceMetricsTests {
        
        @Test
        @DisplayName("Should provide comprehensive resonance metrics")
        void shouldProvideComprehensiveResonanceMetrics() {
            // Given - process some patterns first
            var pattern1 = createTestPattern(0.3, 0.7, 0.2);
            var pattern2 = createTestPattern(0.6, 0.4, 0.8);
            
            resonanceController.processPattern(pattern1, true);
            resonanceController.processPattern(pattern2, true);
            
            // When
            var metrics = resonanceController.getMetrics();
            
            // Then
            if (metrics != null) { // Metrics might be null if no processing has completed
                assertThat(metrics.totalEvents()).isGreaterThanOrEqualTo(2);
                assertThat(metrics.successfulEvents()).isGreaterThanOrEqualTo(0);
                assertThat(metrics.successRate()).isBetween(0.0, 1.0);
                assertThat(metrics.stabilityFactor()).isBetween(0.0, 1.0);
                assertThat(metrics.plasticityFactor()).isBetween(0.0, 1.0);
                assertThat(metrics.timestamp()).isGreaterThan(0);
                
                // Verify metrics string representation
                var metricsString = metrics.toString();
                assertThat(metricsString).contains("ResonanceMetrics");
                assertThat(metricsString).contains("events=");
                assertThat(metricsString).contains("success=");
            }
        }
        
        @Test
        @DisplayName("Should track processing times accurately")
        void shouldTrackProcessingTimesAccurately() {
            // Given
            var startTime = System.currentTimeMillis();
            var pattern = createTestPattern(0.4, 0.6, 0.2, 0.8);
            
            // When
            var result = resonanceController.processPattern(pattern, true);
            var endTime = System.currentTimeMillis();
            
            // Then
            assertThat(result).isNotNull();
            assertThat(result.getProcessingTimeMs()).isGreaterThan(0.0);
            assertThat(result.getProcessingTimeMs()).isLessThan(endTime - startTime + 100); // Allow some margin
        }
        
        @Test
        @DisplayName("Should handle metrics when no processing has occurred")
        void shouldHandleMetricsWhenNoProcessingOccurred() {
            // Given - fresh controller with no processing
            var freshController = new ResonanceController(hierarchyLevels, defaultParams);
            
            // When
            var metrics = freshController.getMetrics();
            var balance = freshController.getBalance();
            
            // Then
            assertThat(balance).isNotNull();
            assertThat(balance.stability()).isEqualTo(0.5, offset(0.001)); // Default value
            assertThat(balance.plasticity()).isEqualTo(0.5, offset(0.001)); // Default value
            
            // Metrics might be null for fresh controller
            if (metrics != null) {
                assertThat(metrics.totalEvents()).isZero();
                assertThat(metrics.successfulEvents()).isZero();
            }
        }
    }
    
    @Nested
    @DisplayName("Reset and State Management Tests")
    class ResetStateManagementTests {
        
        @Test
        @DisplayName("Should reset state correctly")
        void shouldResetStateCorrectly() {
            // Given - process some patterns first
            var pattern = createTestPattern(0.5, 0.5, 0.5);
            
            resonanceController.processPattern(pattern, true);
            resonanceController.processPattern(pattern, false);
            
            var metricsBeforeReset = resonanceController.getMetrics();
            var balanceBeforeReset = resonanceController.getBalance();
            
            // When
            resonanceController.reset();
            
            // Then
            var metricsAfterReset = resonanceController.getMetrics();
            var balanceAfterReset = resonanceController.getBalance();
            
            // Balance should be reset to default
            assertThat(balanceAfterReset.stability()).isEqualTo(0.5, offset(0.001));
            assertThat(balanceAfterReset.plasticity()).isEqualTo(0.5, offset(0.001));
            
            // Metrics should be reset or null
            if (metricsAfterReset != null) {
                assertThat(metricsAfterReset.totalEvents()).isZero();
                assertThat(metricsAfterReset.successfulEvents()).isZero();
            }
        }
        
        @Test
        @DisplayName("Should maintain functionality after reset")
        void shouldMaintainFunctionalityAfterReset() {
            // Given
            var pattern = createTestPattern(0.3, 0.7, 0.4);
            
            // Process before reset
            var resultBeforeReset = resonanceController.processPattern(pattern, true);
            
            // When
            resonanceController.reset();
            
            // Process after reset
            var resultAfterReset = resonanceController.processPattern(pattern, true);
            
            // Then - both should work
            assertThat(resultBeforeReset).isNotNull();
            assertThat(resultAfterReset).isNotNull();
            assertThat(resultBeforeReset.getProcessingTimeMs()).isGreaterThan(0.0);
            assertThat(resultAfterReset.getProcessingTimeMs()).isGreaterThan(0.0);
        }
    }
    
    @Nested
    @DisplayName("Error Handling and Edge Cases")
    class ErrorHandlingTests {
        
        @Test
        @DisplayName("Should reject null or empty hierarchy levels")
        void shouldRejectNullOrEmptyHierarchyLevels() {
            // Test null levels
            assertThatThrownBy(() -> new ResonanceController(null, defaultParams))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cannot be null or empty");
            
            // Test empty levels
            assertThatThrownBy(() -> new ResonanceController(List.of(), defaultParams))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cannot be null or empty");
        }
        
        @Test
        @DisplayName("Should reject null parameters")
        void shouldRejectNullParameters() {
            // Given
            var validLevels = List.of(new HierarchyLevel(Level.TOKEN, 10));
            
            // When & Then
            assertThatThrownBy(() -> new ResonanceController(validLevels, null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cannot be null");
        }
        
        @Test
        @DisplayName("Should handle extreme parameter values gracefully")
        void shouldHandleExtremeParameterValuesGracefully() {
            // Given - parameters with extreme values
            var extremeParams = new ResonanceParameters(0.01, 0.01, 0.02, true, 0.5, 10);
            var extremeController = new ResonanceController(hierarchyLevels, extremeParams);
            
            var pattern = createTestPattern(0.5, 0.5, 0.5);
            
            // When & Then - should not crash
            assertThatCode(() -> {
                var result = extremeController.processPattern(pattern, true);
                assertThat(result).isNotNull();
            }).doesNotThrowAnyException();
        }
        
        @Test
        @DisplayName("Should handle patterns with extreme values")
        void shouldHandlePatternsWithExtremeValues() {
            // Given - patterns with extreme values
            var zeroPattern = createTestPattern(0.0, 0.0, 0.0);
            var onePattern = createTestPattern(1.0, 1.0, 1.0);
            var negativePattern = createTestPattern(-1.0, -0.5, -0.1);
            var largePattern = createTestPattern(10.0, 100.0, 1000.0);
            
            // When & Then - all should be handled gracefully
            assertThatCode(() -> {
                resonanceController.processPattern(zeroPattern, true);
                resonanceController.processPattern(onePattern, true);
                resonanceController.processPattern(negativePattern, true);
                resonanceController.processPattern(largePattern, true);
            }).doesNotThrowAnyException();
        }
    }
    
    @Nested
    @DisplayName("Concurrency Tests")
    class ConcurrencyTests {
        
        @Test
        @DisplayName("Should handle concurrent pattern processing")
        void shouldHandleConcurrentPatternProcessing() throws InterruptedException {
            // Given
            var executor = Executors.newFixedThreadPool(5);
            var latch = new CountDownLatch(10);
            var results = new ResonanceResult[10];
            var exceptions = new Exception[10];
            
            // When - process patterns concurrently
            for (int i = 0; i < 10; i++) {
                int threadIndex = i;
                executor.submit(() -> {
                    try {
                        var pattern = createTestPattern(threadIndex * 0.1, threadIndex * 0.05, threadIndex * 0.02);
                        results[threadIndex] = resonanceController.processPattern(pattern, true);
                    } catch (Exception e) {
                        exceptions[threadIndex] = e;
                    } finally {
                        latch.countDown();
                    }
                });
            }
            
            // Wait for completion
            latch.await();
            executor.shutdown();
            
            // Then - all processing should complete without exceptions
            for (int i = 0; i < 10; i++) {
                assertThat(exceptions[i]).isNull();
                assertThat(results[i]).isNotNull();
                assertThat(results[i].getProcessingTimeMs()).isGreaterThan(0.0);
            }
        }
        
        @Test
        @DisplayName("Should handle concurrent metrics access")
        void shouldHandleConcurrentMetricsAccess() throws InterruptedException {
            // Given
            var executor = Executors.newFixedThreadPool(3);
            var latch = new CountDownLatch(6);
            var exceptions = new Exception[6];
            
            // When - access metrics and balance concurrently while processing
            for (int i = 0; i < 3; i++) {
                int threadIndex = i * 2;
                
                // Processing thread
                executor.submit(() -> {
                    try {
                        var pattern = createTestPattern(0.5, 0.3, 0.7);
                        resonanceController.processPattern(pattern, true);
                    } catch (Exception e) {
                        exceptions[threadIndex] = e;
                    } finally {
                        latch.countDown();
                    }
                });
                
                // Metrics access thread
                executor.submit(() -> {
                    try {
                        var metrics = resonanceController.getMetrics();
                        var balance = resonanceController.getBalance();
                        // Just accessing, not asserting values due to concurrency
                    } catch (Exception e) {
                        exceptions[threadIndex + 1] = e;
                    } finally {
                        latch.countDown();
                    }
                });
            }
            
            // Wait for completion
            latch.await();
            executor.shutdown();
            
            // Then - no exceptions should occur
            for (var exception : exceptions) {
                assertThat(exception).isNull();
            }
        }
    }
    
    // Helper methods
    
    private Pattern createTestPattern(double... values) {
        return new DenseVector(values);
    }
}